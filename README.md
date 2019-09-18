
# Convolutional Recurrent Neural Network (CRNN) for OCR
Steps for OCR:

1.Preprocessing Data

2.Creating Network Architecture(CTC loss function)

3.Training Model

4.Test Model


# Preprocessing
1.Download and unzip the dataset into a folder

2.Preprocess the data: both inputs and outputs

Input:

Read the images and convert them into gray-scale images

Reshape each image to size (128,32)

Expand the dimension of the image from (128,32) to (128,32,1)

Normalize the image pixel values by dividing it with 255


Output:

Read the image file names as the labels of that image

Encode word into digits using a map (‘a’:0, ‘b’:1 …….. ‘z’:26 ......) e.g. "aabb" -> [0,0,1,1] 

Find the maximum length among all words and pad every label to be the same size(max size)


The following codes show how I convert letters into digits and Conversion of image

```
char_list = string.ascii_letters+string.digits
print('char_list:',char_list)
print('total length:', len(char_list))

#read image and convert them into correct size
for i, f_name in enumerate(glob(os.path.join(path,'*/*/*.jpg'))):
    # read input image and convert into gray scale image
    img = cv2.cvtColor(cv2.imread(f_name), cv2.COLOR_BGR2GRAY)   
    # convert each image of shape (32, 128, 1)

    img = cv2.resize(img,(128,32))
    img = np.expand_dims(img , axis = 2)

    # Normalize each image
    img = img/255.

    # get the text from the image
    txt = os.path.basename(f_name).split('_')[1]

    # compute maximum length of the text
    if len(txt) > max_label_len:
        max_label_len = len(txt)

    # split the data into validation and training dataset as 10% and 90% respectively
    if i%10 == 0:     
        val_x.append(img)
        val_y.append(encode_to_labels(txt))
        val_x_len.append(31)
        val_y_len.append(len(txt))
        val_orig_y.append(txt)  
    else:
        train_x.append(img)
        train_y.append(encode_to_labels(txt)) 
        train_x_len.append(31)
        train_y_len.append(len(txt))
        orig_y.append(txt)
        
    # we need to pad output label to max text length
    train_padded_y = pad_sequences(train_y, maxlen=max_label_len, padding='post', value = len(char_list))
    val_padded_y = pad_sequences(val_y, maxlen=max_label_len, padding='post', value = len(char_list))
```

# Network Archtecture


Input shape (32,128,1)

Use CNN to produce feature map

Make feature map compatible with LSTM layer.

Use two Bidirectional LSTM layers each of which has 128 units.
```
# input with shape of height=32 and width=128 
inputs = Input(shape=(32,128,1))
 
# Conv2D: 64 filters, (3,3) kernels, rectified unit, use "same" padding
# Pooling: (2,2) size, stride 2
conv_1 = Conv2D(64, (3,3), activation = 'relu', padding='same')(inputs)
pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)
 
conv_2 = Conv2D(128, (3,3), activation = 'relu', padding='same')(pool_1)
pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)
# And the following convoluation layer would do the same thing

# Batch normalization layer, 
batch_norm_5 = BatchNormalization()(conv_5)
conv_6 = Conv2D(512, (3,3), activation = 'relu', padding='same')(batch_norm_5)
batch_norm_6 = BatchNormalization()(conv_6)
pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)
conv_7 = Conv2D(512, (2,2), activation = 'relu')(pool_6)
squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)
 
# bidirectional LSTM layers with units=128
# we want to return sequences, not the last output
# use dropout 0.2
blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(squeezed)
blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(blstm_1)
 
# our final output has [len(char_list)+1] classes
# we need to use softmax as the activation function
outputs = Dense(len(char_list)+1, activation = 'softmax')(blstm_2)

# model to be used at test time
act_model = Model(inputs, outputs)
```
# Loss Function CTC
```
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
 
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
 
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, labels, input_length, label_length])
```
# Training
we need import data to train model as deep learning
```
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer = 'adam')
filepath="best_model.hdf5"
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [checkpoint]
batch_size = 256
epochs = 10
model.fit(x=[train_x, train_padded_y, train_x_len, train_y_len], y=np.zeros(len(train_x)), batch_size=batch_size, epochs = epochs, validation_data = ([val_x, val_padded_y, val_x_len, val_y_len], [np.zeros(len(val_x))]), verbose = 1, callbacks = callbacks_list)
```
# Testing
we can now test the accuracy of such model by distinctive data sets from training data
```
act_model.load_weights('best_model.hdf5')
prediction = act_model.predict(val_x[:10])
out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
                         greedy=True)[0][0])
i = 0
for x in out:
    print("original_text =  ", val_orig_y[i])
    print("predicted text = ", end = '')
    for p in x:  
        if int(p) != -1:
            print(char_list[int(p)], end = '')       
    print('\n')
    i+=1
```
