
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

```char_list = string.ascii_letters+string.digits
print('char_list:',char_list)
```print('total length:', len(char_list))
