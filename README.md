# H2Former
This repository contains the implementation of our paper "H2Former: An Efficient Hierarchical Hybrid Transformer for Medical Image Segmentation"

## Requirements

python 3.6

numpy 1.16.4

Pytorch 1.8.1

pillow 7.0.0

opencv-python 4.1.0


## Usage

1. Clone the repository, and download the pre-trained ImaenNet model, put them into ./ folder. 
   The details of the training are in train.py file.

2.  And then run the code：python train.py
    Note that the parameters and paths should be set beforehand

4. Once the training is complete, you can run the test.py to test your model.
   Run the code : python test.py.

## LICENSE
 Code can only be used for ACADEMIC PURPOSES. NO COMERCIAL USE is allowed.
 Copyright © College of Computer Science, Nankai University. All rights reserved.

## Note

数据训练的.txt文件只是文件名，代表的是哪些文件是训练集，哪些是测试集，代码中只是给了一个示例，具体的读取还是通过对数据的标签、图像直接读取。
四种病变的标签是放在同一个mask中，跟普通语义分割一样，四种病变用1,2,3,4代表就行，背景用0代表就可以。



