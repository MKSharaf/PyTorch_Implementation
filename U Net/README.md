## Introduction
This is my attempt at implementing UNet as it was written in the paper, but with some slight modifications such as adding batch normalization, and adding padding instead of cropping. 

## How to use it
1) Makes sure both the images and masks have the same name to crosspond to each other, ex. Image name: ```img_001```, Mask name: ```img_001```
2) Open the arg file
3) Change the ```Image_Directory```, and ```Mask_Directory``` arguments to your images and masks folder respectively
4) Change the ```Classes``` argument to match the number of classes in your case 
5) Change the rest of the arguements if necessary
6) Open the terminal
7) Navigate to the ```U Net``` folder
8) Write ```python3 train.py``` and that is it

*Note: This implementation will use CUDA if it is avaliable as a defult, so if you want to change to CPU, you will just have to specify in the code*
