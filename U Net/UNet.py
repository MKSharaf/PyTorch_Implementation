import torch
import math
from torch.nn.functional import relu

def crop_tensor(tensor_cont, tensor_exp):
  tensor_cont_size = tensor_cont.size()[2]
  tensor_exp_size = tensor_exp.size()[2]
  dimension = tensor_cont_size - tensor_exp_size # Getting the difference between the size of the 2 images (ex: 280 - 200 = 80)
  dimension = dimension // 2 
  """
  '//' is used to round to the nearest integer number to ensure no fractions. 
  We also divide by 2 because an image is 2D which means an 80 difference is a difference of 40 in both rows and columns
  """
  return tensor_cont[:, :, dimension: tensor_cont_size - dimension, dimension: tensor_cont_size - dimension]
  """
  To understand why we start by the divided dimension, let's go back to our 280x280 image example. 
  Let's say we have a 280x280 image that we want to turn into a 200x200 image. there are an excessive 80 pixels at both dimensions, in other word,
  there are 80 pixels more in width and height than we need, totalling 160 pixels. Now, to solve this issue we just have to start 40 points from the start 
  of the image and end 40 points before the end of the image in both dimensions. This ensures that we lose 80 points in width, and 80 points in height
  """

class UNet (torch.nn.Module):
  def __init__(self, n):
    super().__init__()
    
    # Encoder

    # First set in contracting path
    self.cl1_1 = torch.nn.Conv2d(1, 64, 3)
    self.cl1_2 = torch.nn.Conv2d(64, 64, 3)
    self.pool1 = torch.nn.MaxPool2d(2, 2)
    
    # Second set in contracting path
    self.cl2_1 = torch.nn.Conv2d(64, 128, 3)
    self.cl2_2 = torch.nn.Conv2d(128, 128, 3)
    self.pool2 = torch.nn.MaxPool2d(2, 2)

    # Third set in contracting path
    self.cl3_1 = torch.nn.Conv2d(128, 256, 3)
    self.cl3_2 = torch.nn.Conv2d(256, 256, 3)
    self.pool3 = torch.nn.MaxPool2d(2, 2)

    # Fourth set in contracting path
    self.cl4_1 = torch.nn.Conv2d(256, 512, 3)
    self.cl4_2 = torch.nn.Conv2d(512, 512, 3)
    self.pool4 = torch.nn.MaxPool2d(2, 2)

    # Fifth set in contracting path
    self.cl5_1 = torch.nn.Conv2d(512, 1024, 3)
    self.cl5_2 = torch.nn.Conv2d(1024, 1024, 3)
    self.upcl1 = torch.nn.ConvTranspose2d(1024, 512, 2, 2) # This could be considered part of the expansive, aka decoder, but for simplicity it stays here

    # Decoder

    # First set in expansive path
    self.cl6_1 = torch.nn.Conv2d(1024, 512, 3)
    self.cl6_2 = torch.nn.Conv2d(512, 512, 3)
    self.upcl2 = torch.nn.ConvTranspose2d(512, 256, 2, 2)

    # Second set in expansive path
    self.cl7_1 = torch.nn.Conv2d(512, 256, 3)
    self.cl7_2 = torch.nn.Conv2d(256, 256, 3)
    self.upcl3 = torch.nn.ConvTranspose2d(256, 128, 2, 2)

    # Third set in expansive path
    self.cl8_1 = torch.nn.Conv2d(256, 128, 3)
    self.cl8_2 = torch.nn.Conv2d(128, 128, 3)
    self.upcl4 = torch.nn.ConvTranspose2d(128, 64, 2, 2)
    
    # Fourth set in expansive path
    self.cl9_1 = torch.nn.Conv2d(128, 64, 3)
    self.cl9_2 = torch.nn.Conv2d(64, 64, 3)

    # Output layer
    self.fl = torch.nn.Conv2d(64, n, 1)


  def forward(self, x):
    # Encoder

    # Output of first set
    actc1_1 = relu(self.cl1_1(x))
    actc1_2 = relu(self.cl1_2(actc1_1))
    cont1 = self.pool1(actc1_2)

    # Output of second set
    actc2_1 = relu(self.cl2_1(cont1))
    actc2_2 = relu(self.cl2_2(actc2_1))
    cont2 = self.pool2(actc2_2)

    # Output of third set
    actc3_1 = relu(self.cl3_1(cont2))
    actc3_2 = relu(self.cl3_2(actc3_1))
    cont3 = self.pool3(actc3_2)

    # Output of fourth set
    actc4_1 = relu(self.cl4_1(cont3))
    actc4_2 = relu(self.cl4_2(actc4_1))
    cont4 = self.pool4(actc4_2)

    # Output of fourth set
    actc5_1 = relu(self.cl5_1(cont4))
    actc5_2 = relu(self.cl5_2(actc5_1))
    exp1 = self.upcl1(actc5_2)

    # Decoder

    # Output of first set
    acte1_1 = relu(self.cl6_1(torch.cat((exp1, crop_tensor(actc4_2, exp1)), 1)))
    acte1_2 = relu(self.cl6_2(acte1_1))
    exp2 = self.upcl2(acte1_2)

    # Output of second set
    acte2_1 = relu(self.cl7_1(torch.cat((exp2, crop_tensor(actc3_2, exp2)), 1)))
    acte2_2 = relu(self.cl7_2(acte2_1))
    exp3 = self.upcl3(acte2_2)

    # Output of third set
    acte3_1 = relu(self.cl8_1(torch.cat((exp3, crop_tensor(actc2_2, exp3)), 1)))
    acte3_2 = relu(self.cl8_2(acte3_1))
    exp4 = self.upcl4(acte3_2)

    # Output of fourth set
    acte4_1 = relu(self.cl9_1(torch.cat((exp4, crop_tensor(actc1_2,exp4)), 1)))
    acte4_2 = relu(self.cl9_2(acte4_1))
    
    Output = self.fl(acte4_2)

    return Output
