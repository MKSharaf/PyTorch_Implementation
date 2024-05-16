import torch
import math
from torch.nn.functional import relu, softmax
import torchvision
from torchview import draw_graph

def in_Block(inf, outf):
  block = torch.nn.Sequential(
    torch.nn.Conv2d(inf, outf, kernel_size=3, padding=1),
    torch.nn.ReLU(inplace=True),
    torch.nn.Conv2d(outf, outf, kernel_size=3, padding=1),
    torch.nn.BatchNorm2d(outf),
    torch.nn.ReLU(inplace=True)
  )
  return block

def Block(inf, outf):
  block = torch.nn.Sequential(
    torch.nn.Conv2d(inf, outf, kernel_size=3, padding=1),
    torch.nn.BatchNorm2d(outf),
    torch.nn.ReLU(inplace=True),
    torch.nn.Conv2d(outf, outf, kernel_size=3, padding=1),
    torch.nn.BatchNorm2d(outf),
    torch.nn.ReLU(inplace=True)
  )
  return block
class UNet (torch.nn.Module):
  def __init__(self, n):
    super().__init__()
    
    #Maxpool

    self.max_pool2d = torch.nn.MaxPool2d(kernel_size=2, stride=2)

    # Encoder

    # First set in contracting path
    self.bck1 = in_Block(3, 64)

    self.bck2 = Block(64, 128)

    self.bck3 = Block(128, 256)

    self.bck4 = Block(256, 512)

    self.bck5 = Block(512, 1024)

    # Decoder

    # First set in expansive path
    self.upcl1 = torch.nn.ConvTranspose2d(1024, 512, kernel_size = 2, stride = 2) 

    self.bck6 = Block(1024, 512)
    
    self.upcl2 = torch.nn.ConvTranspose2d(512, 256, kernel_size = 2, stride = 2)

    # Second set in expansive path
    self.bck7 = Block(512, 256)

    self.upcl3 = torch.nn.ConvTranspose2d(256, 128, kernel_size = 2, stride = 2)

    # Third set in expansive path
    self.bck8 = Block(256, 128)

    self.upcl4 = torch.nn.ConvTranspose2d(128, 64, kernel_size = 2, stride = 2)
    
    # Fourth set in expansive path
    self.bck9 = Block(128, 64)

    # Output layer
    self.fl = torch.nn.Conv2d(64, n, 1)


  def forward(self, x):
    
    block1 = self.bck1(x)

    pool1 = self.max_pool2d(block1)

    block2 = self.bck2(pool1)

    pool2 = self.max_pool2d(block2)

    block3 = self.bck3(pool2)

    pool3 = self.max_pool2d(block3)

    block4 = self.bck4(pool3)

    pool4 = self.max_pool2d(block4)

    block5 = self.bck5(pool4)

    upconv1 = self.upcl1(block5)

    block6 = self.bck6(torch.cat((block4, upconv1), 1))

    upconv2 = self.upcl2(block6)

    block7 = self.bck7(torch.cat((block3, upconv2), 1))

    upconv3 = self.upcl3(block7)

    block8 = self.bck8(torch.cat((block2, upconv3), 1))

    upconv4 = self.upcl4(block8)

    block9 = self.bck9(torch.cat((block1, upconv4), 1))

    Output = self.fl(block9)

    return Output
