import torch
from torch.nn.functional import relu

class UNet:
      def __init__(self, n):
        super().__init__()
        
        #Encoder

        #First set in contracting path
        self.cl1_1 = torch.nn.Conv2d(1, 64, 3)
        self.cl1_2 = torch.nn.Conv2d(64, 64, 3)
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        
        #Second set in contracting path
        self.cl2_1 = torch.nn.Conv2d(64, 128, 3)
        self.cl2_2 = torch.nn.Conv2d(128, 128, 3)
        self.pool2 = torch.nn.MaxPool2d(2, 2)

        #Third set in contracting path
        self.cl3_1 = torch.nn.Conv2d(128, 256, 3)
        self.cl3_2 = torch.nn.Conv2d(256, 256, 3)
        self.pool3 = torch.nn.MaxPool2d(2, 2)

        #Fourth set in contracting path
        self.cl4_1 = torch.nn.Conv2d(256, 512, 3)
        self.cl4_2 = torch.nn.Conv2d(512, 512, 3)
        self.pool4 = torch.nn.MaxPool2d(2, 2)

        #Fifth set in contracting path
        self.cl5_1 = torch.nn.Conv2d(512, 1024, 3)
        self.cl5_2 = torch.nn.Conv2d(1024, 1024, 3)
        self.upcl1 = torch.nn.ConvTranspose2d(1024, 512, 2, 2) #This could be considered part of the expansive, aka decoder, but for simplicity it stays here

        #Decoder

        #First set in expansive path
        self.cl6_1 = torch.nn.Conv2d(1024, 512, 3)
        self.cl6_2 = torch.nn.Conv2d(512, 512, 3)
        self.upcl2 = torch.nn.ConvTranspose2d(512, 256, 2, 2)

        #Second set in expansive path
        self.cl7_1 = torch.nn.Conv2d(512, 256, 3)
        self.cl7_2 = torch.nn.Conv2d(256, 256, 3)
        self.upcl3 = torch.nn.ConvTranspose2d(256, 128, 2, 2)

        #Third set in expansive path
        self.cl8_1 = torch.nn.Conv2d(256, 128, 3)
        self.cl8_2 = torch.nn.Conv2d(128, 128, 3)
        self.upcl4 = torch.nn.ConvTranspose2d(128, 64, 2, 2)
        
        #Fourth set in expansive path
        self.cl9_1 = torch.nn.Conv2d(128, 64, 3)
        self.cl9_2 = torch.nn.Conv2d(64, 64, 3)

        #Output layer
        self.fl = torch.nn.Conv2d(64, n, 1)


        def forward(self, x):
            #Encoder

            #Output of first set
            act1_1 = relu(self.cl1_1(x))
            act1_2 = relu(self.cl1_2(act1_1))
            cont1 = self.pool1(act1_2)

            #Output of second set
            act2_1 = relu(self.cl2_1(cont1))
            act2_2 = relu(self.cl2_2(act2_1))
            cont2 = self.pool2(act2_2)

            #Output of third set
            act3_1 = relu(self.cl3_1(cont2))
            act3_2 = relu(self.cl3_2(act3_1))
            cont3 = self.pool3(act3_2)

            #Output of fourth set
            act4_1 = relu(self.cl4_1(cont3))
            act4_2 = relu(self.cl4_2(act4_1))
            cont4 = self.pool4(act4_2)

            #Output of fourth set
            act5_1 = relu(self.cl5_1(cont4))
            act5_2 = relu(self.cl5_2(act5_1))
            cont5 = self.pool5(act5_2)