import os
import torch
import dataset
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from datetime import datetime
import unet as arch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import configparser


def train(
        imgDir, 
        maskDir, 
        batch, 
        classes, 
        learningRate, 
        epoch, 
        split, 
        device):

    data = dataset.Data(img_dir = imgDir, mask_dir = maskDir)

    train_length=int(split* len(data))
    test_length=len(data)-train_length

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset = data,
        lengths = (train_length,test_length),
        generator = torch.Generator(device = device)
        )
    train_dataset = dataset.transformData(train_dataset, transform = v2.Compose([
        v2.ToTensor(),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        v2.RandomRotation(degrees = (-30, 30)),
        v2.RandomAffine(degrees = 0, translate = (0.1, 0.1)),
        v2.ElasticTransform(alpha = 50.0, sigma = 10),
        v2.Resize((512,512))
        ]))
    test_dataset = dataset.transformData(test_dataset, transform = v2.Compose([
        v2.ToTensor(),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        v2.Resize((512,512))
        ]))

    training_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch, 
        shuffle=True, 
        generator=torch.Generator(device=device))
    validation_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch, 
        shuffle=False, 
        generator=torch.Generator(device=device))

    model = arch.UNet(classes)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = learningRate)
    loss_fn = torch.nn.CrossEntropyLoss()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/training_{}'.format(timestamp))

    epoch_number = 0

    EPOCHS = epoch

    best_vloss = 1_000_000.

    for epoch in tqdm(range(EPOCHS)):
        print('EPOCH {}:'.format(epoch_number + 1))

        model.train(True)
        running_loss = 0.
        last_loss = 0.
        value = 0.

        for i, (img, mask) in enumerate(training_loader):

            img = img.to(device)
            mask = mask.to(device, dtype = torch.int64)

            optimizer.zero_grad()

            output = model(img)

            loss = loss_fn(output, mask)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            
            value = i+1
            
        last_loss = running_loss / value 
        print('  batch {} loss: {}'.format(i + 1, last_loss))
        tb_x = epoch_number * len(training_loader) + i + 1
        writer.add_scalar('Loss/train', last_loss, tb_x)

        running_loss = 0.
        value = 0.

        avg_loss = last_loss

        running_vloss = 0.0
        model.eval()

        with torch.no_grad():
            for i, (vimg, vmask) in enumerate(validation_loader):
                vimg = vimg.to(device)
                vmask = vmask.to(device, dtype = torch.int64)
                voutput = model(vimg)
                vloss = loss_fn(voutput, vmask)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.flush()

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}'.format(epoch_number + 1)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)
    config = configparser.ConfigParser()
    config.read("args")
    imgDir = config['args']['Image_Directory']
    maskDir = config['args']['Mask_Directory']
    batch = int(config['args']['Batch_Size'])
    classes = int(config['args']['Classes'])
    learningRate = float(config['args']['Learning_Rate'])
    epoch = int(config['args']['Epochs'])
    split = float(config['args']['Train_Split'])
    print("Device: ", device)
    train(imgDir, maskDir, batch, classes, learningRate, epoch, split, device)
