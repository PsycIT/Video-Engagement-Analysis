import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torch import optim
from PIL import Image
import numpy as np
import random
import os
import math
import pycm

# configurations
path = "/content/wacv2016-master/dataset"
path_dest = "/content/samples"
split = 0.8
batch_size = 16
seed = 999
n_samples = [228, 412, 412]

# MobileNetV1
# in_channels = 1, n_classes = 3, input_shape = b * 1 * 100 * 100
class BasicConv2d(nn.Module):
    def __init__(self, ksize, inCH, outCH, padding=0, stride=1):
        super(BasicConv2d, self).__init__()
        self.conv2d = nn.Conv2d(kernel_size=ksize, in_channels=inCH, 
                        out_channels=outCH, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(outCH)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DepthwiseConv2d(nn.Module):
    def __init__(self, ksize, inCH, outCH, padding=0, stride=1):
        super(DepthwiseConv2d, self).__init__()
        self.dwConv2d = nn.Conv2d(kernel_size=ksize, in_channels=inCH, 
                            out_channels=inCH, stride=stride, padding=padding, groups=inCH)
        self.bn = nn.BatchNorm2d(inCH)
        self.relu = nn.ReLU(inplace=True)
        self.pointwiseConv2d = BasicConv2d(ksize=1, inCH=inCH, outCH=outCH)

    def forward(self, x):
        x = self.dwConv2d(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pointwiseConv2d(x)
        return x


class MobileNet(nn.Module):
    def __init__(self, classes=10):
        super(MobileNet, self).__init__()
        self.pre_layer = BasicConv2d(ksize=3, inCH=1, outCH=32)
        self.Depthwise = nn.Sequential(
            DepthwiseConv2d(ksize=3, inCH=32, outCH=64, padding=1),
            DepthwiseConv2d(ksize=3, inCH=64, outCH=128, stride=2, padding=1),
            DepthwiseConv2d(ksize=3, inCH=128, outCH=128, padding=1),
            DepthwiseConv2d(ksize=3, inCH=128, outCH=256, padding=1),
            DepthwiseConv2d(ksize=3, inCH=256, outCH=256, padding=1),
            DepthwiseConv2d(ksize=3, inCH=256, outCH=512, stride=2, padding=1),
            DepthwiseConv2d(ksize=3, inCH=512, outCH=512, padding=1),
            DepthwiseConv2d(ksize=3, inCH=512, outCH=512, padding=1),
            DepthwiseConv2d(ksize=3, inCH=512, outCH=512, padding=1),
            DepthwiseConv2d(ksize=3, inCH=512, outCH=512, padding=1),
            DepthwiseConv2d(ksize=3, inCH=512, outCH=512, padding=1),
            DepthwiseConv2d(ksize=3, inCH=512, outCH=1024, stride=2, padding=1),
            DepthwiseConv2d(ksize=3, inCH=1024, outCH=1024, padding=1)
        )
        self.avgpool = nn.AvgPool2d((4, 4))
        self.linear = nn.Linear(1024*3*3, classes)

    def forward(self, x):
        x = self.pre_layer(x)
        x = self.Depthwise(x)
        x = self.avgpool(x)
        x = x.view(-1, 3*3*1024)
        x = self.linear(x)
        return x

# generate samples
# use provided samples.zip or run this cell to generate samples from wacv2016-master
os.mkdir(path_dest)
file_num = []
for i in range(1, 4):
    j = 0
    for file in os.listdir(os.path.join(path, str(i))):
        img = Image.open(os.path.join(path, str(i), file))
        if j < 412:
            if img.size[0] == 100 and img.size[1] == 100 and len(img.size) == 2:
                j += 1
                img.save(os.path.join(path_dest, "{}_{}.jpg".format(i-1, j)))
        else:
            break
    file_num.append(j)
print(file_num)

# dataset split
random.seed(seed)
file_list = []
for i, n in enumerate(n_samples):
  for j in range(n):
    file_list.append("{}_{}.jpg".format(i, j+1))
random.shuffle(file_list)
train_list = file_list[:int(split * len(file_list))]
val_list = file_list[int(split * len(file_list)):]

# dataset class
class WACV2016(Dataset):
    def __init__(self, path, file_list):
        super().__init__()
        self.trans = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        self.path = path
        self.file_names = file_list

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.path, self.file_names[index]))
        label = int(self.file_names[index].split(".")[0].split("_")[0])
        return self.trans(img), label

# ...
train_set = WACV2016(path_dest, train_list)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_set = WACV2016(path_dest, val_list)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

model = MobileNet(classes=3)
model = model.cuda()
citeration = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# training
model.train()
for epoch in range(10):
    train_loss = 0
    train_acc = 0
    for i, (image, label) in enumerate(train_loader):
        image, label = image.cuda(), label.cuda()
        optimizer.zero_grad()
        # forward
        output = model(image)
        loss = citeration(output, label.long())
        pred = torch.torch.argmax(output, 1)
        acc = (pred == label).sum() / batch_size
        train_loss += loss.item()
        train_acc += acc.cpu().numpy()
        # backward
        loss.backward()
        optimizer.step()
    print("Epoch: {}  Loss: {}  Acc: {}".format(
        epoch+1, train_loss/len(train_loader), train_acc/len(train_loader)))

# validation
model.eval()
y_pred = []
y_true = []
for i, (image, label) in enumerate(val_loader):
        image, label = image.cuda(), label.cuda()
        # forward
        output = model(image)
        pred = torch.argmax(output, 1).unsqueeze(0)
        y_pred.append(int(pred.cpu().numpy()))
        y_true.append(int(label.unsqueeze(0).cpu().numpy()))
print(y_pred)
print(y_true)

cm = pycm.ConfusionMatrix(y_true, y_pred, digit=5)

cm.ACC

cm.GI

cm.AUC

cm.AGF
