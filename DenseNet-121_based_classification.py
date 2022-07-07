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

# 정상구동

# configurations
path = "wacv2016/dataset"
path_dest = "samples"
split = 0.8
batch_size = 16
seed = 999
n_samples = [228, 412, 412]

# DenseNet-121
# in_channels = 1, n_classes = 3, input_shape = b * 1 * 100 * 100
class BasicConv2d(nn.Module):
    def __init__(self, ksize, inCH, outCH, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.bn = nn.BatchNorm2d(inCH)
        self.conv2d = nn.Conv2d(kernel_size=ksize, in_channels=inCH,
                         out_channels=outCH, stride=stride, padding=padding)

    def forward(self, x):
        x = self.bn(x)
        x = F.relu(x)
        x = self.conv2d(x)
        return x


class BottleNeck(nn.Module):
    def __init__(self, inCH, k=32):
        super(BottleNeck, self).__init__()
        self.conv2d_1x1 = BasicConv2d(ksize=1, inCH=inCH, outCH=4*k)
        self.conv2d_3x3 = BasicConv2d(ksize=3, inCH=4*k, outCH=k, padding=1)

    def forward(self, x):
        left = self.conv2d_1x1(x)
        left = self.conv2d_3x3(left)
        out = torch.cat([x, left], dim=1)
        return out

class DenseBlock(nn.Module):
    def __init__(self, inCH, layernum=6, k=32):
        super(DenseBlock, self).__init__()
        self.layernum = layernum
        self.k = k
        self.inCH = inCH
        self.outCH = inCH + k * layernum
        self.block = self.make_layer(layernum)

    def forward(self, x):
        out = self.block(x)
        return out

    def make_layer(self, layernum):
        layers = []
        inchannels = self.inCH
        for i in range(layernum):
            layers.append(BottleNeck(inCH=inchannels, k=self.k))
            inchannels += self.k
        return nn.Sequential(*layers)


class Transition(nn.Module):
    def __init__(self, inCH, theta=0.5):
        super(Transition, self).__init__()
        self.outCH = int(math.floor(theta*inCH))
        self.bn = nn.BatchNorm2d(inCH)
        self.conv2d_1x1 = nn.Conv2d(kernel_size=1, in_channels=inCH, out_channels=self.outCH)
        self.avgpool = nn.AvgPool2d((2, 2), stride=2)

    def forward(self, x):
        x = self.bn(x)
        x = F.relu(x)
        x = self.conv2d_1x1(x)
        x = self.avgpool(x)
        return x


class DenseNet121(nn.Module):
    def __init__(self, k=32, theta=0.5, classes=10):
        super(DenseNet121 ,self).__init__()
        self.k=k
        self.theta = theta
        self.pre_layer = BasicConv2d(ksize=3, inCH=1, outCH=2*self.k, padding=1)
        self.DenseBlock_1 = DenseBlock(inCH=2*self.k, layernum=6, k=self.k)
        self.Transition_1 = Transition(inCH=self.DenseBlock_1.outCH, theta=self.theta)
        self.DenseBlock_2 = DenseBlock(inCH=self.Transition_1.outCH, layernum=12, k=self.k)
        self.Transition_2 = Transition(inCH=self.DenseBlock_2.outCH, theta=self.theta)
        self.DenseBlock_3 = DenseBlock(inCH=self.Transition_2.outCH, layernum=24, k=self.k)
        self.Transition_3 = Transition(inCH=self.DenseBlock_3.outCH, theta=self.theta)
        self.DenseBlock_4 = DenseBlock(inCH=self.Transition_3.outCH, layernum=16, k=self.k)
        self.bn = nn.BatchNorm2d(self.DenseBlock_4.outCH)
        self.avgpool = nn.AvgPool2d((4, 4))
        self.linear = nn.Linear(self.DenseBlock_4.outCH*3*3, classes)

    def forward(self, x):
        x = self.pre_layer(x)
        x = self.DenseBlock_1(x)
        x = self.Transition_1(x)
        x = self.DenseBlock_2(x)
        x = self.Transition_2(x)
        x = self.DenseBlock_3(x)
        x = self.Transition_3(x)
        x = self.DenseBlock_4(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.avgpool(x)
        x = x.view(-1, self.DenseBlock_4.outCH*3*3)
        x = self.linear(x)
        return x

# generate samples
# use provided samples.zip or run this cell to generate samples from wacv2016-master
if not os.path.exists(path_dest):
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

model = DenseNet121(classes=3)
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

print(cm.print_matrix())
print(cm.ACC)
print(cm.GI)
print(cm.AUC)
print(cm.AGF)
