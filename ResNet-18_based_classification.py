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
import pycm

# configurations
path = "/content/wacv2016-master/dataset"
path_dest = "/content/samples"
split = 0.8
batch_size = 16
seed = 999
n_samples = [228, 412, 412]

# ResNet-18
# in_channels = 1, n_classes = 3, input_shape = b * 1 * 100 * 100
class ResidualBlock(nn.Module):
    def __init__(self, inCh, outCh, stride):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channels=inCh, out_channels=outCh, 
                            kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(outCh),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=outCh, out_channels=outCh, 
                            kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(outCh)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or inCh != outCh:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=inCh, out_channels=outCh, 
                            kernel_size=1, stride=stride),
                nn.BatchNorm2d(outCh)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, classes=10):
        super(ResNet18, self).__init__()
        self.classes = classes
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer_1 = self.make_layer(ResidualBlock, 64, 64, stride=1)
        self.layer_2 = self.make_layer(ResidualBlock, 64, 128, stride=2)
        self.layer_3 = self.make_layer(ResidualBlock, 128, 256, stride=2)
        self.layer_4 = self.make_layer(ResidualBlock, 256, 512, stride=2)
        self.avgpool = nn.AvgPool2d((3, 3), stride=2)
        self.fc = nn.Linear(512 * 6 * 6, self.classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.avgpool(x)
        x = x.view(-1, 512*6*6)
        x = self.fc(x)
        return x

    def make_layer(self, block, inCh, outCh, stride, block_num=2):
        layers = []
        layers.append(block(inCh, outCh, stride))
        for i in range(block_num - 1):
            layers.append(block(outCh, outCh, 1))
        return nn.Sequential(*layers)

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

model = ResNet18(classes=3)
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
