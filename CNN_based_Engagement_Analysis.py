#!/usr/bin/env python
# coding: utf-8

# data 중 size 다른 image shape 에러 발생
#

from pycm import *
import os
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import random
import struct
import torch
import shutil
from torchvision import transforms,datasets
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import PIL
from skimage.feature import hog
from PIL import Image,ImageDraw
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data.sampler import SubsetRandomSampler
import torch.backends.cudnn as cudnn
import pycm

torch.backends.cudnn.deterministic = True

one_hot=OneHotEncoder()


def copy_files(src_path,dest_path):
    files=os.listdir(src_path)
    while True:
        if len(os.listdir(dest_path))>=412:
            break
        i=random.randint(0,len(os.listdir(src_path))-1)
        img_file=files[i]
        img_path=os.path.join(src_path,img_file)
        img=cv2.imread(img_path)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=Image.fromarray(img)
        frame_draw=img.copy()
        frame=img.resize((100,100), Image.BILINEAR)
        img_dest_path=os.path.join(dest_path,img_file)
        frame.save(img_dest_path)



def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes) 
    return y[labels] 



labels=torch.LongTensor([0])
labels=one_hot_embedding(labels, 3)
labels=labels.reshape(1,3)
labels.shape


accu=[]
for k in range(1): 
    i=random.randint(1,1000)
    # path1='wacv2016/dataset/1'
    # path2='wacv2016/dataset/2'
    # path3='wacv2016/dataset/3'
    dest_path1='wacv2016/1'
    dest_path2='wacv2016/2'
    dest_path3='wacv2016/3'

    # path1=r'C:\Users\occisor\Downloads\Engagement_recognition\wacv2016-master\1'
    # path2=r'C:\Users\occisor\Downloads\Engagement_recognition\wacv2016-master\2'
    # path3=r'C:\Users\occisor\Downloads\Engagement_recognition\wacv2016-master\3'
    # dest_path1=r'C:\Users\occisor\Downloads\Engagement_recognition\wacv2016-master\dataset\1'
    # dest_path2=r'C:\Users\occisor\Downloads\Engagement_recognition\wacv2016-master\dataset\2'
    # dest_path3=r'C:\Users\occisor\Downloads\Engagement_recognition\wacv2016-master\dataset\3'

    # copy_files(path1,dest_path1)
    # copy_files(path2,dest_path2)
    # copy_files(path3,dest_path3)
    
    # path=r'C:\Users\occisor\Downloads\Engagement_recognition\wacv2016-master\dataset'
    path='wacv2016/dataset'

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    image_datasets =datasets.ImageFolder(path,data_transforms['train'])

    batch_size = 1
    validation_split = .2
    shuffle_dataset = True
    random_seed= i

    # Creating data indices for training and validation splits:
    dataset_size = len(image_datasets)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(i)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, 
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size,
                                                    sampler=valid_sampler)

    class LeNet(nn.Module):
        def __init__(self):
            super(LeNet, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, kernel_size=3,padding=2)
            self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
            self.conv2 = nn.Conv2d(6, 9, kernel_size=3,padding=2)  
            self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
            self.conv3 = nn.Conv2d(9,12, kernel_size=3,padding=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2,stride=2)
            self.fc1 = nn.Linear(2352, 3)
            

        def forward(self, x):
            x=F.relu(self.conv1(x))
            x=self.pool1(x)
            x=F.relu(self.conv2(x))
            x=self.pool2(x)
            x=F.relu(self.conv3(x))
            x=self.pool3(x)
            x=x.view((-1,2352))
            # x=x.view((x.size(0),-1))
            x=F.relu(self.fc1(x))
            x=F.softmax(x)
            return x

    net=LeNet()
    net=net.double()
    net=net.cuda()


    InputData=Variable(torch.Tensor(1,3,100,100))
    InputData=InputData.double()
    InputData=InputData.cuda()
    output=net(InputData)

    criterion=nn.MSELoss() 
    optimizer = optim.Adam(net.parameters(), lr=1e-4) # Adam

    
    for epoch in tqdm(range(5)):
        for data in train_loader:
            inputs,labels=data
            inputs=inputs.double()
            inputs=Variable(inputs.cuda())
            labels=torch.LongTensor([labels])
            labels=one_hot_embedding(labels,3)
            labels=labels.reshape(1,3)
            labels=labels.double()
            labels=Variable(labels.cuda())
            net.zero_grad()

            print(inputs)
            output=net(inputs)
            loss=criterion(labels,output)
            loss.backward()
            optimizer.step()
            
    count=0
    y_label = []
    y_pred = []
    net=net.eval()
    for data in validation_loader:
        inputs,labels=data
        inputs=inputs.double()
        inputs=Variable(inputs.cuda())
        labels=torch.LongTensor([labels])
        labels=one_hot_embedding(labels,3)
        labels=labels.reshape(1,3)
        labels=labels.double()
        labels=Variable(labels.cuda())
        output=net(inputs)
        y_pred.append(output.argmax())
        y_label.append(labels.argmax())
        print(output,end="")
        print("output's max arguement is {}".format(output.argmax()),end=" ")
        print("labels max arguement is {}".format(labels.argmax()))
        if output.argmax()==labels.argmax():
            count+=1
        
    ac=count/len(valid_sampler)
    accu.append(ac)
    shutil.rmtree(dest_path1) 
    shutil.rmtree(dest_path2)
    os.mkdir(dest_path1)
    os.mkdir(dest_path2)
    torch.cuda.empty_cache()


y_pred = np.asarray(y_pred)
y_pred = y_pred.tolist()


y_label = np.asarray(y_label)
y_label = y_label.tolist()



cm = ConfusionMatrix(y_label, y_pred,digit=5)


cm.ACC

cm.GI

cm.AGF

torch.save(net.state_dict(), 'Enagagement_analysis_37.pth')

