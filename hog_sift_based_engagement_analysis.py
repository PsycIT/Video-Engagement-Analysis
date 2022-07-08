#!/usr/bin/env python
# coding: utf-8

#importing required libraries
import cv2
import numpy as np
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import os
import imutils
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import shutil
from sklearn.model_selection import train_test_split
import pandas as pd
from pycm import *
import pycm
# get_ipython().run_line_magic('matplotlib', 'inline')


sift = cv2.xfeatures2d.SIFT_create(nfeatures =29)


# creating instance of labelencoder
labelencoder = LabelEncoder()


path1=r'C:\Users\occisor\Downloads\Engagement_recognition\wacv2016-master\2'
path2=r'C:\Users\occisor\Downloads\Engagement_recognition\wacv2016-master\3'
dest_path1=r'C:\Users\occisor\Downloads\Engagement_recognition\wacv2016-master\dataset\2'
dest_path2=r'C:\Users\occisor\Downloads\Engagement_recognition\wacv2016-master\dataset\3'

def copy_files(src_path,dest_path):
    files=os.listdir(src_path)
    while True:
        i=random.randint(0,len(os.listdir(src_path))-1)
        img_file=files[i]
        img_path=os.path.join(src_path,img_file)
        img=cv2.imread(img_path)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=Image.fromarray(img)
        frame_draw=img.copy()
        frame=img.resize((128,64), Image.BILINEAR)
        img_dest_path=os.path.join(dest_path,img_file)
        img.save(img_dest_path)
        if len(os.listdir(dest_path))>=412:
            break



path=r'C:\Users\occisor\Downloads\Engagement_recognition\wacv2016-master\dataset'
accu=[]
for k in range(10):
    copy_files(path1,dest_path1)
    copy_files(path2,dest_path2)
    folders=os.listdir(path)
    X=[]
    Y=[]
    for folder in folders:
        folder_path=os.path.join(path,folder)
        for img_file in os.listdir(folder_path):
            img_file_path=os.path.join(folder_path,img_file)
            img=cv2.imread(img_file_path)
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            fd=hog(img, orientations=9, pixels_per_cell=(8, 8), 
                    cells_per_block=(2, 2), multichannel=True)
            fd=np.asarray(fd)
            kps, descs = sift.detectAndCompute(img, None)
            if descs.shape[0]<29:
                continue
            descs=descs[0:29,:]
            descs=descs.reshape(1,3712)
            desc=np.append(fd,descs)
            X.append(desc)
            Y.append(folder)
            
    Y=labelencoder.fit_transform(Y)
    X= pd.DataFrame(X)
    X=X.loc[:,0:4948]
    Y= pd.DataFrame(Y)
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.20)
    svm_clf = SVC(kernel='rbf', random_state = 1)
    svm_clf.fit(X_train,y_train)
    y_pred=svm_clf.predict(X_test)
    y_pred=y_pred.reshape((y_test.shape[0],y_test.shape[1]))
    y_test=np.asarray(y_test)
    count=0
    for i in range(len(y_test)):
        if y_pred[i]==y_test[i]:
            count+=1
    ac=count/len(y_test)
    accu.append(ac)
    shutil.rmtree(dest_path1) 
    shutil.rmtree(dest_path2)
    os.mkdir(dest_path1)
    os.mkdir(dest_path2)
    print(k)


accu

accu.mean()


y_pred = np.asarray(y_pred)
y_pred = y_pred.tolist()


y_test = np.asarray(y_test)
y_test = y_test.tolist()


cm = ConfusionMatrix(y_test, y_pred,digit=5)

print(cm.print_matrix())
print(cm.ACC)
print(cm.GI)
print(cm.AGF)
