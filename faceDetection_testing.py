# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 09:45:38 2021

@author: sumitgoyal
"""
import face_recognition
import os
import cv2
import random
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, exp, sqrt
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import csv
from keras.callbacks import ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from keras import callbacks
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))



from keras.layers import MaxPool2D,BatchNormalization,Conv2D,Activation,UpSampling2D,Concatenate 

base_dir = 'archive/'
dims = (224,344)

thres = .5
if os.path.exists(base_dir)==False:
    print("Celeba Dataset is not available in current directory")
    raise SystemExit
images_path = []
for dirpath, dname, filename in os.walk(base_dir+"img_align_celeba\img_align_celeba/"):
    for fname in filename:
        if fname.endswith(".jpg"):
            images_path.append(os.path.join(dirpath, fname))


file = open(base_dir+'list_bbox_celeba.csv')    
csvreader = csv.reader(file)
rows = []
for row in csvreader:
        rows.append(row)
rows = rows[1:]

def getModelBaseResnet():

    base_model = ResNet50(include_top=False)  
     
    x = UpSampling2D()(base_model.layers[44].output)
    x = Concatenate()([x,base_model.layers[38].output])
    
    x = Conv2D(128,(1,1),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(64,(3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = UpSampling2D()(x)
    x = Concatenate()([x,base_model.layers[4].output])
    
    x = Conv2D(64,(1,1),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(32,(3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = UpSampling2D()(x)
    
    x = Conv2D(16,(3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(1,(3,3),padding='same',activation='sigmoid')(x)
    
    
    model = Model(base_model.layers[0].output,x)
    for i in range(45):
        model.layers[i].trainable = True    
    model.summary() 
    return model
def getModelBaseVgg():

    base_model = VGG16(include_top=False)  
     
    x = UpSampling2D()(base_model.layers[11].output)
    x = Concatenate()([x,base_model.layers[6].output])
    
    x = Conv2D(128,(1,1),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(64,(3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = UpSampling2D()(x)
    x = Concatenate()([x,base_model.layers[3].output])
    
    x = Conv2D(64,(1,1),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(32,(3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = UpSampling2D()(x)
    
    x = Conv2D(16,(3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(1,(3,3),padding='same',activation='sigmoid')(x)
    
    
    model = Model(base_model.layers[0].output,x)
    for i in range(12):
        model.layers[i].trainable = True
    model.summary() 
    return model
model = getModelBaseResnet()
#model = getModelBaseVgg()


model.load_weights("FaceDetection_resnet_100.h5")





true_positive = 0
for i in range(0,8000,16):
    X= []
    Y = []
    for j in range(i,i+16):
        img = cv2.imread(images_path[j])
        img = cv2.resize(img,dims)
        X.append(img)
        mask = np.zeros((img.shape[0],img.shape[1]),dtype='uint8')
        mask[int(rows[j][2]):int(rows[j][2])+int(rows[j][4]),int(rows[j][1]):int(rows[j][1])+int(rows[j][3])] = 1
        Y.append(mask)
        
    X = preprocess_input(np.array(X).astype('float32'))
    Y = np.array(Y)
    score = model.predict(X)
    

    score[score<thres] = 0
    score[score>=thres] = 1
    score = score[:,:,:,0].astype('uint8')
    inter = np.sum(score*Y,axis=(1,2))
    union = score+Y
    union[union>0]=1
    union = np.sum(union,axis=(1,2))
    
    true_positive += np.sum(inter/union>=.5)
              
           
