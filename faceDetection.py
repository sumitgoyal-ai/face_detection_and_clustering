# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 09:45:38 2021

@author: sumitgoyal
"""

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
from keras.callbacks import ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from keras import callbacks
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))



from keras.layers import MaxPool2D,BatchNormalization,Conv2D,Activation,UpSampling2D,Concatenate 



#gaussian kernel creation
s, k = 500, 500 
probs = [exp(-z*z/(2*s*s))/sqrt(2*pi*s*s) for z in range(-k,k+1)] 
kernel = np.outer(probs, probs)
kernel = kernel-np.min(kernel)
maxc = np.max(kernel)
kernel = kernel/maxc
kernel[kernel<.50]=0





#Data Loader
base_dir = 'faces/'
dims = (224,344)
validation_split = .10
thres = .5
if os.path.exists(base_dir)==False:
    print("Face Dataset is not available in current directory")
    raise SystemExit
images_path = []
for dirpath, dname, filename in os.walk(base_dir):
    for fname in filename:
        if fname.endswith(".jpg"):
            images_path.append(os.path.join(dirpath, fname))
mat = scipy.io.loadmat(os.path.join(base_dir, 'ImageData.mat'))




data = []
#mask creation
for i,path in enumerate(images_path):
    img = cv2.imread(path)
    mask = np.zeros((img.shape[0],img.shape[1]),dtype='float32')
    label = mat['SubDir_Data'][:,i]
    xmin = int(np.min([label[0],label[2],label[4],label[6]]))
    xmax = int(np.max([label[0],label[2],label[4],label[6]]))
    ymin = int(np.min([label[1],label[3],label[5],label[7]]))
    ymax = int(np.max([label[1],label[3],label[5],label[7]]))
    mask[ymin:ymax,xmin:xmax] = cv2.resize(kernel,(xmax-xmin,ymax-ymin))
    img = cv2.resize(img,dims)
    mask = cv2.resize(mask,dims)
    
    data.append([img,mask])



#data split in training and validation
random.seed(0)
random.shuffle(data)

nTotal = len(data)
nValidation = int(validation_split*nTotal)
nTraining = nTotal-nValidation
train_X = []
train_Y = []
for dat in data[:nTraining]:
    train_X.append(dat[0])
    train_Y.append(dat[1])
Val_X = []
Val_Y = []
for dat in data[nTraining:]:
    Val_X.append(dat[0])
    Val_Y.append(dat[1])
    
train_X = preprocess_input(np.array(train_X).astype('float32'))
Val_X = preprocess_input(np.array(Val_X).astype('float32'))

train_Y = np.expand_dims(np.array(train_Y),axis=-1)
Val_Y = np.expand_dims(np.array(Val_Y),axis=-1)



#Resenet based UNET
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
        model.layers[i].trainable = False    
    model.summary() 
    return model

#VGG based UNET
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
        model.layers[i].trainable = False
    model.summary() 
    return model
model = getModelBaseResnet()
model = getModelBaseVgg()
class PlotLosses(callbacks.Callback):
    def __init__(self):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []

        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

        self.i += 1
        
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();
# =============================================================================
# #training 
# plot_losses = PlotLosses()
# model.compile(optimizer="Adam", loss="mse")
# checkpoint = ModelCheckpoint("FaceDetection_resnet_bf.h5",monitor='val_loss',verbose=0, save_best_only=True, mode='auto')
# 
# model.fit(train_X,train_Y,validation_data=(Val_X,Val_Y),epochs=200,batch_size=32,callbacks = [plot_losses, checkpoint])
# model.save_weights("FaceDetection_resnet_bf_100.h5")
# 
# =============================================================================


#Testing
model.load_weights("FaceDetection_vgg_bf_100.h5")
true_positive = 0
total_sample = 0
for i in range(0,nTraining,16):
    Y = train_Y[i:i+16]
    Y[Y<thres] = 0
    Y[Y>=thres] = 1
    Y = Y[:,:,:,0].astype('uint8')
    score = model.predict(train_X[i:i+16])
    

    score[score<thres] = 0
    score[score>=thres] = 1
    score = score[:,:,:,0].astype('uint8')
    inter = np.sum(score*Y,axis=(1,2))
    union = score+Y
    union[union>0]=1
    union = np.sum(union,axis=(1,2))
    
    true_positive += np.sum(inter/union>=.7)
    total_sample +=16

print("training accuracy",true_positive/total_sample)
true_positive = 0
total_sample = 0
for i in range(0,nValidation,16):
    Y = Val_Y[i:i+16]
    Y[Y<thres] = 0
    Y[Y>=thres] = 1
    Y = Y[:,:,:,0].astype('uint8')
    score = model.predict(Val_X[i:i+16])
    

    score[score<thres] = 0
    score[score>=thres] = 1
    score = score[:,:,:,0].astype('uint8')
    inter = np.sum(score*Y,axis=(1,2))
    union = score+Y
    union[union>0]=1
    union = np.sum(union,axis=(1,2))
    
    true_positive += np.sum(inter/union>=.7)
    total_sample +=16

print("validation accuracy",true_positive/total_sample)