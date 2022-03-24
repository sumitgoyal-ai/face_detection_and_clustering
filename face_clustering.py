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
from keras.callbacks import ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from keras import callbacks
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))



from keras.layers import MaxPool2D,BatchNormalization,Conv2D,Activation,UpSampling2D,Concatenate 
s, k = 500, 500 
probs = [exp(-z*z/(2*s*s))/sqrt(2*pi*s*s) for z in range(-k,k+1)] 
kernel = np.outer(probs, probs)
kernel = kernel-np.min(kernel)
maxc = np.max(kernel)
kernel = kernel/maxc
kernel[kernel<.50]=0

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
    
train_X1 = preprocess_input(np.array(train_X).astype('float32'))
Val_X1 = preprocess_input(np.array(Val_X).astype('float32'))

train_Y1 = np.expand_dims(np.array(train_Y),axis=-1)
Val_Y1 = np.expand_dims(np.array(Val_Y),axis=-1)

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


score_train = model.predict(train_X1)
score_train[score_train<thres] = 0
score_train[score_train>=thres] = 1


sift = cv2.SIFT()
bf = cv2.BFMatcher()

key_features = []

for i in range(nTraining):
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(score_train[i].astype(np.uint8), connectivity=4)
    for k in range(1,nLabels):
        size = stats[k, cv2.CC_STAT_AREA]        
        if size>100:
            x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
            w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
            img  = cv2.cvtColor(train_X[i][y:y+h,x:x+w],cv2.COLOR_BGR2GRAY)
            
            kp1, des1 = sift.detectAndCompute(img,None)
            if des1 is None:
                print(i)
            bConsider = False
            avg = []
            maxx = []
            for des in key_features:
                scores = []
                for d in des:
                    des2,kp2,img2 = d
                    matches = bf.knnMatch(des1,des2, k=2)
                    good = []
                    for m,n in matches:
                        if m.distance < 0.8*n.distance:
                            good.append([m])
                    if len(matches)==0:
                        scores.append(0.0)
                    else:
                        scores.append(len(good)/len(matches))
                    

                avg_score = np.average(scores)
                max_score = np.max(scores)
                
                
                avg.append(avg_score)
                maxx.append(max_score)
            if len(avg)>0:
                armax_avg = np.argmax(maxx)
                if maxx[armax_avg]>=.20:
                    bConsider = True
                    key_features[armax_avg].append((des1,kp1,img)) 
            if bConsider == False:
                key_features.append([(des1,kp1,img)]) 
                print(len(key_features),i)
    
for cat in range(len(key_features)):
    print(cat)
    rows = (len(key_features[cat])/3)+1
    fig = plt.figure(figsize=(rows, 3))
               
    for i in range(len(key_features[cat])):
        fig.add_subplot(rows, 3, i+1)
        plt.imshow(key_features[cat][i][2])
# =============================================================================
#         plt.imshow(key_features[cat][i][2])
#         plt.show()
# =============================================================================
        

           
