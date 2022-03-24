# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 16:56:06 2021

@author: sumitg
"""

import cv2
import numpy as np
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import callbacks
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

key_features = []
for i in range(100):
  
    img = cv2.imread(r"\\192.168.4.37\Sumit\Face\img/"+str(i)+".bmp")
    
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
            
# =============================================================================
#             img3 = cv2.drawMatchesKnn(img,kp1,img2,kp2,good,None,flags = 2)
# 
#             plt.imshow(img3),plt.show()
# =============================================================================
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
        
