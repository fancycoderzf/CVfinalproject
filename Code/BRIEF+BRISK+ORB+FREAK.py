# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 11:45:06 2021

@author: fancy
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os 
import time
import pandas as pd

path = r'../Data/'
methods = ['BRIEF','ORB','BRISK','FREAK']
files = []
tmp = os.listdir(path)
for f in tmp:
    files.append(f.split('_')[0])
files=np.unique(np.array(files))

AVGdet = []
AVGdes = []
AVGmatch = []
for method in methods:
    detTime = 0.
    desTime = 0.
    matchTime = 0.
    for filename in files:
        savename = method+'_'+filename+'.jpg'
        
        img1 = cv.imread(path+filename+'_'+'1.jpg')  
        img2 = cv.imread(path+filename+'_'+'2.jpg')   
        
        # decide the detectors and descriptors
        if method == 'BRIEF':
            det = cv.xfeatures2d.StarDetector_create()
            des = cv.xfeatures2d.BriefDescriptorExtractor_create()
        elif method == 'ORB':
            det = cv.ORB_create()
            des = det
        elif method == 'BRISK':
            det = cv.BRISK_create()
            des = det
        else :
            det = cv.BRISK_create()
            des = cv.xfeatures2d.FREAK_create()
        
        
        tik = time.time()
        kp1 = det.detect(img1)
        kp2 = det.detect(img2)
        tok = time.time()-tik
        detTime += tok
        
        tik = time.time()
        kp1,des1 = des.compute(img1,kp1)
        kp2,des2 = des.compute(img2,kp2)
        tok = time.time()-tik
        desTime += tok
        
        # BFMatcher with default params
        tik = time.time()
        bf = cv.BFMatcher()
        matches = bf.match(des1,des2)
        tok = time.time()-tik
        matchTime += tok
        # Apply ratio test
        # good = []
        # for m,n in matches:
        #     if m.distance < 0.75*n.distance:
        #         good.append([m])
                
        # cv.drawMatchesKnn expects list of lists as matches.
        # img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        img3 = cv.drawMatches(img1, kp1, img2, kp2, matches,None)
        
        cv.imwrite(savename,img3)
        
    avgDet = (float)(detTime)/(2*len(files))
    avgDes = (float)(desTime)/(2*len(files))
    avgMatch = (float)(matchTime)/(2*len(files))
    AVGdet.append(avgDet)
    AVGdes.append(avgDes)
    AVGmatch.append(avgMatch)                                 
    print(method,'> average detect cost',avgDet,'s')
    print(method,'> average descripe cost',avgDes,'s')
    print(method,'> average match cost',avgMatch,'s')
    print('\n\n')
    
    

mydata = pd.DataFrame()
mydata['method'] = methods
mydata['DETtime'] = AVGdet
mydata['DEStime'] = AVGdes
mydata['MATCHtime'] = AVGmatch
mydata.to_csv('AVGtimes.csv',sep=',',index=None)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    