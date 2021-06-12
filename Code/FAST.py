# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 11:02:14 2021

@author: fancy
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os 
import time 


path = r'../Data/'
files = os.listdir(path)
method = 'FAST'

sumTime = 0.
for filename in files:
    savename = method+'_'+filename
    img = cv.imread(path+filename)
    
    tik = time.time()
    # 用默认值初始化FAST对象
    fast = cv.FastFeatureDetector_create(threshold = 50)
    # 寻找并绘制关键点
    kp = fast.detect(img,None)
    tok = time.time()
    sumTime += tok-tik
    
    
    img2 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
    # 打印所有默认参数
    # print( "Threshold: {}".format(fast.getThreshold()) )
    # print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
    # print( "neighborhood: {}".format(fast.getType()) )
    # print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )
    # cv.imwrite('fast_true.png',img2)
    
    cv.imwrite(savename,img2)
    
    # # 关闭非极大抑制
    # fast.setNonmaxSuppression(0)
    # kp = fast.detect(img,None)
    # print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )
    # img3 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
    # # cv.imwrite('fast_false.png',img3)
    
    # cv.imwrite('test2.png',img3)
    
avgtime = sumTime/len(files);
print(method,'> average detector cost',avgtime,'s')
