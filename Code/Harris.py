# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 11:12:47 2021

@author: fancy
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os 
import time 

path = r'../Data/'
files = os.listdir(path)
method = 'Harris'


sumTime = 0.
for filename in files:
    savename = method+'_'+filename
    img = cv.imread(path+filename)
    
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    
    tik = time.time()
    dst = cv.cornerHarris(gray,2,3,0.04)
    #result用于标记角点，并不重要
    dst = cv.dilate(dst,None)
    #最佳值的阈值，它可能因图像而异。
    img[dst>0.01*dst.max()]=[0,0,255]
    tok = time.time()
    sumTime += tok-tik
    
    cv.imwrite(savename,img)
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows()

avgtime = sumTime/len(files);
print(method,'> average detector cost',avgtime,'s')
        
