# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 23:55:54 2021

@author: fancy
"""

import cv2 
import numpy as np 

path = r'../Data/'

img = cv2.imread(path+'EGaudi_1.jpg')

#参数为hessian矩阵的阈值
surf = cv2.xfeatures2d.SURF_create(4000)

#设置是否要检测方向
surf.setUpright(True)

#输出设置值
print(surf.getUpright())

#找到关键点和描述符
key_query,desc_query = surf.detectAndCompute(img,None)

img=cv2.drawKeypoints(img,key_query,img)

#输出描述符的个数
print(surf.descriptorSize())

cv2.imshow('sp',img)
cv2.waitKey(0)