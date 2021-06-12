# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 11:45:06 2021

@author: fancy
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

path = r'../Data/'
filename = 'EGaudi'
method = 'ORB'

img1 = cv.imread(path+filename+'_'+'1.jpg')          # queryImage
img2 = cv.imread(path+filename+'_'+'2.jpg')     # trainImage


# find the keypoints
det = cv.ORB_create()
kp1 = det.detect(img1)
kp2 = det.detect(img2)

#  descriptors
des = det
kp1,des1 = des.compute(img1,kp1)
kp2,des2 = des.compute(img2,kp2)


# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.match(des1,des2)

# Apply ratio test
# good = []
# for m,n in matches:
#     if m.distance < 0.75*n.distance:
#         good.append([m])
        
# cv.drawMatchesKnn expects list of lists as matches.
# img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

img3 = cv.drawMatches(img1, kp1, img2, kp2, matches,None)

cv.imwrite(method+'_'+filename+'_'+'match.jpg',img3)