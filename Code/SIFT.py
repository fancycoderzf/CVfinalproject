# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 15:42:43 2021

@author: fancy
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

path = r'../Data/'
filename = 'EGaudi'
method = 'BRIEF'

img1 = cv.imread(path+filename+'_'+'1.jpg')          # queryImage
img2 = cv.imread(path+filename+'_'+'2.jpg')     # trainImage


# Initiate SIFT detector
sift = cv.SIFT()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)


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


