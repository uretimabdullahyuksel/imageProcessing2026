# -*- coding: utf-8 -*-

import cv2

img = cv2.imread("Lenna.png",0)
print("resim boyutu: ", img.shape)
cv2.imshow("original", img)

#resize
imgResized = cv2.resize(img, (800,600))
print("resized img shape: ", imgResized.shape)
cv2.imshow("img resized", imgResized)

#crop
imgCropped= img[:200,:300]#height , weight
cv2.imshow("kirpik Resim", imgCropped)