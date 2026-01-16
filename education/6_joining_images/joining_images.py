# -*- coding: utf-8 -*-

import cv2
import numpy as np

#resmi ice aktar
img = cv2.imread("Lenna.png")
cv2.imshow("original", img)

#horizontal birlestirme
hor = np.hstack((img,img))
cv2.imshow("Yatay", hor)

#vertical birlestirme
ver = np.vstack((img,img))
cv2.imshow("Dikey", ver)