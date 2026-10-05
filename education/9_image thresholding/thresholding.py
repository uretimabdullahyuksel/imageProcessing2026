# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt

#resmi içe aktar
img = cv2.imread("img1.JPG")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.figure()
plt.title("original")
plt.imshow(img, cmap = "gray")
plt.axis("off")
plt.show()

#eşikleme 55 ile 255 arasını beyaz yapacak
_, thresh_img = cv2.threshold(img, thresh=55, maxval=255, type=cv2.THRESH_BINARY)
plt.figure()
plt.title("gray")
plt.imshow(thresh_img, cmap = "gray")
plt.axis("off")
plt.show()

#adaptive threshold
# resim, max değer, adaptif yöntem, skala, eşik ayarlanacak blok boyutu ,ağırlıklı ortalama
thresh_img2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8)

plt.figure()
plt.title("adaptive")
plt.imshow(thresh_img2, cmap = "gray")
plt.axis("off")
plt.show()

