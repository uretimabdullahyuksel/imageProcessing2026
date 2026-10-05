# -*- coding: utf-8 -*-

import cv2
import numpy as np

#resim olustur
img = np.zeros((512,512,3), np.uint8) # siyah bir resim
print(img.shape)

cv2.imshow("siyah", img)

# çizgi
# (resim, baslangıc noktası, bitis nıktası, renk, kalınlık)
cv2.line(img, (100,100), (200,100), (0,255,0), 3) # rgb = (255,0,0) ama opencv de farklı
# opencv de bgr(0,0,255) yaparsak kırmızı olur
cv2.imshow("Cizgi", img)

#dikdörtgen
#(resim, baslangıc noktası, bitis nıktası, renk, içini doldurma)
cv2.rectangle(img, (0,0), (256,256), (255,0,0), cv2.FILLED)
cv2.imshow("Dikdortgen", img)

#ÇEMBER
#(resim, merkez noktası, yarı çap, renk, içini doldurma)
cv2.circle(img, (300,300), (45), (0,0,255), cv2.FILLED)
cv2.imshow("Cember", img)

#yazı
#(resim, baslangıc noktası, font, kalınlık, renk)
cv2.putText(img, "Resim", (350,350), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255))
cv2.imshow("metin", img)