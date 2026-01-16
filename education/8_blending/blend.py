# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt

#karıştırma
img1 = cv2.imread("img1.JPG")
img2 = cv2.imread("img2.JPG")

plt.figure()
plt.imshow(img1)

plt.figure()
plt.imshow(img2)

# resimler bgr a göre yüklendiği ve gösterildiği için farklı

img1 = cv2.imread("img1.JPG")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread("img2.JPG")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(img1)

plt.figure()
plt.imshow(img2)


print(img1.shape)
print(img2.shape)

#boyutlar aynı olmazsa birleştirmede saçma sapan görüntüler ortaya çıkar
img1 = cv2.resize(img1, (600,600))
print(img1.shape)
img2 = cv2.resize(img2, (600,600))
print(img1.shape)

plt.figure()
plt.imshow(img1)

plt.figure()
plt.imshow(img2)

#karıştırılmış resim = alpha* img1 + beta*img2
blended = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
plt.figure()
plt.imshow(blended)