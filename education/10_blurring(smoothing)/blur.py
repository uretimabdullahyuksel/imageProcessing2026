# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")


#bluring (detayı azaltır, gürültüyü engeller)
img = cv2.imread("NYC.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(), plt.imshow(img), plt.axis("off"), plt.title("original"), plt.show()

"""
ortalama bulanıklaştırma
"""
blur = cv2.blur(img, ksize=(3,3))
plt.figure(), plt.imshow(blur), plt.axis("off"), plt.title("blur"), plt.show()


"""
gauss bulanıklaştırma
"""
gauss_blur = cv2.GaussianBlur(img, ksize=(3,3), sigmaX=7)
plt.figure(), plt.imshow(gauss_blur), plt.axis("off"), plt.title("gauss"), plt.show()

"""
median bulanıklaştırma
"""
median_blur = cv2.medianBlur(img, ksize=3)
plt.figure(), plt.imshow(median_blur), plt.axis("off"), plt.title("median"), plt.show()



"""Gaussian Blurring 

Gaussian blurring is highly effective in removing Gaussian noise from an image.

Bu yöntemde kutu filtresi yerine Gauss çekirdeği kullanılır. Bu, cv.GaussianBlur () işleviyle yapılır. Pozitif ve tek olması gereken çekirdeğin genişliğini ve yüksekliğini belirtmeliyiz. Sırasıyla sigmaX ve sigmaY X ve Y yönlerindeki standart sapmayı da belirtmeliyiz. Yalnızca sigmaX belirtilirse, sigmaY, sigmaX ile aynı şekilde alınır. Her ikisi de sıfır olarak verilirse, çekirdek boyutundan hesaplanır. Gauss bulanıklığı, bir görüntüden Gauss gürültüsünün giderilmesinde oldukça etkilidir.
"""
gb = cv2.GaussianBlur(img, ksize = (3,3), sigmaX = 7)
plt.figure()
plt.imshow(gb)
plt.axis("off")
plt.title("Gauss Bulanıklaştırma")


"""
Median Blurring: This is highly effective against salt-and-pepper noise in an image.
Burada cv.medianBlur () işlevi, çekirdek alanı altındaki tüm piksellerin medyanını alır 
ve merkezi öğe bu medyan değerle değiştirilir. 
Bu, bir görüntüdeki tuz ve biber gürültüsüne karşı oldukça etkilidir. 
İlginç bir şekilde, yukarıdaki filtrelerde, merkezi eleman, görüntüdeki bir piksel değeri 
veya yeni bir değer olabilen yeni hesaplanmış bir değerdir. 
Ancak medyan bulanıklaştırmada, merkezi öğe her zaman görüntüdeki bazı piksel değerleriyle değiştirilir. 
Gürültüyü etkili bir şekilde azaltır. Çekirdek boyutu pozitif bir tek tamsayı olmalıdır.
"""

mb = cv2.medianBlur(img, ksize = 3)
plt.figure()
plt.imshow(mb)
plt.axis("off")
plt.title("Medyan Bulanıklaştırma")

def gaussianNoise(image):
    
    row,col,ch= image.shape
    mean = 0
    var = 0.05
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    
    return noisy

def saltPepperNoise(image):
    row,col,ch = image.shape
    s_vs_p = 0.5
    amount = 0.004
    noisy = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[coords] = 1

    # Pepper mode
    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[coords] = 0
  
    return noisy

img = cv2.imread("NYC.jpg")
img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255
plt.figure()
plt.imshow(img)
plt.axis("off")
plt.title("Original Image")

gaussianNoisyImage = gaussianNoise(img)
plt.figure()
plt.imshow(gaussianNoisyImage)
plt.axis("off")
plt.title("Image with Gaussian Noise")

spImage = saltPepperNoise(img)
plt.figure()
plt.imshow(spImage)
plt.axis("off")
plt.title("Image with Salt and Pepper Noise")


# gaussian blurring
gb = cv2.GaussianBlur(gaussianNoisyImage, ksize = (3,3), sigmaX = 7)
plt.figure()
plt.imshow(gb)
plt.axis("off")
plt.title("Image with Gaussian Blurring")

# median blurring
mb = cv2.medianBlur(spImage.astype(np.float32), ksize = 3)
plt.figure()
plt.imshow(mb)
plt.axis("off")
plt.title("Image with Median Blurring")