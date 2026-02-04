# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam



from tensorflow.keras.callbacks import EarlyStopping
from collections import Counter

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

import warnings
from sklearn.utils.class_weight import compute_class_weight


early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)


warnings.filterwarnings("ignore")

batchSize=50
#eğitime girecek resim partisi sayısı
epochSize=30
#resimler kaç kere eğitileceği

#preprocess

def preProcess(img):
    #siyah beyaza ceviriyoruz
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #histogramı 0-255 yaptık
    img = cv2.equalizeHist(img)
    #görüntünün piksel değerlerini 0-1 arasına getirdik
    img = img / 255
    #son hali geri döndürdük
    return img

path = "specialData"
#hata datamızın yolunu belirtiyoruz

myList = os.listdir(path)
#hata datamızı listeye atıyoruz

numOfFails = len(myList)
#hata listemizin uzunluğunu alıyoruz

print("Hata tür sayısı: ",numOfFails)
#kac hata turumuz var alıyoruz

images = []
imageType = []
#resim ve türler için dizi oluşturuyoruz

for i in range(numOfFails): # klasör içinde dolaşıyoruz
    myImageList = os.listdir(path + "\\" + str(i)) # türlerin isimlerini aldık
    for j in myImageList: # türler içinde dolaşıyoruz
        img = cv2.imread(path + "\\" + str(i) + "\\" + j) # resiölerin isimlerini aldık
        img = cv2.resize(img, (32,32)) # resimleri 32,32 yaptık
        images.append(img) #images listesine ekledik
        imageType.append(i) # tür listesine ekledik



#görselleştirme        
# =============================================================================
# print(len(images))
# print(len(imageType))
# #sayıları gördük
# =============================================================================

images = np.array(images)
imageType = np.array(imageType)

import random
for c in range(numOfFails):
    idx = np.where(imageType == c)[0]
    picks = random.sample(list(idx), 5)
    for k, p in enumerate(picks):
        cv2.imshow(f"class {c} - {k}", images[p])
        cv2.waitKey(300)
cv2.destroyAllWindows()


# preprocess sonrası kontrol
for c in range(numOfFails):
    idx = np.where(imageType == c)[0]
    p = idx[0]
    raw = images[p]
    proc = preProcess(raw)
    cv2.imshow(f"raw class {c}", cv2.resize(raw, (200,200)))
    cv2.imshow(f"proc class {c}", cv2.resize((proc*255).astype(np.uint8), (200,200)))
    cv2.waitKey(800)
cv2.destroyAllWindows()


# geri kalan işlemlerde array a çevirmemiz gerekiyor
print("Görüntü sayısı: ",len(images))
print("Görüntü tür sayısı: ",len(imageType))
print("Hata tür sayısı: ",numOfFails)
print(Counter(imageType))
#veriyi ayırma

# x_train, x_test, y_train, y_test = train_test_split(images, 
#                                                     imageType, 
#                                                     test_size = 0.5, 
#                                                     random_state = 42)
# #train ile test verilerimizi yarı yarıya ayırdık. random state 42 parametresi deneme ile bulduk.
# x_train, x_validation, y_train, y_validation = train_test_split(x_train, 
#                                                                 y_train, 
#                                                                 test_size = 0.2, 
#                                                                 random_state = 42)
# #train ile test verilerimizi %20-%80 ayırdık. random state 42 parametresi deneme ile bulduk.

#train ile validation verilerimizi yarı yarıya ayırdık. random state 42 parametresi deneme ile bulduk.
#x_train, x_validation, y_train, y_validation = train_test_split(images, imageType, test_size=0.4, random_state=42)
#validation ile test verilerimizi %80-%20 ayırdık. random state 42 parametresi deneme ile bulduk.
#¨x_validation, x_test, y_validation, y_test = train_test_split(x_validation, y_validation, test_size=0.3, random_state=42)

x_train, x_tmp, y_train, y_tmp = train_test_split(
    images, imageType, test_size=0.4, random_state=42, stratify=imageType
)

x_validation, x_test, y_validation, y_test = train_test_split(
    x_tmp, y_tmp, test_size=0.3, random_state=42, stratify=y_tmp
)


print(images.shape)
print(x_train.shape)
print(x_test.shape)
print(x_validation.shape)
# ayrılmış halleriyle sayıları ve boyutları görelim
print("ALL :", Counter(imageType))
print("TRAIN:", Counter(y_train))
print("VAL  :", Counter(y_validation))
print("TEST :", Counter(y_test))
#görselleştirme
# =============================================================================
# img = preProcess(x_train[1])
# img = cv2.resize(img, (300,300))
# cv2.imshow("preProcess", img)
# =============================================================================


#ön işlemden geçirme

x_train = np.array(list(map(preProcess, x_train)))
#map metodu 2 parametre alır 1. fonksiyondur bu fonksiyonu 2. parametre olan veri listesinin hepsine uygular
#sonra bu işlenmiş verileri liste yapıyoruz bu listeyi x_traim e atıyoruz
x_test = np.array(list(map(preProcess, x_test)))
#aynı işlemi x_test için yaptık  
x_validation = np.array(list(map(preProcess, x_validation)))
#aynı işlemi validation için yaptık


# ön işlem sonrası boyutlandırma

x_train = x_train.reshape(-1, 32, 32, 1)
#bu resimleri reshape yapıyoruz -1 in anlamı ne kadar resm varsa hepsine uygula demektir
x_test = x_test.reshape(-1, 32, 32 ,1)
#aynı işlemi teste uygula
x_validation = x_validation.reshape(-1, 32, 32 ,1)
#aynı işlemi validationa uygula

print(x_train.shape)
print(x_test.shape)
print(x_validation.shape)
#boyutları görelim(ilk olarak traine uygulamadım train harici tek boyutlu çıktı)


#data generate

dataGen = ImageDataGenerator(
                                # rescale = 1/255,
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                # shear_range=0.1,
                                zoom_range=0.1,
                                # horizontal_flip=True,
                                # fill_mode="nearest",
                                # rotation_range=10
                             )
# genişleterek, uzatarak, yakınlaştırarak, çevirerek yeni veriler üretiyoruz.
dataGen.fit(x_train)
#bu data jenaratörünü x_train resimleri için çalıştırıyoruz


#kategorilere ayırma

y_train = to_categorical(y_train, numOfFails)
#train verilerini hata kategorilerine ayırıyoruz
y_test = to_categorical(y_test, numOfFails)
#test verilerini hata kategorilerine ayırıyoruz
y_validation = to_categorical(y_validation, numOfFails)
#validation verilerini hata kategorilerine ayırıyoruz

classes = np.unique(imageType)
cw = compute_class_weight("balanced", classes=classes, y=imageType)
class_weights = {int(c): float(w) for c, w in zip(classes, cw)}
print("class_weights:", class_weights)


#eğitim modeli oluşturulması

model = Sequential()
#sequential bir temel oluşturuyoruz
model.add(Conv2D(input_shape = (32,32,1), # modele giren resimlerin boyutları
                 filters = 32, #8 adet filtre yapıyoruz
                 kernel_size = (5,5), #kernelimim 5 e 5 matris
                 activation = "relu", #aktivasyonu relu ile yapıyoruz
                 padding = "same"))#bir sıra piksel eklemeyi same padding ile yapıyoruz
#modele ekleme yapıyoruz
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
#piksel ekleme yapıyoruz

model.add(Dropout(0.2))
#yukarıda yeni veri üretmiştik bu ezberlemeyi getirir(overfitting) bu nedenle syreltme yapıyoruz.

model.add(Conv2D(filters = 64, #16 adet filtre yapalım
                 kernel_size = (3,3), #kernelimiz 3 e 3 olsun
                 activation = "relu", #aktivasyonumuz relu
                 padding = "same"))# bir sıra piksel ekleyelim
#modele ikinci ekleme yaparken tekrar modele giren resimlerin boyutlarını söylemeye gerek yok
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
#piksel ekleme yapıyoruz

model.add(Dropout(0.3))
#yukarıda yeni veri üretmiştik bu ezberlemeyi getirir(overfitting) bu nedenle syreltme yapıyoruz.

model.add(Conv2D(filters = 128, #16 adet filtre yapalım
                 kernel_size = (3,3), #kernelimiz 3 e 3 olsun
                 activation = "relu", #aktivasyonumuz relu
                 padding = "same"))# bir sıra piksel ekleyelim
#modele ikinci ekleme yaparken tekrar modele giren resimlerin boyutlarını söylemeye gerek yok
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
#piksel ekleme yapıyoruz

model.add(Dropout(0.3))
#yukarıda yeni veri üretmiştik bu ezberlemeyi getirir(overfitting) bu nedenle syreltme yapıyoruz.
model.add(Flatten())
#düzleştirme yapıyoruz
model.add(Dense(units = 128, activation = "relu"))
#evrişim ağları ekliyoruz 256 hücre olsun aktivasyon olarak relu kullanalım
model.add(Dropout(0.3))
#tektar seyreltme yapalım
model.add(Dense(units = numOfFails, activation = "softmax"))
#çıktı katmanımızı yazıyoruz çıkışta hata türümüz kadar hücremiz olacak birine karar verecek 
#aktivasyon olarak softmax kullanıcaz

model.compile(loss = "categorical_crossentropy", #loss parametremiz kategoriselleştirme
              optimizer=Adam(learning_rate=1e-3), #optimizer ımız adaptif momentum
              metrics=["accuracy"]) #değerlendirmemiz accuaracy
#modelimizi derliyoruz

#modelin eğitim aşaması

hist = model.fit(
    dataGen.flow(x_train, y_train, batch_size=batchSize),
    validation_data=(x_validation, y_validation),
    callbacks=[early_stopping],
    epochs=epochSize,
    steps_per_epoch=x_train.shape[0]//batchSize,
    shuffle=1,
    class_weight=class_weights
)

"""
hist = model.fit(dataGen.flow(x_train, y_train, batch_size = batchSize),
                           validation_data = (x_validation, y_validation),
                           callbacks=[early_stopping],
                           epochs = epochSize, steps_per_epoch = x_train.shape[0]//batchSize, shuffle = 1)
"""
#modelin çıktısının görselleştirilmesi için hist e atıyoruz
#resimler 15 kez eğitilecek her adımdaki eğitim resim sayısına kalansız bölünecek sekilde eğitilecek
#data jenaratör train üzerine uygulanacak

model.save("hataAlgilamaEgitim.h5")
#modeli kaydediyoruz

# Evaluation
score = model.evaluate(x_test, y_test, verbose=1)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

y_pred = model.predict(x_test)
y_pred_class = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_true, y_pred_class)

train_pred = np.argmax(model.predict(x_train), axis=1)
train_true = np.argmax(y_train, axis=1)
print("Train confusion:\n", confusion_matrix(train_true, train_pred))


# Calculate Precision, Recall, F1 Score
precision = precision_score(y_true, y_pred_class, average=None)
recall = recall_score(y_true, y_pred_class, average=None)
f1 = f1_score(y_true, y_pred_class, average=None)

f, ax = plt.subplots(figsize=(5, 5))


sns.heatmap(cm, annot=True, linewidths=0.01, cmap="Greens", linecolor="gray", fmt=".1f", ax=ax)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Print Precision, Recall, F1 Score
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Plotting the metrics
labels = list(range(numOfFails))  # Assuming numOfFails represents the number of classes

x = range(len(labels))

plt.figure(figsize=(4, 5))

plt.plot(x, precision, label='Precision', marker='o')
plt.plot(x, recall, label='Recall', marker='o')
plt.plot(x, f1, label='F1 Score', marker='o')

plt.xlabel('Class')
plt.ylabel('Score')
plt.title('Precision, Recall and F1 Score per Class')
plt.xticks(x, labels)
plt.legend()
plt.grid(True)
plt.show()

# degerlendirme

hist.history.keys()
# histogramın içindekileri görebiliriz. kayıplar ve doğrulukları yazacağız
plt.figure(figsize=(4, 5))
plt.plot(hist.history["loss"], label = "Eğitim Loss")
#loss da kaybı görücez
plt.plot(hist.history["val_loss"], label = "Val Loss")
#doğrulama kaybını görücez
plt.legend()
plt.show()


plt.figure(figsize=(4, 5))
plt.plot(hist.history["accuracy"], label = "Eğitim accuracy")
#doğruluğu görücez
plt.plot(hist.history["val_accuracy"], label = "Val accuracy")
#doğrulama doğrularını görücez
plt.legend()
plt.show()

# Classification Report
print("Classification Report:\n", classification_report(y_true, y_pred_class))
# # degerlendirme
