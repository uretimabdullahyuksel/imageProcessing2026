# -*- coding: utf-8 -*-
"""
Created on Thu May 16 17:08:51 2024

@author: abdullah.yuksel
"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import Adam
warnings.filterwarnings("ignore")

#hata datamızın yolunu belirtiyoruz
path = "specialData"

# # Constants
# #kaçar adet resimlerle eğitim yapılacağını belirtiyoruz.
# batch_size = 16
# #kaç defa eğitim yapılacağını belirtiyoruz.
# epoch_size = 50
# #pso da kaç adet parçacık ile araştırma yapılacağını belirtiyoruz.
# pso_parcacik = 5
# #pso iterasyonunun kaç defa olacağını belirtiuoruz
# pso_iter = 1

# Constants
# batch_size = 28
# epoch_size = 50
# pso_parcacik = 10
# pso_iter = 2

# lb = [8, 3, 16, 3, 32, 5, 256, 0.2]
# ub = [8, 3, 16, 3, 32, 5, 256, 0.2]

# lb = [6, 4, 14, 2, 30, 2, 200, 0.1, 0.0001]
# ub = [10, 6, 18, 4, 34, 4, 300, 0.3, 0.001]

# lb = [6, 4, 14, 2, 30, 2, 200, 0.1]
# ub = [10, 6, 18, 4, 34, 4, 300, 0.3]

# PSO Hyperparameter Optimization
# lb = [4, 2, 8, 2, 16, 2, 128, 0.1]
# ub = [64, 5, 128, 5, 256, 5, 512, 0.5]


#en iyi
# batch_size = 3
# epoch_size = 50
# pso_parcacik = 5
# pso_iter = 1

# lb = [6, 4, 14, 2, 30, 2, 200, 0.1]
# ub = [10, 6, 18, 4, 34, 4, 300, 0.3]

pso_parcacik = 10
pso_iter = 3

lb = [24, 4, 52, 2, 112, 2, 480, 0.1, 3, 0.001, 10]
ub = [48, 6, 76, 4, 148, 4, 564, 0.3, 60, 0.0001, 100]

#preprocess function
def preProcess(img):
    #siyah beyaza ceviriyoruz
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #histogramı 0-255 yaptık
    img = cv2.equalizeHist(img)
    #görüntünün piksel değerlerini 0-1 arasına getirdik
    img = img / 255
    #son hali geri döndürdük
    return img

# Load Data
def load_data(path):
    #hata datamızı listeye atıyoruz
    myList = os.listdir(path)
    #hata listemizin uzunluğunu alıyoruz
    numOfFails = len(myList)
    #resim ve türler için dizi oluşturuyoruz
    images, imageType = [], []
    for i in range(numOfFails): # klasör içinde dolaşıyoruz
        myImageList = os.listdir(os.path.join(path, str(i))) # türlerin isimlerini aldık
        for j in myImageList: # türler içinde dolaşıyoruz
            img = cv2.imread(os.path.join(path, str(i), j)) # görüntülerin isimlerini aldık
            img = cv2.resize(img, (32, 32)) # resimleri 32,32 yaptık
            images.append(img) # images listesine ekledik
            imageType.append(i) # tür listesine ekledik
    return np.array(images), np.array(imageType), numOfFails # geri kalan işlemlerde array a çevirmemiz gerekiyor

# Objective Function for PSO
def cnn_model(hyperparams):
    #♣hiperparametre tanımlarını yapıyoruz
    # filters1, kernel1, filters2, kernel2, filters3, kernel3, dense_units, dropout_rate, learning_rate = hyperparams
    filters1, kernel1, filters2, kernel2, filters3, kernel3, dense_units, dropout_rate, batch_size, learning_rate, epoch_size  = hyperparams
    # filters1, kernel1, filters2, kernel2, filters3, kernel3, dense_units, dropout_rate, batch_size, epoch_size  = hyperparams
    kernel1 = int(kernel1)
    kernel2 = int(kernel2)
    kernel3 = int(kernel3)
    filters1 = int(filters1)
    filters2 = int(filters2)
    filters3 = int(filters3)
    dense_units = int(dense_units)
    batch_size = int(batch_size)
    dropout_rate = max(0, min(dropout_rate, 1))  # Dropout oranının 0 ile 1 arasında olmasını sağla
    epoch_size = int(epoch_size)  # epoch sayısını tamsayıya dönüştür
    #eğitim modeli oluşturulması
    #sequential bir temel oluşturuyoruz
    model = Sequential()
    #sequential bir temel oluşturuyoruz
    #modele ekleme yapıyoruz
    model.add(Conv2D(filters=int(filters1),  # filtre tanımnlıyoruz
                     kernel_size=(kernel1, kernel1), #kernel tanımlaması yapıyoruz
                     padding='same', #bir sıra piksel eklemeyi same padding ile yapıyoruz
                     input_shape=(32, 32, 1))) # modele giren resimlerin boyutları
    
    #batch normalizasyon yapıyoruz.
    model.add(BatchNormalization())
    model.add(Activation('relu')) #aktivasyonu relu ile yapıyoruz
    #piksel ekleme yapıyoruz
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #yukarıda yeni veri üretmiştik bu ezberlemeyi getirir(overfitting) bu nedenle syreltme yapıyoruz.
    model.add(Dropout(dropout_rate))
    
    model.add(Conv2D(filters=int(filters2),   # filtre tanımnlıyoruz
                     kernel_size=(kernel2, kernel2),  #kernel tanımlaması yapıyoruz
                     padding='same')) #bir sıra piksel eklemeyi same padding ile yapıyoruz
    #batch normalizasyon yapıyoruz.
    model.add(BatchNormalization())
    model.add(Activation('relu')) #aktivasyonu relu ile yapıyoruz
    #piksel ekleme yapıyoruz
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #yukarıda yeni veri üretmiştik bu ezberlemeyi getirir(overfitting) bu nedenle syreltme yapıyoruz.
    model.add(Dropout(dropout_rate))
    
    model.add(Conv2D(filters=int(filters3),   # filtre tanımnlıyoruz
                     kernel_size=(kernel2, kernel3),  #kernel tanımlaması yapıyoruz
                     padding='same')) #bir sıra piksel eklemeyi same padding ile yapıyoruz
    #batch normalizasyon yapıyoruz.
    model.add(BatchNormalization())
    model.add(Activation('relu')) #aktivasyonu relu ile yapıyoruz
    #piksel ekleme yapıyoruz
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #yukarıda yeni veri üretmiştik bu ezberlemeyi getirir(overfitting) bu nedenle syreltme yapıyoruz.
    model.add(Dropout(dropout_rate))
    
    #düzleştirme yapıyoruz
    model.add(Flatten())
    #evrişim ağları ekliyoruz 256 hücre olsun aktivasyon olarak relu kullanalım
    model.add(Dense(dense_units, activation='relu'))
    #tektar seyreltme yapalım
    model.add(Dropout(dropout_rate))
    #çıktı katmanımızı yazıyoruz çıkışta hata türümüz kadar hücremiz olacak birine karar verecek 
    #aktivasyon olarak softmax kullanıcaz
    model.add(Dense(units = numOfFails, activation='softmax'))
    
    #modelimizi derliyoruz
    # model.compile(loss='categorical_crossentropy', #loss parametremiz kategoriselleştirme
    #               # optimizer=Adam(learning_rate=learning_rate), #optimizer ımız adaptif momentum
    #               optimizer='Adam', #optimizer ımız adaptif momentum
    #               metrics=['accuracy']) #değerlendirmemiz accuaracy
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', #loss parametremiz kategoriselleştirme
                  optimizer=optimizer, #optimizer ımız adaptif momentum
                  metrics=['accuracy']) #değerlendirmemiz accuaracy
    
    #modelin eğitim aşaması
    #modelin çıktısının görselleştirilmesi için hist e atıyoruz
    #resimler 15 kez eğitilecek her adımdaki eğitim resim sayısına kalansız bölünecek sekilde eğitilecek
    #data jenaratör train üzerine uygulanacak
    hist = model.fit(dataGen.flow(x_train, y_train, batch_size=batch_size),
                     validation_data=(x_validation, y_validation),
                     epochs=epoch_size, steps_per_epoch=x_train.shape[0] // batch_size, shuffle=1, verbose=0)
    
    val_acc = hist.history['val_accuracy'][-1]
    return 1 - val_acc  # Minimize 1 - val_accuracy

# PSO Hyperparameter Optimization
def pso_optimize(cnn_model, lb, ub, swarmsize, maxiter):
    # Initialize particles
    num_params = len(lb) #Optimize edilecek hiperparametrelerin sayısı.
    particles = np.random.uniform(lb, ub, (swarmsize, num_params)) #Parçacıkların başlangıç pozisyonları
    velocities = np.random.uniform(-1, 1, (swarmsize, num_params)) #Parçacıkların başlangıç hızları
    
    pbest_positions = particles.copy() #Her parçacığın en iyi pozisyonu
    pbest_scores = np.full(swarmsize, np.inf) #Her parçacığın en iyi skoru
    gbest_position = None #Sürüdeki en iyi pozisyon
    gbest_score = np.inf #Sürüdeki en iyi skor


    # PSO main loop
    for iteration in range(maxiter):  
        for i in range(swarmsize):
            print("iterasyon: ",iteration+1)
            print("cnn e giden parcacik: ",i+1)
            score = cnn_model(particles[i])
            print("cnn sonucu skor: ",1-score)
            if score < pbest_scores[i]:
                pbest_scores[i] = score
                pbest_positions[i] = particles[i]
            if score < gbest_score:
                gbest_score = score
                gbest_position = particles[i]

        # Update velocities and positions
        w = 0.5
        c1 = 2
        c2 = 2
        # for i in range(swarmsize):
        r1, r2 = np.random.rand(num_params), np.random.rand(num_params)  # Each particle gets its own random values
        velocities[i] = (w * velocities[i] + c1 * r1 * (pbest_positions[i] - particles[i]) + c2 * r2 * (gbest_position - particles[i]))
        particles[i] += velocities[i]
    
        # Clamp positions to bounds
        particles = np.clip(particles, lb, ub)

        print(f"Iteration {iteration+1}/{maxiter}, Best Score: {1-gbest_score}")

    return gbest_position, gbest_score

# görüntü sayısı, görüntü türü, hata tür sayısı alınıyor.
images, imageType, numOfFails = load_data(path)

#görselleştirme        
# =============================================================================
print("Görüntü sayısı: ",len(images))
print("Görüntü tür sayısı: ",len(imageType))
print("Hata tür sayısı: ",numOfFails)
#kac hata turumuz var alıyoruz
 #sayıları gördük
# =============================================================================

#train ile validation verilerimizi yarı yarıya ayırdık. random state 42 parametresi deneme ile bulduk.
x_train, x_validation, y_train, y_validation = train_test_split(images, imageType, test_size=0.4, random_state=42)
#validation ile test verilerimizi %80-%20 ayırdık. random state 42 parametresi deneme ile bulduk.
x_validation, x_test, y_validation, y_test = train_test_split(x_validation, y_validation, test_size=0.3, random_state=42)

print("Görüntü boyutu: ",images.shape)
print("eğitim boyutu: ",x_train.shape)
print("doğrulama boyutu: ",x_validation.shape)
print("test boyutu: ",x_test.shape)


#ön işlemden geçirme
#map metodu 2 parametre alır 1. fonksiyondur bu fonksiyonu 2. parametre olan veri listesinin hepsine uygular
#sonra bu işlenmiş verileri liste yapıyoruz bu listeyi x_traim e atıyoruz
x_train = np.array(list(map(preProcess, x_train)))#.reshape(-1, 32, 32, 1)
#aynı işlemi x_test için yaptık  
x_test = np.array(list(map(preProcess, x_test)))#.reshape(-1, 32, 32, 1)
#aynı işlemi validation için yaptık
x_validation = np.array(list(map(preProcess, x_validation)))#.reshape(-1, 32, 32, 1)


# ön işlem sonrası boyutlandırma
print("ön işlem sonrası boyutlandırma")
x_train = x_train.reshape(-1, 32, 32, 1)
#bu resimleri reshape yapıyoruz -1 in anlamı ne kadar resm varsa hepsine uygula demektir
x_test = x_test.reshape(-1, 32, 32 ,1)
#aynı işlemi teste uygula
x_validation = x_validation.reshape(-1, 32, 32 ,1)
#aynı işlemi validationa uygula

print("eğitim boyutu: ",x_train.shape)
print("doğrulama boyutu: ",x_validation.shape)
print("test boyutu: ",x_test.shape)

#boyutları görelim(ilk olarak traine uygulamadım train harici tek boyutlu çıktı)

#data generate
# genişleterek, uzatarak, yakınlaştırarak, çevirerek yeni veriler üretiyoruz.
dataGen = ImageDataGenerator(
    width_shift_range=0.1,  #  yatay kaydırma
    height_shift_range=0.1, #  dikey kaydırma
    shear_range=0.1,        #  eğilme
    zoom_range=0.1,         #  zoom
    # horizontal_flip=True,   #  yatay çevirme
    fill_mode="nearest",    #  eksik pikselleri en yakın komşu değeriyle doldurma
    rotation_range=10       #  +-10 derece arasında döndürme
)
#bu data jenaratörünü x_train resimleri için çalıştırıyoruz
dataGen.fit(x_train)

#kategorilere ayırma
#train verilerini hata kategorilerine ayırıyoruz
y_train = to_categorical(y_train, numOfFails)
#test verilerini hata kategorilerine ayırıyoruz
y_test = to_categorical(y_test, numOfFails)
#validation verilerini hata kategorilerine ayırıyoruz
y_validation = to_categorical(y_validation, numOfFails)


#pso optimizasyon başlıyor
print("pso optimizasyon başlıyor")
#best_hyperparams, best_score = pso(cnn_model, lb, ub, swarmsize=pso_parcacik, maxiter=pso_iter)
best_hyperparams, best_score = pso_optimize(cnn_model, lb, ub, swarmsize=pso_parcacik, maxiter=pso_iter)

print("Best Hyperparameters:", best_hyperparams)
print("Best Validation Accuracy:", 1-best_score)

# Save the best model with the optimized hyperparameters
# filters1, kernel1, filters2, kernel2, filters3, kernel3, dense_units, dropout_rate, learning_rate = best_hyperparams
filters1, kernel1, filters2, kernel2, filters3, kernel3, dense_units, dropout_rate, batch_size, learning_rate, epoch_size  = best_hyperparams
# filters1, kernel1, filters2, kernel2, filters3, kernel3, dense_units, dropout_rate, batch_size, epoch_size  = best_hyperparams
kernel1 = int(kernel1)
kernel2 = int(kernel2)
kernel3 = int(kernel3)
filters1 = int(filters1)
filters2 = int(filters2)
filters3 = int(filters3)
dense_units = int(dense_units)
batch_size = int(batch_size)
dropout_rate = max(0, min(dropout_rate, 1))  # Dropout oranının 0 ile 1 arasında olmasını sağla
epoch_size = int(epoch_size)  # epoch sayısını tamsayıya dönüştür

print("kernel1: ", kernel1)
print("kernel2: ", kernel2)
print("kernel3: ", kernel3)

print("filter1: ", filters1)
print("filter2: ", filters2)
print("filter3: ", filters3)

print("dense: ", dense_units)
print("batch_size: ", batch_size)
print("dropout_rate: ", dropout_rate)
print("epoch_size: ", epoch_size)
print("learning_rate: ", learning_rate)

model = Sequential()
model.add(Conv2D(filters=int(filters1), kernel_size=(kernel1, kernel1), padding='same', input_shape=(32, 32, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(dropout_rate))

model.add(Conv2D(filters=int(filters2), kernel_size=(kernel2, kernel2), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(dropout_rate))

model.add(Conv2D(filters=int(filters3), kernel_size=(kernel3, kernel3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(dropout_rate))

model.add(Flatten())
model.add(Dense(dense_units, activation='relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(numOfFails, activation='softmax'))

# model.compile(loss='categorical_crossentropy', 
#               # optimizer=Adam(learning_rate=learning_rate), 
#               optimizer='Adam', 
#               metrics=['accuracy'])
optimizer = Adam(learning_rate=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

hist = model.fit(dataGen.flow(x_train, y_train, batch_size=batch_size),
          validation_data=(x_validation, y_validation),
          epochs=epoch_size, steps_per_epoch=x_train.shape[0] // batch_size, shuffle=1)

model.save("optimized_kontakAlgilamaEgitim.h5")

# Evaluation
score = model.evaluate(x_test, y_test, verbose=1)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

y_pred = model.predict(x_validation)
y_pred_class = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_validation, axis=1)
cm = confusion_matrix(y_true, y_pred_class)

# Calculate Precision, Recall, F1 Score
precision = precision_score(y_true, y_pred_class, average=None)
recall = recall_score(y_true, y_pred_class, average=None)
f1 = f1_score(y_true, y_pred_class, average=None)

# Calculate FAR and FNR
FP = cm.sum(axis=0) - np.diag(cm)
FN = cm.sum(axis=1) - np.diag(cm)
FAR = FP / (FP + cm.sum(axis=0))
FNR = FN / (FN + np.diag(cm))

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
print("FAR (False Acceptance Rate) per Class:", FAR)
print("FNR (False Rejection Rate) per Class:", FNR)

# Plotting the metrics
# labels = list(range(numOfFails))  # Assuming numOfFails represents the number of classes
labels = list(range(len(np.unique(y_true))))  # Update this line to use the actual number of classes

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

# Plotting FAR and FNR
plt.figure(figsize=(4, 5))

plt.plot(x, FAR, label='FAR', marker='o')
plt.plot(x, FNR, label='FNR', marker='o')

plt.xlabel('Class')
plt.ylabel('Rate')
plt.title('FAR and FNR per Class')
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