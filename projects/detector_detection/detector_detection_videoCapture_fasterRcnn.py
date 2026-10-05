# -*- coding: utf-8 -*-

import cv2
import torch
from torchvision import models, transforms
import numpy as np
from keras.models import load_model
from PIL import Image

# Preprocess fonksiyonu
def preProcess(img):
    # Siyah beyaza çeviriyoruz
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Histogramı 0-255 yaptık
    img = cv2.equalizeHist(img)
    # Görüntünün piksel değerlerini 0-1 arasına getirdik
    img = img / 255.0
    # Son hali geri döndürüyoruz
    return img

# Faster R-CNN kullanarak nesne tespiti ve kırpma
def detect_and_crop_object(frame):
    transform = transforms.Compose([transforms.ToTensor()])
    image = Image.fromarray(frame).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = faster_rcnn_model(image_tensor)

    # En yüksek puanlı bounding box'ı al
    boxes = predictions[0]['boxes'].cpu()
    scores = predictions[0]['scores'].cpu()

    if len(scores) > 0 and scores[0] > 0.7:
        best_box = boxes[0].numpy().astype(int)
        left, top, right, bottom = best_box
        cropped_frame = frame[top:bottom, left:right]
        return cropped_frame

    return None

def resize_frame(frame, target_size=(100, 100)):
    return cv2.resize(frame, target_size)

print("Kamera açılıyor...")
cap = cv2.VideoCapture(0)
cap.set(3, 480)
cap.set(4, 480)

# GPU kullanımı için kontrol
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Faster R-CNN modeli yükleniyor...")
# Modeli GPU'ya taşı
faster_rcnn_model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)
faster_rcnn_model.eval()

print("Keras modeli yükleniyor...")
model = load_model("optimized_hataAlgilamaEgitim_3-10.h5")

frame_counter = 0
process_every_n_frames = 50  # Her 5 karede bir işlem yap
print("İşlem başlıyor...")
while True:
    success, frame = cap.read()
    
    if success:
        # cropped_frame = detect_and_crop_object(frame)
        small_frame = resize_frame(frame)
        
        # if success and frame_counter % process_every_n_frames == 0:
        cropped_frame = detect_and_crop_object(small_frame)
            
        frame_counter += 1
            
        if cropped_frame is not None:
            img = cv2.resize(cropped_frame, (32, 32))
            cv2.imshow("crop", cropped_frame)
            img = preProcess(img)
            img = img.reshape(1, 32, 32, 1)
            
            predictions = model.predict(img)
            failIndex = np.argmax(predictions)
            probVal = np.amax(predictions)

            print(failIndex, probVal)

            if probVal > 0.7:
                if failIndex == 0:
                    cv2.putText(frame, "Dedektor Yok    " + str(probVal), (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
                elif failIndex == 1:
                    cv2.putText(frame, "Led Cubuk Hatasi    " + str(probVal), (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
                elif failIndex == 2:
                    cv2.putText(frame, "Leke Hatasi    " + str(probVal), (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
                elif failIndex == 3:
                    cv2.putText(frame, "Sorun Yok    " + str(probVal), (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
                else:
                    cv2.putText(frame, "Nesne Yok    " + str(probVal), (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)

        cv2.imshow("Siniflandirma", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()