# -*- coding: utf-8 -*-

import cv2 
import time

#video ismini al
video_name = "MOT17-04-DPM.mp4"

# video içeri aktar
cap = cv2.VideoCapture(video_name)

print("genislik: ", cap.get(3))
print("yukseklik: ", cap.get(4))

if cap.isOpened() == False:
    print("hata")
    
while True:
    ret, frame = cap.read()
    if ret == True:
        time.sleep(0.01) #bunu kullanmazsak çok hızlı akar
        
        cv2.imshow("Video", frame)
    else:
        break
    
    if cv2.waitKey(1) & 0xff == ord("q"):
        break

cap.release() #stop capture
cv2.destroyAllWindows()