# -*- coding: utf-8 -*-

import cv2
import time
import mediapipe as mp

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

mpHand = mp.solutions.hands
hands = mpHand.Hands(static_image_mode=1, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDrw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    results = hands.process(imgRGB)
    print(results.multi_hand_landmarks)
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDrw.draw_landmarks(img, handLms, mpHand.HAND_CONNECTIONS)
            
            for id, lm in enumerate(handLms.landmark):
               # print(id,lm)
                h, w, c = img.shape
                
                cx, cy = int(lm.x*w), int(lm.y*h)
                
                #bas parmak
                if id==4:
                    cv2.circle(img, ((cx,cy)), 9, (255,0,0), cv2.FILLED)
                if id==8:
                    cv2.circle(img, ((cx,cy)), 9, (255,255,0), cv2.FILLED)
                if id==12:
                    cv2.circle(img, ((cx,cy)), 9, (0,255,0), cv2.FILLED)
                if id==16:
                    cv2.circle(img, ((cx,cy)), 9, (0,255,255), cv2.FILLED)
                if id==20:
                    cv2.circle(img, ((cx,cy)), 9, (0,0,255), cv2.FILLED)
                    

    #FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, "fps:"+str(int(fps)), (10,75), cv2.FONT_HERSHEY_PLAIN, 3, ((255,0,0)),5)
    #ekrandaki yazılar eklendi
    
    cv2.imshow("img", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()