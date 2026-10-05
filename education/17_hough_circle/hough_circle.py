import cv2
import numpy as np

# LED ON fotoğrafından dedektörleri bul
img_on=cv2.imread("LED_ON_1.jpg")
img_on = cv2.resize(img_on, (640, 480))

gray_on = cv2.cvtColor(img_on,cv2.COLOR_BGR2GRAY)
img_on_blur = cv2.GaussianBlur(gray_on,(9,9),2)
edges_on = cv2.Canny(img_on_blur,20,85)

# Dedektörleri bul
circles = cv2.HoughCircles(edges_on,cv2.HOUGH_GRADIENT,1,40,param1=65,param2=35,minRadius=21,maxRadius=70)

if circles is not None:
    circles = np.uint16(np.around(circles))
    print(f"Toplam dedektör sayısı: {len(circles[0])}")
    
    # LED OFF fotoğrafını yükle karşılaştırma için
    img_off = cv2.imread("LED_OFF_1.jpg")
    img_off = cv2.resize(img_off, (640, 480))
    
    img_display = img_on.copy()
    alarmsiz_dedektorler = []
    alarma_girmis_dedektorler = []
    
    for idx, i in enumerate(circles[0,:]):
        x, y, r = i[0], i[1], i[2]
        
        # LED_ON'da dedektör çevresindeki alanı kontrol et
        roi_on = img_on[max(0, y-r):min(480, y+r), max(0, x-r):min(640, x+r)]
        # LED_OFF'ta aynı alanı kontrol et
        roi_off = img_off[max(0, y-r):min(480, y+r), max(0, x-r):min(640, x+r)]
        
        if roi_on.size > 0 and roi_off.size > 0:
            # Kırmızı değerleri kontrol et
            red_on = np.mean(roi_on[:,:,2])
            red_off = np.mean(roi_off[:,:,2])
            
            # LED_OFF'ta kırmızı, LED_ON'da daha yüksekse alarma girmişs
            # (Yani kırmızı LED yandıysa)
            red_diff = red_on - red_off
            
            print(f"Dedektör {idx}: LED_ON Red={red_on:.1f}, LED_OFF Red={red_off:.1f}, Fark={red_diff:.1f}")
            
            if red_diff > 25:
                alarma_girmis_dedektorler.append(idx)
                cv2.circle(img_display, (x, y), r, (0, 0, 255), 3)  # Kırmızı = Alarma
                print(f"  → ALARMA GİRMİŞ ✓")
            else:
                alarmsiz_dedektorler.append(idx)
                cv2.circle(img_display, (x, y), r, (0, 255, 255), 3)  # Sarı = Normal
                print(f"  → ALARMA GİRMEMİŞ")
        
    print(f"\nAlarma girmişs dedektör sayısı: {len(alarma_girmis_dedektorler)}")
    print(f"Alarma girmemiş dedektör sayısı: {len(alarmsiz_dedektorler)}")
    print(f"Alarma girmemiş dedektörler: {alarmsiz_dedektorler}")
    
    cv2.imshow("Dedektör Analizi (Sarı=Normal, Kırmızı=Alarma)", img_display)
else:
    print("Daire bulunamadı")

cv2.waitKey(0)
cv2.destroyAllWindows()