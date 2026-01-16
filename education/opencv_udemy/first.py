import cv2

# Load an color image in grayscale
img = cv2.imread('messi5.jpg',0)

# oku
cv2.imshow('image',img)

# cv2.waitKey () bir klavye bağlama işlevidir.
# İşlev, herhangi bir klavye olayı için belirtilen milisaniye kadar bekler.
k = cv2.waitKey(0) & 0xFF 

if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows() # yarattığımız tüm pencereleri yok eder.
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('messi_gray.png',img)
    cv2.destroyAllWindows()

