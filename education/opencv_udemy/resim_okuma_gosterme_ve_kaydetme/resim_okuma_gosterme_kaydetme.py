import cv2

img = cv2.imread("messi5.jpg", cv2.IMREAD_GRAYSCALE)

cv2.namedWindow("Grayscale Image", cv2.WINDOW_NORMAL)   
cv2.imshow("Grayscale Image", img)
cv2.imwrite("messi5_grayscale.jpg", img)    
cv2.waitKey(0)

