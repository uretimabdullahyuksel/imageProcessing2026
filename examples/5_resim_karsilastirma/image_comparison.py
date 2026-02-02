import cv2
import numpy as np

path1 = "aircraft.jpg"
path2 = "aircraft2.jpg"

img1 = cv2.imread(path1)
img1 = cv2.resize(img1,(640,550))

img2 = cv2.imread(path2)
img2 = cv2.resize(img2,(640,550))

img3 = cv2.medianBlur(img1,7)


# diff = difference
diff = cv2.subtract(img1,img2)
b,g,r = cv2.split(diff)


if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0 :
    print("completely equal")
else:
    print("NOT completely equal")


cv2.imshow("Difference",diff)

cv2.waitKey(0)
cv2.destroyAllWindows()
