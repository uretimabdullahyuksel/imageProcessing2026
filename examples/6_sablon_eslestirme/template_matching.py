import cv2
import numpy as np
import sys

image_path = "varistor.jpg"
template_path = "varistor1.jpg"

img = cv2.imread(image_path)
if img is None:
    sys.exit(f"Ana resim okunamadi: {image_path}")

template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
if template is None:
    sys.exit(f"Template okunamadi: {template_path}")

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

h, w = template.shape  # dikkat: (h,w)

result = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)

threshold = 0.60
locations = np.where(result >= threshold)

for pt in zip(*locations[::-1]):  # (x,y)
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 3)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()