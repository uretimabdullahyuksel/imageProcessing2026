from inference import get_model
import cv2
from collections import Counter

IMAGE_PATH = "LED_ON_1.jpg"
CONF_TH = 0.85

CLASS_COLORS = {
    0: (0, 255, 0),     # yeşil
    1: (0, 255, 255),   # sarı
    2: (0, 0, 255),     # kırmızı
}

image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Resim okunamadı: {IMAGE_PATH}")

model = get_model(model_id="varistor/detector-lz5nb-instant-1")  # <-- sende görünen model id
results = model.infer(image)[0]

counts = Counter()

for p in results.get("predictions", []):
    conf = float(p.get("confidence", 0.0))
    cls  = int(p.get("class_id", -1))
    if conf < CONF_TH or cls not in (0, 1, 2):
        continue

    counts[cls] += 1

    # Roboflow hosted inference genelde center-based bbox döndürür:
    x = float(p["x"])
    y = float(p["y"])
    w = float(p["width"])
    h = float(p["height"])

    x1 = int(x - w / 2)
    y1 = int(y - h / 2)
    x2 = int(x + w / 2)
    y2 = int(y + h / 2)

    color = CLASS_COLORS.get(cls, (255, 255, 255))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, f"{cls} {conf:.2f}", (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

print(f"Class0={counts[0]}  Class1={counts[1]}  Class2={counts[2]}  (conf>={CONF_TH})")

# ekrana yazdır
cv2.putText(image, f"C0:{counts[0]}  C1:{counts[1]}  C2:{counts[2]}",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

cv2.imshow("Detections", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
