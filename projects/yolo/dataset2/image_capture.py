# -*- coding: utf-8 -*-
import cv2
from collections import Counter
from ultralytics import YOLO

MODEL_PATH = r"D:\git\projects\yolo\dataset2\runs\detect\train2\weights\best.pt"
IMG_PATH   = r"LED_OFF_1.jpg"   # sabit
CONF_THRES = 0.50
IOU_THRES  = 0.50

MAX_W, MAX_H = 1200, 700  # ekrana sığacak limit (istersen değiştir)

def resize_for_screen(img, max_w=MAX_W, max_h=MAX_H):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)  # 1.0'dan büyükse büyütme
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def main():
    model = YOLO(MODEL_PATH)

    img = cv2.imread(IMG_PATH)
    if img is None:
        print("Resim okunamadı:", IMG_PATH)
        return

    results = model.predict(source=img, conf=CONF_THRES, iou=IOU_THRES, verbose=False)[0]

    # sınıf id’leri
    cls_list = results.boxes.cls.cpu().numpy().astype(int).tolist()
    counts = Counter(cls_list)

    c0 = counts.get(0, 0)
    c1 = counts.get(1, 0)
    c2 = counts.get(2, 0)
    total = len(cls_list)

    print(f"Toplam: {total}  |  Class0={c0}  Class1={c1}  Class2={c2}")

    # istersen ekranda da göster
    annotated = results.plot()  # kutuları çizer
    cv2.putText(annotated, f"Total:{total}  C0:{c0}  C1:{c1}  C2:{c2}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    show = resize_for_screen(annotated)
    cv2.imshow("YOLO Count", show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

    # kaydetmek istersen:
    cv2.imwrite("annotated.jpg", annotated)

if __name__ == "__main__":
    main()
