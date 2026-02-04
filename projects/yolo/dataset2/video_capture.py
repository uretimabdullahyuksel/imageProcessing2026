# -*- coding: utf-8 -*-
import cv2
from ultralytics import YOLO
from collections import Counter

VIDEO_PATH = "hsv3.mp4"
MODEL_PATH = r"runs\detect\train2\weights\best.pt"  # <- gerekirse düzelt
CONF_THRES = 0.50
IOU_THRES = 0.45

MAX_W, MAX_H = 1200, 700  # ekranda gösterirken küçültme limiti

def resize_for_screen(img, max_w=MAX_W, max_h=MAX_H):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)  # büyütme yok
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def main():
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Video açılamadı:", VIDEO_PATH)
        return

    cv2.namedWindow("YOLO Video Count", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO Video Count", MAX_W, MAX_H)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO tahmin
        r = model.predict(frame, conf=CONF_THRES, iou=IOU_THRES, verbose=False)[0]

        # sınıf say
        counts = Counter()
        if r.boxes is not None and len(r.boxes) > 0:
            cls_list = r.boxes.cls.int().tolist()
            counts.update(cls_list)

        c0 = counts.get(0, 0)
        c1 = counts.get(1, 0)
        c2 = counts.get(2, 0)
        total = c0 + c1 + c2

        # kutulu görüntü (ultralytics çizdirsin)
        annotated = r.plot()  # boxes + labels + conf

        # üst bilgi yaz
        text = f"Total:{total}  Class0:{c0}  Class1:{c1}  Class2:{c2}  conf>{CONF_THRES}"
        cv2.putText(annotated, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        show = resize_for_screen(annotated)
        cv2.imshow("YOLO Video Count", show)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # q veya ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
