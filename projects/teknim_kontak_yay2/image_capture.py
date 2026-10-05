# -*- coding: utf-8 -*-

import os
import cv2
from ultralytics import YOLO


#MODEL_PATH = r"D:\git\projects\teknim_kontak_yay2\runs\yayli_yaysiz_cpu_test2\weights\best.pt"
#MODEL_PATH = r"D:\git\projects\teknim_kontak_yay2\runs\yayli_yaysiz_cpu_test\weights\best.pt"
MODEL_PATH = r"D:\git\projects\teknim_kontak_yay2\runs\yayli_yaysiz_cpu_test-2\weights\best.pt"
IMG_PATH   = r"D:\git\projects\teknim_kontak_yay2\images\21.png"

OUTPUT_DIR = r"D:\git\projects\teknim_kontak_yay2\outputs_yaysiz"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_YAYLI = 0
CLASS_YAYSIZ = 1

CONF_THRES = 0.35
IOU_THRES = 0.45
PRED_IMGSZ = 1280

MAX_W, MAX_H = 1200, 800


def resize_for_screen(img, max_w=MAX_W, max_h=MAX_H):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def main():
    print("Model yükleniyor...")
    model = YOLO(MODEL_PATH)
    print("Model sınıfları:", model.names)

    img = cv2.imread(IMG_PATH)

    if img is None:
        print("Resim okunamadı:", IMG_PATH)
        return

    print("Resim boyutu:", img.shape)

    result = model.predict(
        source=img,
        conf=CONF_THRES,
        iou=IOU_THRES,
        imgsz=PRED_IMGSZ,
        verbose=False
    )[0]

    annotated = img.copy()
    yaysiz_count = 0

    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())

            if cls_id != CLASS_YAYSIZ:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            yaysiz_count += 1

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(
                annotated,
                f"YAYSIZ {conf:.2f}",
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )

            print(f"{yaysiz_count}. yaysız | conf={conf:.3f} | box=({x1}, {y1}, {x2}, {y2})")

    print("\nGENEL SONUÇ")
    print("Toplam yaysız kontak:", yaysiz_count)

    output_path = os.path.join(OUTPUT_DIR, "annotated_yaysiz.jpg")
    cv2.imwrite(output_path, annotated)

    show = resize_for_screen(annotated)
    cv2.imshow("Yaysiz Kontak Tespiti", show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Kaydedilen çıktı:", output_path)


if __name__ == "__main__":
    main()