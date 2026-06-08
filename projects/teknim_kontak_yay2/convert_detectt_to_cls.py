# -*- coding: utf-8 -*-

import os
import cv2
from pathlib import Path

SOURCE_DATASET = Path(r"D:\git\projects\teknim_kontak_yay2")
OUTPUT_DATASET = Path(r"D:\git\projects\teknim_kontak_yay2_cls")

CLASS_NAMES = {
    0: "yayli",
    1: "yaysiz"
}

SPLITS = ["train", "valid", "test"]

# Crop etrafına biraz pay bırakmak için
PADDING_RATIO = 0.20


def yolo_to_xyxy(label, img_w, img_h):
    cls_id, x_center, y_center, w, h = label

    x_center *= img_w
    y_center *= img_h
    w *= img_w
    h *= img_h

    x1 = int(x_center - w / 2)
    y1 = int(y_center - h / 2)
    x2 = int(x_center + w / 2)
    y2 = int(y_center + h / 2)

    return int(cls_id), x1, y1, x2, y2


def add_padding(x1, y1, x2, y2, img_w, img_h, padding_ratio):
    bw = x2 - x1
    bh = y2 - y1

    pad_x = int(bw * padding_ratio)
    pad_y = int(bh * padding_ratio)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(img_w, x2 + pad_x)
    y2 = min(img_h, y2 + pad_y)

    return x1, y1, x2, y2


def convert_split(split):
    images_dir = SOURCE_DATASET / split / "images"
    labels_dir = SOURCE_DATASET / split / "labels"

    if not images_dir.exists():
        print(f"[UYARI] Klasör yok: {images_dir}")
        return

    if not labels_dir.exists():
        print(f"[UYARI] Klasör yok: {labels_dir}")
        return

    for cls_name in CLASS_NAMES.values():
        (OUTPUT_DATASET / split / cls_name).mkdir(parents=True, exist_ok=True)

    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_files.extend(images_dir.glob(ext))

    crop_count = 0

    for img_path in image_files:
        label_path = labels_dir / f"{img_path.stem}.txt"

        if not label_path.exists():
            continue

        img = cv2.imread(str(img_path))

        if img is None:
            print(f"[UYARI] Resim okunamadı: {img_path}")
            continue

        img_h, img_w = img.shape[:2]

        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        obj_index = 0

        for line in lines:
            parts = line.strip().split()

            if len(parts) < 5:
                continue

            try:
                values = list(map(float, parts[:5]))
            except ValueError:
                continue

            cls_id, x1, y1, x2, y2 = yolo_to_xyxy(values, img_w, img_h)

            if cls_id not in CLASS_NAMES:
                continue

            x1, y1, x2, y2 = add_padding(
                x1, y1, x2, y2,
                img_w, img_h,
                PADDING_RATIO
            )

            crop = img[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            cls_name = CLASS_NAMES[cls_id]

            out_name = f"{img_path.stem}_{obj_index:03d}.jpg"
            out_path = OUTPUT_DATASET / split / cls_name / out_name

            cv2.imwrite(str(out_path), crop)

            obj_index += 1
            crop_count += 1

    print(f"[OK] {split}: {crop_count} crop oluşturuldu.")


def main():
    print("Detection dataset classification dataset'e çevriliyor...")
    print("Kaynak:", SOURCE_DATASET)
    print("Çıkış:", OUTPUT_DATASET)

    for split in SPLITS:
        convert_split(split)

    print("\nTamamlandı.")
    print("Classification dataset hazır:")
    print(OUTPUT_DATASET)


if __name__ == "__main__":
    main()