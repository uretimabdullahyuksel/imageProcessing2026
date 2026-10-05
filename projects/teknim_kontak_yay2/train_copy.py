# -*- coding: utf-8 -*-

from ultralytics import YOLO

DATA_DIR = r"D:\git\projects\teknim_kontak_yay2_cls"

model = YOLO("yolo26s-cls.pt")  # olmazsa yolov8s-cls.pt kullan

model.train(
    data=DATA_DIR,
    epochs=50,
    imgsz=224,
    batch=16,
    patience=10,
    workers=0,
    device="cpu",
    project=r"D:\git\projects\teknim_kontak_yay2\runs_cls",
    name="yayli_yaysiz_cls_cpu_test"
)

print("Classification eğitimi tamamlandı.")