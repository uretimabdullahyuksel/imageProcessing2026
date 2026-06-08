# -*- coding: utf-8 -*-

from ultralytics import YOLO

DATA_YAML = r"D:\git\projects\teknim_kontak_yay2\data.yaml"

model = YOLO("yolo26s.pt")

model.train(
    data=DATA_YAML,
    epochs=50,
    imgsz=640,
    batch=2,
    patience=15,
    workers=0,
    device="cpu",
    project=r"D:\git\projects\teknim_kontak_yay2\runs",
    name="yayli_yaysiz_cpu_test"
)

print("Eğitim tamamlandı.")