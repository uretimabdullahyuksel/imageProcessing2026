from ultralytics import YOLO

model = YOLO("yolo26s.pt")  # olmazsa "yolov8s.pt" kullan

model.train(
    data=r"D:\git\projects\teknim_kontak_yay\dataset\data.yaml",
    epochs=200,
    imgsz=1280,
    batch=4,
    patience=40,
    workers=0,
    project=r"D:\git\projects\teknim_kontak_yay\runs",
    name="yayli_yaysiz_train"
)