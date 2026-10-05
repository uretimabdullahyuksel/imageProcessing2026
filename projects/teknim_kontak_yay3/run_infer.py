from ultralytics import YOLO
import sys

MODEL = "runs/pso_final/pso_it1_s2_1780619305/weights/best.pt"
SOURCE = "images2"
OUT_PROJECT = "runs/infer"
OUT_NAME = "pso_final_infer"

print(f"Loading model {MODEL}")
model = YOLO(MODEL)
print("Model loaded, running predict...")
model.predict(source=SOURCE, imgsz=640, save=True, project=OUT_PROJECT, name=OUT_NAME, verbose=False)
print(f"Predict finished. outputs in {OUT_PROJECT}/{OUT_NAME}")
