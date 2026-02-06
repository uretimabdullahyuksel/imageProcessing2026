from ultralytics import YOLO
import cv2

MODEL_PATH = r"runs\detect\train\weights\best.pt"
CONF = 0.40
IOU = 0.3   # (opsiyonel) NMS daha agresif olsun diye

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Camera not opened")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    r = model.predict(frame, conf=CONF, iou=IOU, verbose=False)[0]

    # sadece CONF üstünü say
    count = 0
    if r.boxes is not None and len(r.boxes) > 0:
        confs = r.boxes.conf.cpu().numpy()
        count = int((confs >= CONF).sum())

    annotated = r.plot()  # çizimler conf ayarına göre zaten filtreli gelir

    if count != 36:
        cv2.putText(annotated, f"SAYI: {count} HATALI ", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 5)
    else:
        cv2.putText(annotated, f"SAYI: {count} (OK)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Varistor Counter", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
