from ultralytics import YOLO
import cv2

MODEL_PATH = r"runs\detect\train\weights\best.pt"
CONF = 0.40
IOU = 0.3

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Camera not opened")

WIN = "Varistor Counter"

# ✅ Tam ekran pencereyi önce oluştur
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)  # normal oluştur
cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # tam ekran yap

while True:
    ret, frame = cap.read()
    if not ret:
        break

    r = model.predict(frame, conf=CONF, iou=IOU, verbose=False)[0]

    count = 0
    if r.boxes is not None and len(r.boxes) > 0:
        confs = r.boxes.conf.cpu().numpy()
        count = int((confs >= CONF).sum())

    annotated = r.plot()

    if count != 36:
        cv2.putText(annotated, f"SAYI: {count} HATALI", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 5)
    else:
        cv2.putText(annotated, f"SAYI: {count} (OK)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow(WIN, annotated)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # q veya ESC çıkış
        break

cap.release()
cv2.destroyAllWindows()
