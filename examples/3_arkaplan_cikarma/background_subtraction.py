import cv2

cap = cv2.VideoCapture("car.mp4")

if not cap.isOpened():
    raise SystemExit("car.mp4 acilamadi! Dosya yolu/ismi yanlis veya codec sorunu var.")

subtractor = cv2.createBackgroundSubtractorMOG2(
    history=100, varThreshold=150, detectShadows=True
)

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Frame okunamadi (video bitti ya da dosya okunamiyor).")
        break

    frame = cv2.resize(frame, (640, 480))
    mask = subtractor.apply(frame)

    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
