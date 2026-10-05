# -*- coding: utf-8 -*-
import cv2
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict

VIDEO_PATH = "hsv3.mp4"
MODEL_PATH = r"runs\detect\train2\weights\best.pt"  # kendi model yolun

CONF_THRES = 0.50
IOU_THRES = 0.45

MAX_W, MAX_H = 1200, 700  # görüntü çok büyükse küçültme

def resize_for_screen(img, max_w=MAX_W, max_h=MAX_H):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def main():
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Video açılamadı:", VIDEO_PATH)
        return

    tracker = sv.ByteTrack()

    total_counts = defaultdict(int)   # program boyunca toplam
    counted_ids = set()               # her ID bir kere sayılacak

    cv2.namedWindow("YOLO Counter", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO Counter", MAX_W, MAX_H)

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        r = model.predict(frame, conf=CONF_THRES, iou=IOU_THRES, verbose=False)[0]
        det = sv.Detections.from_ultralytics(r)

        det = tracker.update_with_detections(det)

        # yeni ID'leri say
        if det.tracker_id is not None:
            for cls_id, tid in zip(det.class_id, det.tracker_id):
                if tid is None:
                    continue
                tid = int(tid)
                if tid not in counted_ids:
                    counted_ids.add(tid)
                    total_counts[int(cls_id)] += 1

        annotated = box_annotator.annotate(scene=frame.copy(), detections=det)
        annotated = label_annotator.annotate(scene=annotated, detections=det)

        c0 = total_counts[0]
        c1 = total_counts[1]
        c2 = total_counts[2]

        line1 = f"BOS: {c0}   NORMAL: {c1}"
        line2 = f"ALARM: {c2}"

        cv2.putText(annotated, line1, (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(annotated, line2, (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)


        show = resize_for_screen(annotated)
        cv2.imshow("YOLO Counter", show)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # q veya ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
