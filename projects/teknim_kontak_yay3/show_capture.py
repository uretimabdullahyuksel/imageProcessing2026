#!/usr/bin/env python3
"""
show_capture.py

Simple single-image viewer that runs the trained YOLO model on one image
and displays detections with boxes and labels (no file saving).

Usage:
  python show_capture.py --source images/7.jpg --conf 0.6
"""
import argparse
from ultralytics import YOLO
import cv2


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='images/7.jpg', help='Image file to run inference on')
    p.add_argument('--model', default='runs/pso_final/pso_it1_s2_1780619305/weights/best.pt', help='Path to model weights')
    p.add_argument('--imgsz', type=int, default=640, help='Inference image size')
    p.add_argument('--conf', type=float, default=0.6, help='Confidence threshold to display')
    p.add_argument('--scale', type=float, default=0.6, help='Scale factor for display (0-1). 1=no scaling')
    args = p.parse_args()

    print('Loading model', args.model)
    model = YOLO(args.model)
    print('Running inference on', args.source)
    results = model.predict(source=args.source, imgsz=args.imgsz, save=False, verbose=False)
    if len(results) == 0:
        print('No results')
        return
    r = results[0]
    img = cv2.imread(args.source)
    if img is None:
        print('Failed to read image', args.source)
        return
    # scale image for display if requested and draw on the scaled image so labels remain readable
    try:
        scale = float(args.scale)
    except Exception:
        scale = 1.0
    if scale <= 0:
        scale = 1.0
    h, w = img.shape[:2]
    disp_w = max(1, int(w * scale))
    disp_h = max(1, int(h * scale))
    disp = cv2.resize(img.copy(), (disp_w, disp_h), interpolation=cv2.INTER_AREA) if scale != 1.0 else img.copy()

    if hasattr(r, 'boxes') and r.boxes is not None and len(r.boxes):
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clsids = r.boxes.cls.cpu().numpy().astype(int)
        # compute moderate font_scale and thickness so labels remain readable but not huge
        # use a gentle inverse dependence on scale with capped values
        font_scale = 0.6 * (1.0 / max(scale, 0.1)) ** 0.25
        font_scale = max(0.3, min(font_scale, 1.5))
        thickness = max(1, int(round(1.5 * (1.0 / max(scale, 0.1)) ** 0.25)))
        thickness = min(thickness, 4)
        for (x1, y1, x2, y2), cid, conf in zip(boxes, clsids, confs):
            if conf < args.conf:
                continue
            color = (0,255,0) if int(cid)==0 else (0,0,255)
            sx1, sy1, sx2, sy2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)
            # draw single rectangle (no large black outline)
            cv2.rectangle(disp, (sx1, sy1), (sx2, sy2), color, thickness)
            label = f"{model.names[int(cid)]} {conf:.2f}"
            # draw text with thin black outline for contrast (no filled background)
            cv2.putText(disp, label, (sx1 + 2, sy1 - 6), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness+1, cv2.LINE_AA)
            # make main text green and slightly thinner for readability
            text_thickness = max(1, thickness - 1)
            cv2.putText(disp, label, (sx1 + 2, sy1 - 6), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,0), text_thickness, cv2.LINE_AA)

    cv2.imshow('detections', disp)
    print('Press any key in the image window to close it')
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
