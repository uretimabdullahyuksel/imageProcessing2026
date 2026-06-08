#!/usr/bin/env python3
"""
infer_capture.py

Run inference with the trained YOLO model and visualize/save detections similar
to `image_capture.py` but focused on single-image or folder runs with a confidence
threshold and simple class filtering.

Usage examples:
  python infer_capture.py --source images2/7.jpg --model runs/pso_final/pso_it1_s2_1780619305/weights/best.pt --conf 0.6 --save
  python infer_capture.py --source images2 --model runs/pso_final/pso_it1_s2_1780619305/weights/best.pt --conf 0.6 --save --display

"""
import argparse
import os
from ultralytics import YOLO
import cv2

CLASS_COLORS = {
    0: (0, 255, 0),  # yayli - green
    1: (0, 0, 255),  # yaysiz - red
}


def draw_boxes_on_image(img, boxes, clsids, confs, names, conf_thresh=0.0):
    for (x1, y1, x2, y2), cid, conf in zip(boxes, clsids, confs):
        if conf < conf_thresh:
            continue
        color = CLASS_COLORS.get(int(cid), (255, 255, 0))
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{names[int(cid)]} {conf:.2f}"
        cv2.putText(img, label, (x1, max(15, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)
    return img


def process_results(results, model, conf_thresh, out_dir, save, display):
    os.makedirs(out_dir, exist_ok=True)
    for r in results:
        # r.path is the image path used by ultralytics
        img_path = getattr(r, 'path', None) or getattr(r, 'orig_img_path', None)
        if img_path is None:
            continue
        img = cv2.imread(img_path)
        if img is None:
            print('Warning: failed to read', img_path)
            continue
        if hasattr(r, 'boxes') and r.boxes is not None and len(r.boxes):
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clsids = r.boxes.cls.cpu().numpy().astype(int)
            img_out = draw_boxes_on_image(img.copy(), boxes, clsids, confs, model.names, conf_thresh)
        else:
            img_out = img
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(out_dir, f"{base}_annot_conf{int(conf_thresh*100):02d}.jpg")
        if save:
            cv2.imwrite(out_path, img_out)
            print('Saved', out_path)
        if display:
            cv2.imshow('detections', img_out)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                return
    if display:
        cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True, help='Image file or directory (images)')
    p.add_argument('--model', default='runs/pso_final/pso_it1_s2_1780619305/weights/best.pt', help='Path to model weights')
    p.add_argument('--imgsz', type=int, default=640, help='Inference image size')
    p.add_argument('--conf', type=float, default=0.5, help='Confidence threshold to show boxes')
    p.add_argument('--save', action='store_true', help='Save annotated images')
    p.add_argument('--display', action='store_true', help='Display images interactively')
    p.add_argument('--out', default='runs/infer_capture', help='Output folder for saved images')
    args = p.parse_args()

    model = YOLO(args.model)
    print('Loaded model', args.model)

    # Ultralyics predict accepts a directory or a file
    results = model.predict(source=args.source, imgsz=args.imgsz, save=False, verbose=False)
    process_results(results, model, args.conf, args.out, args.save, args.display)


if __name__ == '__main__':
    main()
