#!/usr/bin/env python3
"""
real_capture.py

Run real-time inference on a webcam and display class predictions and boxes.

Usage:
  python real_capture.py --model runs/.../weights/best.pt --cam 0 --conf 0.6 --scale 0.5

Controls:
  q or ESC - quit
  p         - pause/resume
  c         - save a snapshot to `snapshots/` (created automatically)

Notes:
  - This script uses `ultralytics.YOLO` to run inference on each frame. For best
    performance use a smaller `--imgsz` or a GPU-enabled environment.
"""
import os
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def draw_boxes_on_disp(disp, boxes, clsids, confs, names, scale, conf_thresh, font_scale_base=0.6):
    """Draw detection boxes and labels onto the display image.

    Parameters:
    - disp: display image (possibly scaled down)
    - boxes: array of xyxy boxes in the original frame coordinates
    - clsids: class id array
    - confs: confidence scores array
    - names: model.names mapping
    - scale: factor used to convert original coords -> display coords
    - conf_thresh: minimum confidence to show
    """
    h, w = disp.shape[:2]
    # font and thickness adapt to display size; keep labels readable but small
    font_scale = max(0.3, min(1.2, font_scale_base * (1.0)))
    thickness = max(1, int(round(1.2)))
    for (x1, y1, x2, y2), cid, conf in zip(boxes, clsids, confs):
        if conf < conf_thresh:
            continue
        # scale coords for display
        sx1, sy1, sx2, sy2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)
        color = (0, 255, 0) if int(cid) == 0 else (0, 0, 255)
        cv2.rectangle(disp, (sx1, sy1), (sx2, sy2), color, thickness)
        label = f"{names[int(cid)]} {conf:.2f}"
        cv2.putText(disp, label, (sx1 + 2, max(15, sy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(disp, label, (sx1 + 2, max(15, sy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), max(1, thickness - 1), cv2.LINE_AA)


# Configuration: edit these values directly in the file
CONFIG = {
    # Path to the trained weights you want to use. Use relative path from project root.
    'MODEL': 'runs/pso_final/pso_it1_s2_1780619305/weights/best.pt',
    # Camera index (0 = default webcam). Set to video file path later if needed.
    'CAM': 0,
    # Inference image size (send to model.predict). Lower is faster on CPU.
    'IMGSZ': 640,
    # Confidence threshold for displaying detections (0.0 - 1.0)
    'CONF': 0.6,
    # Display scale: how much to shrink the window for visibility (0.1 - 1.0)
    'SCALE': 0.6,
    # Folder where snapshots are saved when pressing 'c'
    'SAVE_DIR': 'snapshots',
}

def _warn_if_args_present():
    # Keep argparse import but discourage using CLI; this helps detect accidental extra args.
    import sys
    if len(sys.argv) > 1:
        print('Note: this script reads configuration from the CONFIG dict at the top of the file.')
        print('Command-line arguments are ignored. Edit CONFIG in the file to change settings.')


def main():
    # Warn if user passed CLI args (we ignore them) and load config values
    _warn_if_args_present()
    model_path = CONFIG['MODEL']
    cam = CONFIG['CAM']
    imgsz = CONFIG['IMGSZ']
    conf_thresh = CONFIG['CONF']
    scale = CONFIG['SCALE']
    save_dir = CONFIG['SAVE_DIR']

    print('Loading model', model_path)
    model = YOLO(model_path)

    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        print('Failed to open camera', cam)
        return

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    paused = False
    last_time = time.time()
    fps = 0.0

    # Controls: q/ESC quit, p pause/resume, c save snapshot
    print('Press q or ESC to quit, p to pause, c to save a snapshot')
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print('Camera read failed')
                break
        else:
            # when paused, just display last frame
            time.sleep(0.05)

        h, w = frame.shape[:2]
        scale = max(0.01, float(scale))
        disp_w = max(1, int(w * scale))
        disp_h = max(1, int(h * scale))
        disp = cv2.resize(frame.copy(), (disp_w, disp_h), interpolation=cv2.INTER_AREA) if scale != 1.0 else frame.copy()

        if not paused:
            # Run model inference on the current raw frame. We pass the
            # original `frame` so predictions are in original image coords.
            # Note: on CPU this may be slow; reduce `IMGSZ` for better FPS.
            results = model.predict(source=frame, imgsz=imgsz, save=False, verbose=False)
            if len(results) > 0:
                r = results[0]
                if hasattr(r, 'boxes') and r.boxes is not None and len(r.boxes):
                    boxes = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    clsids = r.boxes.cls.cpu().numpy().astype(int)
                    draw_boxes_on_disp(disp, boxes, clsids, confs, model.names, scale, conf_thresh)

        # compute FPS
        now = time.time()
        dt = now - last_time
        last_time = now
        fps = 0.9 * fps + 0.1 * (1.0 / dt) if dt > 0 else fps

        cv2.putText(disp, f'FPS: {fps:.1f}', (10, disp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('real_capture', disp)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('c'):
            ts = int(time.time())
            out_path = Path(save_dir) / f'snap_{ts}.jpg'
            cv2.imwrite(str(out_path), frame)
            print('Saved', out_path)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
