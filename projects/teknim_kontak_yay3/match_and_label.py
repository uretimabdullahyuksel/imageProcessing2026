#!/usr/bin/env python3
"""
Match cropped examples to full images and produce YOLO-format labels.

Usage:
  python match_and_label.py --full_dir full_images --examples 0:yayli 1:yaysiz --out out_dir

The `--examples` argument takes one or more mappings of folder:labelname where
the folder contains example crops for that class. Example: `0:yayli 1:yaysiz`.

The script uses template matching (cv2.matchTemplate) for each example image
against each full image. Matches above `--thresh` produce a bounding box and
are saved as YOLO labels in `out/labels` and annotated images in `out/annotated`.

NOTE: This is a heuristic approach; if scale/rotation differ you may need
feature-matching (ORB/FLANN) or manual annotation.
"""
import argparse
import os
from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict


def parse_examples_arg(specs):
    mapping = {}
    for s in specs:
        if ':' in s:
            folder, label = s.split(':', 1)
        else:
            folder, label = s, s
        mapping[folder] = label
    return mapping


def non_max_suppression(boxes, scores, iou_threshold=0.3):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def yolo_box_from_xyxy(x1, y1, x2, y2, img_w, img_h):
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return cx, cy, w, h


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--full_dir', required=True, help='Folder with full images')
    p.add_argument('--examples', required=True, nargs='+', help='Example folders mapping e.g. 0:yayli 1:yaysiz')
    p.add_argument('--out', default='out', help='Output folder')
    p.add_argument('--thresh', type=float, default=0.7, help='Template match threshold (0..1)')
    p.add_argument('--iou', type=float, default=0.3, help='NMS IoU threshold')
    p.add_argument('--max_per_example', type=int, default=10, help='Max matches per example image per full image')
    args = p.parse_args()

    ex_map = parse_examples_arg(args.examples)
    out = Path(args.out)
    labels_dir = out / 'labels'
    ann_dir = out / 'annotated'
    labels_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    # Build list of example images per class
    examples = []  # list of (class_id, label_name, img_path)
    class_names = []
    for i, (folder, label) in enumerate(ex_map.items()):
        class_names.append(label)
        class_id = i
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"Warning: example folder not found: {folder_path}")
            continue
        for img in sorted(folder_path.iterdir()):
            if img.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                examples.append((class_id, label, img))

    full_paths = sorted(Path(args.full_dir).glob('*'))
    summary = defaultdict(int)

    for full_path in full_paths:
        full = cv2.imread(str(full_path))
        if full is None:
            continue
        full_gray = cv2.cvtColor(full, cv2.COLOR_BGR2GRAY)
        h_full, w_full = full_gray.shape[:2]
        all_boxes = []
        all_scores = []
        all_class_ids = []

        for class_id, label, ex_path in examples:
            ex = cv2.imread(str(ex_path))
            if ex is None:
                continue
            ex_gray = cv2.cvtColor(ex, cv2.COLOR_BGR2GRAY)
            h_ex, w_ex = ex_gray.shape[:2]
            if h_ex >= h_full or w_ex >= w_full:
                continue

            res = cv2.matchTemplate(full_gray, ex_gray, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= args.thresh)
            scores = res[loc]
            pts = list(zip(*loc[::-1]))  # (x,y)
            # sort matches by score desc
            matches = sorted(zip(pts, scores), key=lambda x: x[1], reverse=True)[:args.max_per_example]

            for (pt, score) in matches:
                x, y = pt
                x1 = int(x)
                y1 = int(y)
                x2 = int(x + w_ex)
                y2 = int(y + h_ex)
                all_boxes.append([x1, y1, x2, y2])
                all_scores.append(float(score))
                all_class_ids.append(int(class_id))

        # apply NMS
        keep = non_max_suppression(all_boxes, all_scores, iou_threshold=args.iou)
        kept = [i for i in keep]

        # write labels for this full image
        base = full_path.stem
        label_file = labels_dir / f"{base}.txt"
        with open(label_file, 'w', encoding='utf-8') as f:
            for i in kept:
                cls = all_class_ids[i]
                x1, y1, x2, y2 = all_boxes[i]
                cx, cy, w, h = yolo_box_from_xyxy(x1, y1, x2, y2, w_full, h_full)
                f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                summary[cls] += 1

        # draw annotations
        for i in kept:
            cls = all_class_ids[i]
            x1, y1, x2, y2 = all_boxes[i]
            cv2.rectangle(full, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(full, f"{cls}:{class_names[cls]}", (x1, max(15, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.imwrite(str(ann_dir / full_path.name), full)
        print(f"Processed {full_path.name}: {len(kept)} matches")

    print('\nSummary:')
    for cls, count in summary.items():
        label = class_names[cls] if cls < len(class_names) else str(cls)
        print(f"  class {cls} ({label}): {count}")


if __name__ == '__main__':
    main()
