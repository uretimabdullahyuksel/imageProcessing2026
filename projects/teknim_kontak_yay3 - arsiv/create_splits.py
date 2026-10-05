#!/usr/bin/env python3
"""
Create train/val/test splits for a YOLO dataset (images + labels)

Usage:
  python create_splits.py --src images --labels labels --out . --train 0.8 --val 0.1 --test 0.1

Defaults assume your current workspace has images in `images/` and labels in `labels/`.
This script copies matching image/label pairs into `train/images`, `train/labels`, `val/...`, `test/...`.
"""
import argparse
import os
import random
import shutil
from pathlib import Path


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--src', default='images', help='Source images directory')
    p.add_argument('--labels', default='labels', help='Source labels directory')
    p.add_argument('--out', default='.', help='Project root where train/val/test folders will be created')
    p.add_argument('--train', type=float, default=0.8, help='Train fraction')
    p.add_argument('--val', type=float, default=0.1, help='Val fraction')
    p.add_argument('--test', type=float, default=0.1, help='Test fraction')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    return p.parse_args()


def ensure_dirs(root):
    for split in ('train', 'val', 'test'):
        (root / split / 'images').mkdir(parents=True, exist_ok=True)
        (root / split / 'labels').mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    random.seed(args.seed)

    src_images = Path(args.src)
    src_labels = Path(args.labels)
    root = Path(args.out)

    if not src_images.exists():
        print('Source images folder not found:', src_images)
        return

    img_files = [p for p in src_images.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    img_files = sorted(img_files)
    if len(img_files) == 0:
        print('No image files found in', src_images)
        return

    ensure_dirs(root)

    # shuffle and split
    random.shuffle(img_files)
    n = len(img_files)
    t = int(n * args.train)
    v = int(n * (args.train + args.val))

    splits = {
        'train': img_files[:t],
        'val': img_files[t:v],
        'test': img_files[v:]
    }

    counts = {}
    for split, files in splits.items():
        for img_path in files:
            stem = img_path.stem
            # copy image
            dst_img = root / split / 'images' / img_path.name
            shutil.copy2(img_path, dst_img)
            # copy label if exists
            src_lbl = src_labels / f"{stem}.txt"
            if src_lbl.exists():
                dst_lbl = root / split / 'labels' / f"{stem}.txt"
                shutil.copy2(src_lbl, dst_lbl)
        counts[split] = len(files)

    print('Splits created under', root)
    for s, c in counts.items():
        print(f"  {s}: {c} images")
    print('Note: label files are copied only if a matching .txt exists in the labels folder.')


if __name__ == '__main__':
    main()
