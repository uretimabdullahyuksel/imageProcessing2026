#!/usr/bin/env python3
"""
Simple PSO hyperparameter tuning scaffold for YOLO (ultralytics).

This script provides a safe, minimal PSO implementation that launches
separate training runs for different hyperparameter candidates.

IMPORTANT:
- Training many candidates is expensive. Use small `swarm_size` and
  `iters` for quick experiments (e.g., swarm_size=4, iters=3).
- The script calls `YOLO(...).train(...)` directly. Confirm the
  ultralytics API on your environment; adjust param names if needed.

Usage (example):
    python train_with_pso.py --data data.yaml --base yolo26n.pt --epochs 10

"""
import argparse
import csv
import os
import time
import random
from copy import deepcopy
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='Path to data.yaml')
    p.add_argument('--base', default='yolo26n.pt', help='Base model (yolo26n.pt etc)')
    p.add_argument('--project', default='runs/pso', help='Output project folder')
    p.add_argument('--swarm', type=int, default=4, help='PSO swarm size (small for tests)')
    p.add_argument('--iters', type=int, default=3, help='PSO iterations (small for tests)')
    p.add_argument('--epochs', type=int, default=5, help='Epochs per candidate (use >50 for real)')
    p.add_argument('--imgsz', type=int, default=640, help='Train image size')
    return p.parse_args()


# simple mapping from normalized [0,1] vector to real hyperparam ranges
HYPER_RANGES = {
    'lr': (1e-4, 1e-2),          # learning rate
    'momentum': (0.8, 0.99),     # SGD momentum
    'weight_decay': (0.0, 5e-4), # weight decay
    'batch': (2, 8),             # batch size (int)
}


def denormalize(p):
    """Map normalized particle p (dict of 0..1) to real hyperparams."""
    return {
        'lr': HYPER_RANGES['lr'][0] + p['lr'] * (HYPER_RANGES['lr'][1] - HYPER_RANGES['lr'][0]),
        'momentum': HYPER_RANGES['momentum'][0] + p['momentum'] * (HYPER_RANGES['momentum'][1] - HYPER_RANGES['momentum'][0]),
        'weight_decay': HYPER_RANGES['weight_decay'][0] + p['weight_decay'] * (HYPER_RANGES['weight_decay'][1] - HYPER_RANGES['weight_decay'][0]),
        'batch': int(round(HYPER_RANGES['batch'][0] + p['batch'] * (HYPER_RANGES['batch'][1] - HYPER_RANGES['batch'][0]))),
    }


def random_particle():
    return {k: random.random() for k in HYPER_RANGES.keys()}


def parse_results_csv(csv_path):
    if not os.path.exists(csv_path):
        return None
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        last_row = None
        for row in reader:
            last_row = row
        if last_row is None:
            return None
        try:
            return float(last_row.get('metrics/mAP50(B)', 0.0))
        except ValueError:
            return None


def train_candidate(base, data, out_project, name, hp, epochs, imgsz):
    """Run a training job for given hyperparams. Returns a dict with run info.

    Note: This will be slow. Use small epochs for quick tests.
    """
    out_project = os.path.abspath(out_project)
    print(f"Training {name} with hp={hp} project={out_project}")
    model = YOLO(base)
    model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=hp['batch'],
        lr0=hp['lr'],
        project=out_project,
        name=name,
        verbose=False
    )

    run_folder = os.path.join(out_project, name)
    results_csv = os.path.join(run_folder, 'results.csv')
    val_map50 = parse_results_csv(results_csv)
    res = {
        'name': name,
        'hp': hp,
        'run_folder': run_folder,
        'val_map50': val_map50,
        'results_csv': results_csv,
    }
    return res


def pso_optimize(args):
    swarm_size = args.swarm
    iters = args.iters

    # initialize particles
    particles = [random_particle() for _ in range(swarm_size)]
    best_particle = None
    best_score = -1.0

    for it in range(iters):
        print(f"PSO iteration {it+1}/{iters}")
        for i, p in enumerate(particles):
            hp = denormalize(p)
            run_name = f"pso_it{it+1}_s{i+1}_{int(time.time())}"
            info = train_candidate(args.base, args.data, args.project, run_name, hp, args.epochs, args.imgsz)
            if info['val_map50'] is not None:
                score = info['val_map50']
                print(f"  run={info['name']} folder={info['run_folder']} val_map50={score:.4f}")
            else:
                score = hp['batch'] * hp['lr']
                print(f"  run={info['name']} folder={info['run_folder']} heuristic_score={score:.6g}")
            if score > best_score:
                best_score = score
                best_particle = deepcopy((p, hp, info))

        # simple particle update: random walk around current best
        for i in range(len(particles)):
            if best_particle is None:
                particles[i] = random_particle()
            else:
                bp = best_particle[0]
                # small gaussian perturbation
                particles[i] = {k: min(1.0, max(0.0, bp[k] + random.gauss(0, 0.08))) for k in bp}

    print("PSO done. Best candidate:")
    print(best_particle)
    if best_particle is not None and best_particle[2].get('val_map50') is not None:
        print(f"Best val_map50={best_particle[2]['val_map50']:.4f}")
    else:
        print("Best candidate chosen by heuristic because no results.csv metric was found.")


if __name__ == '__main__':
    args = parse_args()
    args.project = os.path.abspath(args.project)
    os.makedirs(args.project, exist_ok=True)
    pso_optimize(args)
