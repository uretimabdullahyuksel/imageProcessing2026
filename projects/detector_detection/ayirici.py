import os, random, shutil

SRC_DIR = "specialData3"     # sende 0/1/2 burada
OUT_DIR = "dataset"          # oluşturulacak
VAL_RATIO = 0.2
SEED = 42

random.seed(SEED)

classes = sorted([d for d in os.listdir(SRC_DIR) if os.path.isdir(os.path.join(SRC_DIR, d))],
                 key=lambda x: int(x))

for split in ["train", "val"]:
    for c in classes:
        os.makedirs(os.path.join(OUT_DIR, split, c), exist_ok=True)

for c in classes:
    src_c = os.path.join(SRC_DIR, c)
    imgs = [f for f in os.listdir(src_c) if f.lower().endswith((".jpg",".jpeg",".png"))]
    random.shuffle(imgs)

    val_count = int(len(imgs) * VAL_RATIO)
    val_imgs = imgs[:val_count]
    train_imgs = imgs[val_count:]

    for f in train_imgs:
        shutil.copy2(os.path.join(src_c, f), os.path.join(OUT_DIR, "train", c, f))
    for f in val_imgs:
        shutil.copy2(os.path.join(src_c, f), os.path.join(OUT_DIR, "val", c, f))

    print(f"Class {c}: train={len(train_imgs)} val={len(val_imgs)}")

print("\nHazır: dataset/train ve dataset/val")
