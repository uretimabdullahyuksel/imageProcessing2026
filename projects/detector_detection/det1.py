# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model


# -------------------------
# AYARLAR
# -------------------------
DATA_DIR = "specialData3"
IMG_SIZE = 128          # MobileNet için 128 iyi başlangıç
BATCH_SIZE = 32
EPOCHS_HEAD = 10        # sadece classifier eğitimi
EPOCHS_FINE = 10        # az biraz fine-tune
LR_HEAD = 1e-3
LR_FINE = 1e-5
SEED = 42

# göstergenin merkezini crop'la (arka planı azaltır)
CENTER_CROP_RATIO = 0.75   # 0.60-0.85 arası deneyebilirsin


def list_class_folders(data_dir: str):
    cls = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    cls = sorted(cls, key=lambda x: int(x))
    return cls


def center_crop(img, ratio=0.75):
    h, w = img.shape[:2]
    s = int(min(h, w) * ratio)
    cy, cx = h // 2, w // 2
    y1 = max(0, cy - s // 2)
    y2 = min(h, cy + s // 2)
    x1 = max(0, cx - s // 2)
    x2 = min(w, cx + s // 2)
    return img[y1:y2, x1:x2]


def load_dataset(data_dir: str, img_size: int):
    class_dirs = list_class_folders(data_dir)
    if len(class_dirs) == 0:
        raise RuntimeError(f"'{data_dir}' içinde sınıf klasörü yok (0/1/2 olmalı).")

    X, y = [], []
    bad = 0

    for cls in class_dirs:
        cls_path = os.path.join(data_dir, cls)
        for fn in os.listdir(cls_path):
            fp = os.path.join(cls_path, fn)
            img = cv2.imread(fp)
            if img is None:
                bad += 1
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # ARKA PLANI AZALT: merkez crop
            img = center_crop(img, CENTER_CROP_RATIO)

            img = cv2.resize(img, (img_size, img_size))
            X.append(img)
            y.append(int(cls))

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    print("Bozuk/okunamayan:", bad)
    return X, y, len(class_dirs)


def make_augmenter():
    # Keras augmentation katmanları (GPU/CPU fark etmez)
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.10),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomTranslation(0.08, 0.08),
        tf.keras.layers.RandomContrast(0.10),
    ])


def build_mobilenet(num_classes: int, img_size: int):
    inp = Input(shape=(img_size, img_size, 3))
    x = make_augmenter()(inp)
    x = preprocess_input(x)  # MobileNetV2 için şart

    base = MobileNetV2(include_top=False, weights="imagenet", input_tensor=x)
    base.trainable = False   # önce dondur

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation="softmax")(x)

    model = Model(inp, out)
    return model, base


def main():
    X, y, num_classes = load_dataset(DATA_DIR, IMG_SIZE)
    print("Toplam:", len(X))
    print("Dağılım:", Counter(y))

    # stratify ile böl
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.4, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.3, random_state=SEED, stratify=y_tmp
    )

    print("TRAIN:", Counter(y_train))
    print("VAL  :", Counter(y_val))
    print("TEST :", Counter(y_test))

    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat   = to_categorical(y_val, num_classes)
    y_test_cat  = to_categorical(y_test, num_classes)

    model, base = build_mobilenet(num_classes, IMG_SIZE)

    early = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    # 1) Head training
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_HEAD),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=EPOCHS_HEAD,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=[early],
        verbose=1
    )

    # 2) Fine-tune (son blokları aç)
    base.trainable = True
    # çok açma: son ~30 layer kalsın trainable
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR_FINE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=EPOCHS_FINE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=[early],
        verbose=1
    )

    # Kaydet
    model.save("mobilenet_3class.keras")
    print("Model kaydedildi: mobilenet_3class.keras")

    # Test
    loss, acc = model.evaluate(X_test, y_test_cat, verbose=1)
    print("Test loss:", loss)
    print("Test acc :", acc)

    y_pred = np.argmax(model.predict(X_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion:\n", cm)

    names = [f"Class {i}" for i in range(num_classes)]
    print("\nReport:\n", classification_report(y_test, y_pred, target_names=names, zero_division=0))


if __name__ == "__main__":
    main()
