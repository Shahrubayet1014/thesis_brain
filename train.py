"""
train.py

Simple training script for the brain tumor image dataset located at
Dataset/Brain with subfolders: Glioma_tumor, Meningioma_tumor, No_tumor, Pituitary_tumor

Usage (PowerShell):
  python train.py --dataset Dataset/Brain --epochs 10 --batch_size 16

The script uses TensorFlow Keras and transfer learning (EfficientNetB0).
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


def build_model(num_classes: int, image_size: int, dropout_rate: float = 0.4):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(image_size, image_size, 3)
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)

    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model


def main(args):
    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        raise SystemExit(f"Dataset directory not found: {dataset_dir}")

    image_size = args.image_size
    batch_size = args.batch_size

    print("Loading datasets from:", dataset_dir)
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_dir,
        labels="inferred",
        label_mode="int",
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(image_size, image_size),
        batch_size=batch_size,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_dir,
        labels="inferred",
        label_mode="int",
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(image_size, image_size),
        batch_size=batch_size,
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print("Classes:", class_names)

    AUTOTUNE = tf.data.AUTOTUNE

    # Basic augmentation
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.06),
        ]
    )

    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    model = build_model(num_classes=num_classes, image_size=image_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    out_dir = Path("saved_models")
    out_dir.mkdir(exist_ok=True)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=out_dir / "best_model.h5", monitor="val_accuracy", save_best_only=True
    )
    earlystop_cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    reduce_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[checkpoint_cb, earlystop_cb, reduce_cb],
    )

    # Save final model and history
    model.save(out_dir / "final_model")
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=2)

    # Evaluation: predict on validation set and print a classification report
    y_true = []
    y_pred = []
    for images, labels in val_ds:
        probs = model.predict(images)
        preds = np.argmax(probs, axis=-1)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.tolist())

    print("\nClassification report on validation set:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    # Save reports
    rep = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    with open(out_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Dataset/Brain", help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()
    main(args)
