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

from models.model_engine import get_model
from models.augmentations import get_augmentation_pipeline, mixup_batch


def main(args):
    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        raise SystemExit(f"Dataset directory not found: {dataset_dir}")

    image_size = args.image_size
    batch_size = args.batch_size

    print("Loading datasets from:", dataset_dir)
    # Load and convert to grayscale
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_dir,
        labels="inferred",
        label_mode="int",
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        color_mode="grayscale",  # Load as grayscale
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
        color_mode="grayscale",  # Load as grayscale
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print("Classes:", class_names)

    AUTOTUNE = tf.data.AUTOTUNE

    # Build augmentation pipeline based on CLI flags
    aug_choices = args.augment or []
    data_augmentation = get_augmentation_pipeline(aug_choices, image_size=image_size)

    def apply_augment(x, y):
        return data_augmentation(x, training=True), y

    train_ds = train_ds.map(lambda x, y: apply_augment(x, y))
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    # If using mixup, convert labels to one-hot and apply mixup during training
    use_mixup = "mixup" in (args.augment or [])
    if use_mixup:
        # convert labels to one-hot
        num_classes = len(class_names)
        def one_hot_map(x, y):
            y = tf.one_hot(y, num_classes)
            return x, y

        train_ds = train_ds.map(one_hot_map)
        val_ds = val_ds.map(lambda x, y: (x, tf.one_hot(y, num_classes)))

    # build selected model from models package
    model = get_model(name=args.model, num_classes=num_classes, image_size=image_size, dropout_rate=0.4, pretrained=True)

    # if mixup used, use categorical loss and expect one-hot labels
    if use_mixup:
        loss = tf.keras.losses.CategoricalCrossentropy()
    else:
        loss = tf.keras.losses.SparseCategoricalCrossentropy()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss,
        metrics=["accuracy"],
    )

    out_dir = Path("saved_models")
    out_dir.mkdir(exist_ok=True)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=out_dir / "best_model.h5", monitor="val_accuracy", save_best_only=True
    )
    earlystop_cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    reduce_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)

    # If mixup, we need to apply mixing on the batches before passing to model.fit
    if use_mixup:
        # wrapper generator to apply mixup on the fly
        def mixup_map(x, y):
            mixed_x, mixed_y = mixup_batch(x, y, alpha=args.mixup_alpha)
            return mixed_x, mixed_y

        train_ds_mix = train_ds.map(lambda x, y: tf.py_function(lambda a, b: mixup_batch(a, b, args.mixup_alpha), [x, y], Tout=[tf.float32, tf.float32]))
        # Note: tf.py_function drops shape information; provide without shapes for now
        history = model.fit(
            train_ds_mix,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=[checkpoint_cb, earlystop_cb, reduce_cb],
        )
    else:
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
    parser.add_argument(
        "--model",
        type=str,
        default="efficientnet",
        choices=["efficientnet", "customcnn", "mobilenetv3", "resnet"],
        help="Model architecture to use (default: efficientnet)",
    )
    parser.add_argument(
        "--augment",
        type=str,
        nargs="*",
        default=None,
        choices=["basic", "color", "geometric", "cutout", "mixup"],
        help="List of augmentations to apply. Use 'mixup' for mixup training. Example: --augment basic color",
    )
    parser.add_argument("--mixup_alpha", type=float, default=0.2, help="Alpha parameter for mixup Beta distribution")
    parser.add_argument("--cutout_size", type=int, default=32, help="Cutout mask size (pixels) if cutout used")
    args = parser.parse_args()
    main(args)
