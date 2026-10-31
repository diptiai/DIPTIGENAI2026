"""Train one CNN experiment on Fashion-MNIST.

Run from the project root, one experiment at a time:
    python -m src.train --experiment baseline
    python -m src.train --experiment e1_arch
    python -m src.train --experiment e2_dropout
    python -m src.train --experiment e3_augment

Each run saves:
    outputs/models/<experiment>.keras              best checkpoint (by validation loss)
    outputs/experiments/<experiment>_history.json  per-epoch metrics
    outputs/experiments/<experiment>_summary.json  config + best-epoch metrics
The test set is never used here.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.data_loader import load_fashion_mnist, summarize
from src.model import build_cnn
from src.visualize import plot_dataset_samples

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SEED = 42

# Experiments build on each other (handbook 2.7). Each step adds one major change:
#   baseline   -> simple CNN
#   e1_arch    -> deeper/wider architecture (third conv block, more filters)
#   e2_dropout -> E1 + dropout before the classifier
#   e3_augment -> E2 + training-time augmentation
EXPERIMENTS = {
    "baseline":   {"filters": [32, 64],      "dropout": 0.0, "augment": False},
    "e1_arch":    {"filters": [32, 64, 128], "dropout": 0.0, "augment": False},
    "e2_dropout": {"filters": [32, 64, 128], "dropout": 0.3, "augment": False},
    "e3_augment": {"filters": [32, 64, 128], "dropout": 0.3, "augment": True},
}


def main():
    parser = argparse.ArgumentParser(description="Train a Fashion-MNIST CNN experiment.")
    parser.add_argument("--experiment", choices=list(EXPERIMENTS), required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3,
                        help="Early-stopping patience on validation loss.")
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(SEED)

    config = {
        "experiment": args.experiment,
        **EXPERIMENTS[args.experiment],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "patience": args.patience,
        "optimizer": "adam",
        "seed": SEED,
    }

    # ---- Data (step 01: inspect) ----
    data = load_fashion_mnist(seed=SEED)
    x_train, y_train = data["train"]
    x_val, y_val = data["val"]
    print("Dataset summary:")
    print(summarize({"train": data["train"], "val": data["val"]}))

    samples_path = OUT / "dataset_samples.png"
    if not samples_path.exists():
        plot_dataset_samples(x_train, y_train, samples_path)

    # ---- Model ----
    model = build_cnn(
        filters=config["filters"],
        dropout=config["dropout"],
        augment=config["augment"],
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # ---- Training ----
    checkpoint_path = OUT / "models" / f"{args.experiment}.keras"
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]

    start = time.time()
    fit = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    elapsed = time.time() - start

    # ---- Logging ----
    history = {k: [float(v) for v in vals] for k, vals in fit.history.items()}
    best = int(np.argmin(history["val_loss"]))  # the epoch the checkpoint corresponds to

    summary = {
        **config,
        "epochs_run": len(history["loss"]),
        "best_epoch": best + 1,
        "train_accuracy": history["accuracy"][best],
        "train_loss": history["loss"][best],
        "val_accuracy": history["val_accuracy"][best],
        "val_loss": history["val_loss"][best],
        "train_time_sec": round(elapsed, 1),
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
    }

    exp_dir = OUT / "experiments"
    (exp_dir / f"{args.experiment}_history.json").write_text(json.dumps(history, indent=2))
    (exp_dir / f"{args.experiment}_summary.json").write_text(json.dumps(summary, indent=2))

    print("\nExperiment summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
