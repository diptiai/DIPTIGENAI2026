"""Evaluate a trained experiment on the held-out test set and generate figures.

Run from the project root, after all experiments have been trained:
    python -m src.evaluate                       # selects best model by validation accuracy
    python -m src.evaluate --experiment e3_augment

Outputs:
    outputs/experiment_table.csv   baseline-to-final comparison (validation metrics)
    outputs/test_results.json      final test accuracy and per-class accuracy
    outputs/confusion_matrix.png
    outputs/misclassified.png
    outputs/learning_curves.png
"""
import argparse
import csv
import json
from pathlib import Path

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src.data_loader import CLASS_NAMES, load_fashion_mnist
from src.train import EXPERIMENTS, SEED
from src.visualize import plot_confusion_matrix, plot_learning_curves, plot_misclassified

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def load_summaries():
    """Read every experiment summary, in the order defined in EXPERIMENTS."""
    summaries = {}
    for path in (OUT / "experiments").glob("*_summary.json"):
        s = json.loads(path.read_text())
        summaries[s["experiment"]] = s
    return {name: summaries[name] for name in EXPERIMENTS if name in summaries}


def write_experiment_table(summaries):
    """Baseline-to-final comparison table. Uses validation metrics only."""
    columns = [
        "experiment", "filters", "dropout", "augment",
        "epochs_run", "best_epoch", "train_accuracy", "val_accuracy", "val_loss",
    ]
    path = OUT / "experiment_table.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for s in summaries.values():
            row = {k: s[k] for k in columns if k != "filters"}
            row["filters"] = "-".join(map(str, s["filters"]))
            writer.writerow(row)
    return path


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Fashion-MNIST CNN.")
    parser.add_argument("--experiment", choices=list(EXPERIMENTS), default=None,
                        help="Experiment to evaluate. Default: highest validation accuracy.")
    args = parser.parse_args()

    summaries = load_summaries()
    if not summaries:
        raise SystemExit("No experiment summaries found. Run `python -m src.train` first.")

    table_path = write_experiment_table(summaries)
    print(f"Experiment table written to {table_path}")

    # Model selection uses validation accuracy only, never test accuracy.
    if args.experiment:
        name = args.experiment
    else:
        name = max(summaries.values(), key=lambda s: s["val_accuracy"])["experiment"]
    print(f"Evaluating '{name}' on the test set.")

    model = tf.keras.models.load_model(OUT / "models" / f"{name}.keras")

    data = load_fashion_mnist(seed=SEED)
    x_test, y_test = data["test"]

    probs = model.predict(x_test, batch_size=256, verbose=0)
    y_pred = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    test_accuracy = float((y_pred == y_test).mean())

    print(f"\nTest accuracy: {test_accuracy:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=3))

    cm = confusion_matrix(y_test, y_pred)
    per_class = dict(zip(CLASS_NAMES, (cm.diagonal() / cm.sum(axis=1)).round(4).tolist()))

    plot_confusion_matrix(cm, OUT / "confusion_matrix.png")
    plot_misclassified(x_test, y_test, y_pred, confidence, OUT / "misclassified.png")

    history = json.loads((OUT / "experiments" / f"{name}_history.json").read_text())
    plot_learning_curves(history, OUT / "learning_curves.png")

    results = {
        "experiment": name,
        "test_accuracy": test_accuracy,
        "per_class_accuracy": per_class,
    }
    (OUT / "test_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved figures and test_results.json to {OUT}")


if __name__ == "__main__":
    main()
