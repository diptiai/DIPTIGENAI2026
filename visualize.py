"""Figures for the report: dataset samples, learning curves, confusion matrix,
and misclassified images."""
import matplotlib

matplotlib.use("Agg")  # headless-safe; no display required

import matplotlib.pyplot as plt
import numpy as np

from src.data_loader import CLASS_NAMES


def plot_dataset_samples(x, y, out_path):
    """One example image per class."""
    fig, axes = plt.subplots(1, len(CLASS_NAMES), figsize=(15, 2))
    for c, ax in enumerate(axes):
        i = np.where(y == c)[0][0]
        ax.imshow(x[i].squeeze(), cmap="gray")
        ax.set_title(CLASS_NAMES[c], fontsize=8)
        ax.axis("off")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_learning_curves(history, out_path):
    """Training vs validation accuracy and loss per epoch."""
    epochs = range(1, len(history["loss"]) + 1)
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(11, 4))

    ax_acc.plot(epochs, history["accuracy"], marker="o", label="train")
    ax_acc.plot(epochs, history["val_accuracy"], marker="o", label="validation")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    ax_loss.plot(epochs, history["loss"], marker="o", label="train")
    ax_loss.plot(epochs, history["val_loss"], marker="o", label="validation")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm, out_path):
    """Class-by-class prediction counts. Rows = true class, columns = predicted."""
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues")
    ticks = range(len(CLASS_NAMES))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix (test set)")

    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, cm[i, j], ha="center", va="center", fontsize=8,
                color="white" if cm[i, j] > threshold else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_misclassified(x, y_true, y_pred, confidence, out_path, n=16):
    """Show the most confident mistakes: they reveal systematic failures,
    not just borderline cases."""
    wrong = np.where(y_true != y_pred)[0]
    if len(wrong) == 0:
        print("No misclassified images to plot.")
        return

    order = wrong[np.argsort(-confidence[wrong])][:n]
    cols = 8
    rows = int(np.ceil(len(order) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.9), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    for ax, i in zip(axes.ravel(), order):
        ax.imshow(x[i].squeeze(), cmap="gray")
        ax.set_title(
            f"True: {CLASS_NAMES[y_true[i]]}\n"
            f"Pred: {CLASS_NAMES[y_pred[i]]} ({confidence[i]:.2f})",
            fontsize=7,
        )

    fig.suptitle("Most confident misclassifications (test set)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
