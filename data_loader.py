"""Load and preprocess Fashion-MNIST.

Normalization is identical for train, validation and test splits.
Augmentation is NOT done here; it lives inside the model and only runs
during training, so validation and test data are never altered.
"""
import numpy as np
import tensorflow as tf

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def _preprocess(x):
    """Scale pixels to [0, 1] and add a channel axis: (N, 28, 28) -> (N, 28, 28, 1)."""
    x = x.astype("float32") / 255.0
    return x[..., np.newaxis]


def load_fashion_mnist(val_fraction=0.1, seed=42):
    """Return a dict with 'train', 'val' and 'test' (images, labels) tuples.

    The validation split is carved out of the official training set with a
    fixed seed, so the split is reproducible. The official test set is kept
    untouched for final evaluation only.
    """
    (x_full, y_full), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_full, x_test = _preprocess(x_full), _preprocess(x_test)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x_full))
    n_val = int(len(x_full) * val_fraction)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    return {
        "train": (x_full[train_idx], y_full[train_idx]),
        "val": (x_full[val_idx], y_full[val_idx]),
        "test": (x_test, y_test),
    }


def summarize(splits):
    """Human-readable dataset summary: sample counts, image shape and class balance."""
    lines = []
    for name, (x, y) in splits.items():
        counts = np.bincount(y, minlength=len(CLASS_NAMES))
        lines.append(f"{name:<6} n={len(x):<6} shape={x.shape[1:]} per-class={counts.tolist()}")
    return "\n".join(lines)
