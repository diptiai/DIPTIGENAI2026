"""CNN architecture for Fashion-MNIST.

Structure (handbook section 2.2):
    [augmentation (training only)] -> (Conv2D + ReLU -> MaxPool) x N
    -> Flatten -> Dense + ReLU -> [Dropout] -> Dense + Softmax
"""
import tensorflow as tf
from tensorflow.keras import layers


def build_augmentation():
    """Label-preserving transformations for clothing images.

    Keras preprocessing layers are only active during training (fit), so
    these never run on validation or test data.
    Horizontal flips are valid for these classes; rotations, translations
    and zoom are kept small so the garment stays recognizable.
    """
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.03),       # ~ +/- 11 degrees
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomZoom(0.1),
        ],
        name="augmentation",
    )


def build_cnn(
    input_shape=(28, 28, 1),
    num_classes=10,
    filters=(32, 64),
    dense_units=128,
    dropout=0.0,
    augment=False,
):
    """Build the CNN.

    Args:
        filters: number of filters per convolution block (one block per entry).
        dropout: dropout rate before the classifier; 0 disables it.
        augment: whether to prepend training-time augmentation.
    """
    inputs = layers.Input(shape=input_shape)
    x = build_augmentation()(inputs) if augment else inputs

    for i, f in enumerate(filters, start=1):
        x = layers.Conv2D(f, 3, padding="same", activation="relu", name=f"conv{i}")(x)
        x = layers.MaxPooling2D(2, name=f"pool{i}")(x)

    x = layers.Flatten()(x)
    x = layers.Dense(dense_units, activation="relu", name="dense")(x)
    if dropout > 0:
        x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return tf.keras.Model(inputs, outputs, name="fashion_cnn")
