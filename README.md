# Image Classifier Using CNN (Fashion-MNIST)

A convolutional neural network that classifies 28×28 grayscale images of clothing and accessories into 10 classes. The project starts from a baseline CNN and improves it step by step (architecture, dropout, augmentation), then analyzes errors rather than only reporting accuracy.

## Dataset

- **Fashion-MNIST**: 70,000 grayscale images, 28×28 pixels, 10 classes.
- Official split: 60,000 training images and 10,000 test images.
- A validation set of 10% of the training data (6,000 images) is held out with a fixed seed (42). The model is trained on the remaining 54,000.
- The test set is used **once per final model**, in `src/evaluate.py`. It is never used for training, early stopping or model selection.
- Dataset is downloaded automatically by Keras on first run and cached in `~/.keras/datasets/`.

Classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.

## Architecture

```
Input (28×28×1)
 → [Augmentation, training only]         (E3 onward)
 → Conv2D(f1, 3×3, ReLU) → MaxPool(2×2)
 → Conv2D(f2, 3×3, ReLU) → MaxPool(2×2)
 → [Conv2D(f3, 3×3, ReLU) → MaxPool(2×2)]   (E1 onward)
 → Flatten → Dense(128, ReLU)
 → [Dropout(0.3)]                        (E2 onward)
 → Dense(10, Softmax)
```

Preprocessing: pixel values are scaled to [0, 1] and the same scaling is applied to train, validation and test data.

## Experiments

Each experiment changes one major factor relative to the previous one.

| Experiment | Change | Filters | Dropout | Augmentation |
|---|---|---|---|---|
| `baseline` | Simple CNN, starting point | 32, 64 | – | – |
| `e1_arch` | Deeper/wider architecture | 32, 64, 128 | – | – |
| `e2_dropout` | Add dropout to reduce overfitting | 32, 64, 128 | 0.3 | – |
| `e3_augment` | Add label-preserving augmentation | 32, 64, 128 | 0.3 | Flip, rotation, translation, zoom |

Training settings (same for all runs): Adam, learning rate 1e-3, batch size 128, up to 20 epochs, early stopping on validation loss with patience 3, best weights restored, seed 42.

Augmentation uses small transformations: horizontal flip, rotation of about ±11°, translation of up to 10%, and zoom of up to 10%. These keep garments recognizable. Vertical flips and large rotations are not used because they can change how a garment looks (for example, a sneaker or a bag).

## Setup

Python 3.9–3.12 is required by TensorFlow 2.17.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The pinned versions are the ones this project was written against. Verify them in your environment before relying on exact numbers.

## Training

Run each experiment from the project root:

```bash
python -m src.train --experiment baseline
python -m src.train --experiment e1_arch
python -m src.train --experiment e2_dropout
python -m src.train --experiment e3_augment
```

Each run writes:
- `outputs/models/<experiment>.keras`: best checkpoint, selected by validation loss
- `outputs/experiments/<experiment>_history.json`: per-epoch metrics
- `outputs/experiments/<experiment>_summary.json`: configuration and best-epoch metrics
- `outputs/dataset_samples.png`: one example per class (created on the first run)

## Evaluation

After training, run:

```bash
python -m src.evaluate                       # picks the best model by validation accuracy
python -m src.evaluate --experiment e3_augment   # or evaluate a specific one
```

This writes:
- `outputs/experiment_table.csv`: baseline-to-final comparison (validation metrics)
- `outputs/test_results.json`: test accuracy and per-class accuracy
- `outputs/confusion_matrix.png`: class-by-class prediction counts
- `outputs/misclassified.png`: most confident errors, with true and predicted labels
- `outputs/learning_curves.png`: training vs validation accuracy and loss

Model selection uses validation accuracy only. The reported test accuracy is for the selected model and was not used to choose it.

## Results

Fill in after running the pipeline. Values come from `outputs/experiment_table.csv` and `outputs/test_results.json`.

| Experiment | Train acc | Val acc | Val loss | Observation |
|---|---|---|---|---|
| baseline | | | | Starting point |
| e1_arch | | | | Effect of added capacity |
| e2_dropout | | | | Train/validation gap |
| e3_augment | | | | Robustness |
| **Final (test)** | | | | Final interpretation |

Final test accuracy: _to be filled_

## Error analysis

After evaluation, review:
- `confusion_matrix.png` for systematic confusions (for example, Shirt vs T-shirt or Pullover vs Coat).
- `misclassified.png` for the most confident errors and what they show about the model's failure modes.
- `learning_curves.png` for overfitting. A widening gap between training and validation curves suggests overfitting; a flat validation curve suggests the model has reached its capacity under the current setup.

## Project structure

```
image_classifier/
├── data/                    # reserved for dataset cache (Keras stores data in ~/.keras by default)
├── notebooks/               # exploratory analysis (optional)
├── src/
│   ├── data_loader.py       # loading, normalization, validation split, dataset summary
│   ├── model.py             # CNN and augmentation definitions
│   ├── train.py             # training with early stopping, checkpoints and logs
│   ├── evaluate.py          # test metrics, experiment table, figures
│   └── visualize.py         # plotting functions
├── outputs/
│   ├── models/              # saved checkpoints (.keras)
│   ├── experiments/         # per-experiment history and summary JSON
│   ├── experiment_table.csv
│   ├── test_results.json
│   ├── learning_curves.png
│   ├── confusion_matrix.png
│   ├── misclassified.png
│   └── dataset_samples.png
├── requirements.txt
└── README.md
```

## Limitations

- **Single seed.** Results come from one training run per experiment. Differences of about a percentage point may be within run-to-run variance. Repeating each experiment with several seeds would give confidence intervals.
- **Grayscale, small images.** Fashion-MNIST is a clean, centered, low-resolution benchmark. Accuracy here does not transfer directly to photographs of clothing. CIFAR-10 (RGB, 32×32) is the planned extension.
- **Augmentation is a single choice.** Only one augmentation set was tested. Stronger or different transformations could change the result.
- **No hyperparameter search.** Filter counts, dropout rate and learning rate were set by hand, not tuned systematically.
- **Robustness is only partly tested.** Augmentation is expected to help generalization, but the project does not yet evaluate performance on deliberately shifted, noisy or occluded test images.
