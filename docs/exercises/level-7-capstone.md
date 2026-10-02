# Level 7 capstone: an image classifier above 90%

> **Level 7 · Capstone** · ⏱️ 8–12 h · Prerequisites: all of [Level 7](../chapters/07-deep-learning-pytorch/index.md)

Train an image classifier above 90% test accuracy, with augmentation, experiment tracking, and an error analysis. This capstone pulls the whole level together: a clean data pipeline, sanity checks before training, a ResNet-style CNN, a training loop with checkpoints and logged metrics, one honest test evaluation, and an analysis of what the model gets wrong and why. Everything runs on a laptop CPU in minutes. Finish it without notes before moving on to [Level 8](../chapters/08-modern-deep-learning/index.md).

## The scenario

You've joined Threadline, a (fictional) second-hand clothing marketplace. Sellers upload a photo of each item and pick a category by hand, and about one listing in twelve is miscategorized, which hurts search and annoys buyers. The catalog team wants a model that suggests the category from the photo.

The Head of Catalog gives you the brief:

1. "Get it right at least 90% of the time on photos it has never seen, and prove it honestly."
2. "Sellers' photos aren't as tidy as our catalog. Make it robust to small shifts and to items photographed facing the other way."
3. "I want to see training as it happens, and I want to be able to rerun it and get the same model."
4. "Tell me *which* categories it mixes up and why. If two categories are hopelessly confused, we'll show sellers both as suggestions instead of one."
5. "We don't have GPUs. It has to train on a laptop in under ten minutes."

## The data

Use **FashionMNIST**: 70,000 grayscale $28 \times 28$ images in 10 classes, 60,000 for training and 10,000 for testing, with 1,000 test images per class. torchvision downloads it (about 30 MB) the first time:

<!-- skip-run -->
```python
from torchvision import datasets

train = datasets.FashionMNIST("data", train=True, download=True)
test = datasets.FashionMNIST("data", train=False, download=True)
print(train.data.shape, train.data.dtype, train.classes)
```

| Label | Class | Label | Class |
|---|---|---|---|
| 0 | T-shirt/top | 5 | Sandal |
| 1 | Trouser | 6 | Shirt |
| 2 | Pullover | 7 | Sneaker |
| 3 | Dress | 8 | Bag |
| 4 | Coat | 9 | Ankle boot |

The official test split is your test set. Carve your validation set out of the 60,000 training images.

## What to do

Organize the project as four scripts: `data.py`, `model.py`, `train.py`, and `evaluate.py`.

### Part A: the data pipeline (`data.py`)

- Split the training set into training and validation sets (for example 54,000 and 6,000) with a fixed seed.
- Compute the normalization mean and standard deviation from the **training split only**.
- Write a `Dataset` that applies augmentation with `torchvision.transforms.v2` to the training split only. Choose augmentations that preserve the label and match the variation in the brief.
- Build seeded `DataLoader`s with `num_workers=0`.

### Part B: baseline and sanity checks

- Train a linear (softmax regression) baseline for a few epochs and record its validation accuracy. That's the bar your CNN must clear by a wide margin.
- Check that your CNN's loss at initialization is close to $\ln 10 \approx 2.30$.
- Overfit a single batch of 32 images to near-zero loss before training on everything.

### Part C: the model (`model.py`)

- Build a ResNet-style CNN: a stem convolution, residual blocks with batch norm and projection shortcuts where the shape changes, downsampling by strided convolutions, global average pooling, and a linear head.
- Report its parameter count. Keep it small enough that an epoch takes well under a minute on a CPU.

### Part D: training and tracking (`train.py`)

- Seed everything. Use AdamW and a learning-rate schedule (one-cycle or cosine).
- Each epoch, log training loss and accuracy, validation loss and accuracy, the learning rate, and the time, to a CSV file (and to TensorBoard if you have it installed).
- Save a checkpoint (model, optimizer, epoch, config) whenever validation accuracy improves.
- Make the script configurable from the command line (epochs, width, learning rate, augmentation on or off, data directory).

### Part E: one honest evaluation (`evaluate.py`)

- Load the best checkpoint and evaluate on the test set **once**. Report test accuracy and per-class accuracy.
- Plot the training curves from your log.

### Part F: error analysis

- Plot the confusion matrix and list the five most-confused pairs (true → predicted).
- Show a gallery of the misclassified test images the model was most confident about.
- Write a short analysis: which classes are confused, whether the mistakes look like model errors, genuinely ambiguous images, or labeling errors, and what you would try next. Answer the Head of Catalog's question about showing two suggestions.

### Part G (stretch)

- Ablate augmentation: train the same model without it and compare validation curves.
- Try test-time augmentation (average predictions over an image and its mirror) and measure the change.
- Report top-2 accuracy, the metric that matters if the product shows two suggestions.

## Deliverables

1. `data.py`, `model.py`, `train.py`, `evaluate.py`, runnable from a clean folder.
2. The training log (`history.csv`) and the best checkpoint.
3. Figures: training curves, confusion matrix, gallery of mistakes.
4. A short report (a README or notebook): the baseline, the model, the test accuracy, the error analysis, and your recommendation.

## Acceptance checklist

- [ ] Test accuracy is at least 90%, measured once on the official test set with the checkpoint chosen on validation.
- [ ] The validation set comes from the training data, and normalization statistics come from the training split only.
- [ ] Augmentation is applied to training data only, and each augmentation is justified as label-preserving.
- [ ] A baseline is reported, the initial loss is checked against $\ln 10$, and a single batch can be overfit.
- [ ] The model uses residual blocks, and its parameter count is reported.
- [ ] Runs are seeded; rerunning `train.py` with the same arguments gives the same numbers on the same machine.
- [ ] Per-epoch metrics are logged to a file, and the training-curves figure is generated from that file.
- [ ] Checkpoints contain the model and optimizer state, the epoch, and the config.
- [ ] The confusion matrix, the most-confused pairs, and a gallery of misclassified images are produced by `evaluate.py`.
- [ ] The report explains the main confusions and makes a concrete recommendation.
- [ ] Training takes under about ten minutes on a laptop CPU.

## Hints

??? tip "Hint for Part A: which augmentations?"
    Clothing photos vary in position and in which way an item faces, so random crops after a small padding (shifts of 2 to 4 pixels) and horizontal flips are safe. Vertical flips and large rotations aren't: an upside-down shirt never appears at test time. Apply augmentation to images in $[0, 1]$, then normalize.

??? tip "Hint for Part C: a model that fits the budget"
    Three residual stages at $28 \times 28$, $14 \times 14$, and $7 \times 7$ with 16, 32, and 64 channels give under 100,000 parameters and train at a few thousand images per second on a modern laptop CPU. Doubling the width roughly quadruples the cost. Measure the time of one epoch before committing to a plan.

??? tip "Hint for Part D: reaching 90% quickly"
    A one-cycle schedule (`OneCycleLR`) with AdamW and a peak learning rate of a few times $10^{-3}$ reaches good accuracy in few epochs. Use your LR range test from Chapter 2 if you're unsure. Expect training accuracy (on augmented images, in train mode) to sit slightly below validation accuracy; that's normal with augmentation and dropout.

??? tip "Hint for Part F: reading the confusion matrix"
    Normalize each row by the class count so colors show the fraction of each true class, and annotate cells with raw counts. Sort the off-diagonal cells to find the top pairs. For the gallery, sort mistakes by the predicted class's probability: confident mistakes are the most informative, and some will turn out to be labeling errors in the dataset itself.

??? tip "Hint for tracking without TensorBoard"
    A `csv.DictWriter` that writes one row per epoch, flushed after each write, is enough: you can watch it with `tail -f` while training and plot it afterward. If `tensorboard` is installed, log the same scalars with `SummaryWriter`, inside a `try`/`except ImportError` so the script runs either way.

## Solution

A complete worked solution, with all four scripts, real training output, and the error analysis, is in the [Level 7 capstone solution](solutions/level-7-capstone.md). Build your own first: the point is to discover, by doing it, where the time goes and what the errors look like.
