# 🦴 BUU-LSpine Lumbar Vertebrae Segmentation
### Dense U-Net — Binary Segmentation (No Augmentation)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-orange)
![Task](https://img.shields.io/badge/Task-Medical%20Image%20Segmentation-green)
![Dataset](https://img.shields.io/badge/Dataset-BUU--LSpine-lightgrey)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Pipeline Summary](#-pipeline-summary)
- [Output Graphs & Plots — Explained](#-output-graphs--plots--explained)
- [Metrics Used & Their Significance](#-metrics-used--their-significance)
- [Results Analysis](#-results-analysis)
- [Key Insights & Conclusions](#-key-insights--conclusions)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [Configuration Reference](#-configuration-reference)

---

## 🔍 Project Overview

This project implements a **Dense U-Net** deep learning model for **binary pixel-wise segmentation** of lumbar vertebrae from spinal X-ray images. The model takes an X-ray image as input and produces a segmentation mask distinguishing between:

| Class ID | Class Name | Pixel Colour |
|----------|-----------|--------------|
| `0` | Background | Black `(0, 0, 0)` |
| `1` | Vertebra (L1–L5 merged) | White `(255, 255, 255)` |

> **Why binary?** Although the dataset contains individual vertebra labels (L1 through L5), this baseline notebook merges all five vertebra classes into a single "Vertebra" foreground class. This simplifies the problem and establishes a strong baseline before attempting multi-class segmentation.

This run uses **no data augmentation** — training and validation use the same resize + normalisation transforms — making it a clean, controlled baseline.

---

## 📁 Dataset

**Dataset:** BUU-LSpine (Burapha University Lumbar Spine Dataset)

- **Format:** X-ray images (`.jpg`/`.png`) paired with per-image COCO-format JSON annotation files containing polygon segmentations of each vertebra.
- **Annotation style:** Each JSON file covers exactly **one** image. `image_id` is not unique across files — the **filename stem** is the sole unique identifier used throughout the pipeline.
- **Masks:** Built on-the-fly from polygon coordinates using OpenCV's `fillPoly`; no pre-saved mask files are needed.

### Data Split

| Split | Samples |
|-------|---------|
| Train | 500 |
| Validation | 100 |
| Test | Remaining (all samples after train + val) |

> **Note:** If the total dataset is smaller than 600 samples, the pipeline automatically falls back to a proportional 70/15/15 split.

---

## 🧠 Model Architecture

The model is a **Dense U-Net** — a U-Net variant where standard convolutional blocks are replaced by **DenseNet-style dense blocks**, enabling dense feature reuse across layers.

```
Input (3 × H × W)
    │
    ▼
 [Stem Conv 7×7]
    │
    ▼
 Encoder ×4                    ← DenseBlock + SEBlock (skip) + TransitionDown
    │
    ▼
 Bottleneck DenseBlock
    │
    ▼
 Decoder ×4                    ← TransitionUp + Concat(skip) + DenseBlock
    │         └── Deep Supervision heads (stages 0 & 1, training only)
    ▼
 Segmentation Head (1×1 Conv)
    │
    ▼
Output logits (NUM_CLASSES × H × W)
```

### Key Architectural Features

**DenseBlock:** Each layer receives feature maps from *all previous layers* in the block. This promotes feature reuse and gradient flow.

**SEBlock (Squeeze-and-Excitation):** Applied to skip connections. Learns channel-wise attention weights — "which feature channels matter most for this skip?"

**Deep Supervision:** Auxiliary segmentation heads at decoder stages 0 and 1 provide gradient signal at intermediate layers during training (weighted at 0.3×), helping earlier layers learn faster.

**Gradient Checkpointing:** Recomputes activations during the backward pass instead of storing them — reduces GPU memory usage by ~40% at a ~20% speed cost. Essential for running on low-VRAM GPUs.

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Growth Rate | 16 |
| Initial Features | 32 |
| Dense Layers per Block | (3, 3, 3, 3, 4) |
| Input Channels | 3 (grayscale replicated ×3) |
| Output Classes | 2 |
| Image Size | 384×384 (< 8 GB VRAM) or 512×512 (≥ 8 GB VRAM) |

---

## ⚙️ Pipeline Summary

```
Raw X-rays + COCO JSON annotations
         │
         ▼
1. Parse JSONs → build registry (filename stem as unique key)
         │
         ▼
2. Build binary masks on-the-fly (all vertebrae → class 1)
         │
         ▼
3. Split: Train (500) / Val (100) / Test (remainder)
         │
         ▼
4. Transforms: Resize → Normalize (no augmentation)
         │
         ▼
5. Dense U-Net training (80 epochs, early stopping)
   Loss = 0.5 × CrossEntropy + 0.5 × Dice
   Optimizer: AdamW | Scheduler: CosineAnnealingWarmRestarts
         │
         ▼
6. Evaluate on test set → metrics + visualisations
```

---

## 📊 Output Graphs & Plots — Explained

All plots are saved under `output_no_aug/plots/` and `output_no_aug/predictions/`.

---

### 1. `sample_<file_id>.png` — Raw Sample Visualisation

**What it shows:** A 3-panel figure for each of the first 2 samples in the dataset (before training).

| Panel | Content |
|-------|---------|
| Left | Original X-ray image |
| Middle | Ground-truth mask (black = background, white = vertebra) |
| Right | Overlay of X-ray with mask (60% image + 40% mask colour) |

**How to interpret it:**
- The overlay confirms that polygon annotations correctly align with vertebra regions in the X-ray.
- A clean white region over the vertebrae in the middle panel indicates well-formed masks.
- If the overlay looks misaligned, it signals a coordinate mismatch between the image and annotations.

**What to look for:** The vertebrae should appear as a continuous white column in the centre of the X-ray in the mask panel.

---

### 2. `training_curves.png` — Training History Dashboard

**What it shows:** A 2×3 grid of training metric curves across all epochs.

#### Row 1 — Loss Curves

| Subplot | X-axis | Y-axis | What it measures |
|---------|--------|--------|-----------------|
| Combined Loss | Epoch | Loss value | Weighted sum of CE + Dice loss for train & val |
| Cross-Entropy Loss | Epoch | CE Loss | Classification penalty per pixel |
| Dice Loss | Epoch | Dice Loss | Overlap penalty between prediction and ground truth |

**How to interpret:**
- **Good sign ✅:** Both train and val curves trend downward and converge close together.
- **Overfitting ⚠️:** Train loss keeps dropping while val loss plateaus or rises — the model is memorising training data.
- **Underfitting ⚠️:** Both curves stay high and flat — the model is not learning.
- The gap between train and val curves indicates the degree of overfitting; a narrow gap is ideal.

#### Row 2 — Metric Curves

| Subplot | What it measures |
|---------|-----------------|
| Validation mIoU | Mean Intersection over Union on val set per epoch |
| Validation Mean Dice | Mean Dice coefficient on val set per epoch |
| Learning Rate Schedule | LR value per epoch (log scale) |

**How to interpret mIoU & Dice curves:**
- Both should trend **upward** over epochs.
- Sudden drops followed by recovery are normal with CosineAnnealingWarmRestarts — the LR resets cause brief performance dips before the model finds better minima.

**Learning Rate Schedule:**
- The LR follows a cosine annealing pattern with warm restarts (`T_0=20`, `T_mult=2`): it decays from the initial LR to near-zero over 20 epochs, then restarts — and the cycle length doubles each time (20 → 40 → 80 epochs).
- These restarts help escape local minima and are visible as periodic spikes in the LR plot.

---

### 3. `live_dice_miou_curve.png` — Live Val Dice & mIoU Curve

**What it shows:** A single plot updated after every epoch showing Val Mean Dice and Val mIoU together, with dashed horizontal lines marking the best values achieved so far.

**How to interpret:**
- The two curves should track each other closely (Dice and IoU are mathematically related: `Dice = 2·IoU / (1 + IoU)`).
- Best-value reference lines help identify the epoch of peak performance at a glance.
- Saved live during training so you can monitor progress without waiting for the full dashboard.

---

### 4. `epoch_NNN/<file_id>_miouXX.X_diceXX.X.png` — Epoch Snapshots

**What it shows:** After every training epoch, 4 random validation images are saved as 3-panel figures:

| Panel | Content |
|-------|---------|
| Left | De-normalised input X-ray (grayscale) |
| Middle | Predicted segmentation mask (model output) |
| Right | Ground-truth mask rendered from the COCO JSON |

The predicted mask panel also shows a small text overlay with the per-class Dice score.

**How to interpret:**
- **Early epochs:** The prediction will likely be noisy or largely black (predicting background everywhere) — this is normal.
- **Mid training:** The model begins to identify the rough location of the vertebral column.
- **Late epochs:** The prediction should closely match the ground-truth mask, with clean boundaries around the vertebrae.
- The filename encodes `mIoU` and `Dice` for that sample — higher = better match.

**What good looks like:** The predicted mask panel should show a solid white column in the spine region, nearly identical to the ground-truth panel.

---

### 5. `test_per_class_metrics.png` — Per-Class IoU and Dice Bar Charts

**What it shows:** Two side-by-side bar charts showing IoU and Dice coefficient for each class (Background and Vertebra) on the **test set**.

**How to interpret:**
- Background class will almost always score very high (>0.95) because it occupies the majority of pixels.
- The **Vertebra class bar is the one that matters** — this reflects the model's actual ability to delineate bone structure.
- Numbers are annotated above each bar for easy reading.

**Benchmark guidance:**

| Vertebra Dice | Interpretation |
|---------------|----------------|
| < 0.70 | Poor — significant misclassification |
| 0.70 – 0.80 | Acceptable baseline |
| 0.80 – 0.90 | Good — clinically useful |
| > 0.90 | Excellent — state-of-the-art range |

---

### 6. `confusion_matrix.png` — Confusion Matrix (Counts + Normalised)

**What it shows:** Two heatmaps side-by-side for the test set:
- **Left:** Raw pixel count confusion matrix
- **Right:** Row-normalised confusion matrix (each row sums to 1.0)

**How to read the matrix:**

```
                  Predicted
              Background  Vertebra
True  Background [  TN  ] [  FP  ]
      Vertebra   [  FN  ] [  TP  ]
```

| Cell | Meaning |
|------|---------|
| TN (top-left) | Background pixels correctly predicted as background |
| FP (top-right) | Background pixels wrongly predicted as vertebra |
| FN (bottom-left) | Vertebra pixels wrongly predicted as background |
| TP (bottom-right) | Vertebra pixels correctly predicted as vertebra |

**How to interpret:**
- A good model has high values on the **diagonal** (TN and TP) and low off-diagonal values.
- In the normalised matrix, diagonal values close to **1.0** indicate near-perfect per-class accuracy.
- Large FN (bottom-left) = the model is missing vertebrae (under-segmenting).
- Large FP (top-right) = the model is over-predicting vertebrae into background regions.

---

### 7. `qualitative_predictions.png` — Side-by-Side Test Predictions

**What it shows:** An 8-row grid of 4 panels per row for randomly selected test samples:

| Column | Content |
|--------|---------|
| 1 — Input Image | Grayscale X-ray |
| 2 — Ground Truth | True binary mask |
| 3 — Prediction | Model's predicted mask |
| 4 — Overlay | Predicted mask blended over the X-ray |

Each row is labelled with the `file_id` and per-sample `mIoU`.

**How to interpret:**
- Compare columns 2 and 3 directly — they should look nearly identical for a well-trained model.
- The overlay (column 4) provides an intuitive visual of where the model places the vertebra boundary on the actual image.
- Rows with low mIoU values in the label are the "hard cases" — check these for systematic failure patterns (e.g., consistently missing a certain region or failing on a particular image orientation).

---

### 8. `class_distribution.png` — Pixel Class Distribution per Split

**What it shows:** Three bar charts (one per data split) showing the percentage of pixels belonging to Background vs. Vertebra.

**How to interpret:**
- This plot reveals **class imbalance** — in spinal X-rays, background pixels typically dominate (often 80–90% of all pixels).
- Large imbalance is why the loss function uses class-weighted Cross-Entropy and Dice Loss — both help the model not simply predict "everything is background."
- If the distributions are consistent across Train/Val/Test, the split is representative.

---

## 📐 Metrics Used & Their Significance

### Core Segmentation Metrics

**Pixel Accuracy**
The percentage of all pixels (across all images) that are correctly classified.
```
Pixel Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
> ⚠️ **Limitation:** Misleading under class imbalance. A model predicting "all background" on a dataset with 90% background achieves 90% accuracy but is useless. Always read alongside IoU and Dice.

---

**IoU — Intersection over Union (Jaccard Index)**
The ratio of the overlap between prediction and ground truth to their combined area.
```
IoU = TP / (TP + FP + FN)
```
- Range: 0 (no overlap) to 1 (perfect overlap).
- **mIoU** (mean IoU) averages IoU across foreground classes (Background excluded).
- IoU is stricter than Dice — it penalises both false positives and false negatives equally.

---

**Dice Coefficient (F1 Score)**
Measures the harmonic overlap between prediction and ground truth.
```
Dice = 2 × TP / (2 × TP + FP + FN)
```
- Range: 0 to 1. Higher is better.
- Mathematically related to IoU: `Dice = 2·IoU / (1 + IoU)`
- More commonly reported in medical image segmentation papers.
- **Mean Dice** averages over foreground classes.

---

**Precision**
Of all pixels the model predicted as vertebra, what fraction actually are vertebra?
```
Precision = TP / (TP + FP)
```
High precision = few false alarms (the model is conservative).

---

**Recall (Sensitivity / True Positive Rate)**
Of all actual vertebra pixels, what fraction did the model correctly detect?
```
Recall = TP / (TP + FN)
```
High recall = few missed vertebrae (the model is thorough).

---

**Specificity (True Negative Rate)**
Of all actual background pixels, what fraction were correctly predicted as background?
```
Specificity = TN / (TN + FP)
```
High specificity = the model is not confusing background for vertebra.

---

**Balanced Accuracy**
The average of Sensitivity and Specificity — robust to class imbalance.
```
Balanced Accuracy = (Recall + Specificity) / 2
```

---

**Matthews Correlation Coefficient (MCC)**
A single number summarising the quality of binary classification, accounting for all four cells of the confusion matrix. Considered the most informative single metric for imbalanced datasets.
```
MCC = (TP×TN − FP×FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```
- Range: −1 (completely wrong) to +1 (perfect). 0 = no better than random.

---

### Loss Functions

**Combined Loss = 0.5 × Cross-Entropy + 0.5 × Dice**

| Component | Role |
|-----------|------|
| Cross-Entropy Loss | Per-pixel classification penalty; weighted by inverse class frequency to handle imbalance |
| Dice Loss | Directly optimises the overlap metric; helps when foreground regions are small |
| Combined | Balances the strengths of both — CE provides stable gradients early on; Dice refines boundaries |

**Deep Supervision Auxiliary Loss (0.3× weight):** Extra loss terms from intermediate decoder stages encourage the model's early layers to produce semantically meaningful features, improving gradient flow.

---

## 📈 Results Analysis

### What the Training Curves Tell Us

**Loss Curves:**
- The combined loss, CE loss, and Dice loss should all decrease over epochs.
- If training loss decreases much faster than validation loss, the model is overfitting — expected to some degree without augmentation.
- **No augmentation** means the model sees the same exact images every epoch, making overfitting more likely compared to an augmented run. This baseline is specifically designed to reveal this effect.

**mIoU and Dice Curves:**
- Both metrics should improve over epochs, plateauing as the model converges.
- Periodic dips caused by cosine annealing LR restarts are expected and healthy — they help the model escape flat regions and often lead to higher peaks.
- Early stopping (patience = 20 epochs) prevents wasted computation and reduces overfitting.

### What the Test Metrics Tell Us

**Pixel Accuracy** will naturally be very high (often >90%) due to background dominance — this is **not** a reliable indicator of model quality for this task.

**The metrics that matter most for this project:**

| Metric | Target Range | Why It Matters |
|--------|-------------|----------------|
| Vertebra IoU | > 0.75 | Standard threshold for "good" medical segmentation |
| Vertebra Dice | > 0.85 | Primary metric in spinal segmentation literature |
| Vertebra Recall | > 0.85 | Missing vertebrae (FN) is clinically critical |
| MCC | > 0.80 | Best single metric for imbalanced data |

**Effect of No Augmentation (This Run):**
- Without augmentation, the model may struggle to generalise to X-rays with different contrast, rotation, or patient positioning.
- Test performance will likely be lower than an augmented counterpart.
- This run serves as the **controlled baseline** against which augmented experiments should be compared.

### Class Imbalance Impact
Spinal X-rays typically contain ~80–90% background pixels. The pipeline addresses this through:
1. **Inverse-frequency class weights** in Cross-Entropy loss (computed from 100 random training samples).
2. **Dice Loss** which is naturally robust to imbalance by focusing on overlap rather than raw pixel counts.

---

## 💡 Key Insights & Conclusions

### 1. Architecture Choices Are Memory-Aware
The model uses `growth_rate=16` and `init_features=32` (reduced from 24 and 48 respectively), with gradient checkpointing enabled. This allows training on GPUs with as little as 4–6 GB VRAM. If OOM occurs, the pipeline automatically retries with progressively smaller configurations.

### 2. Filename-First Design Prevents Data Leakage
The BUU-LSpine dataset has non-unique `image_id` values across JSON files (each file restarts from 1). The pipeline correctly uses the **filename stem** as the global unique key, completely avoiding cross-file ID collisions that would cause silent mask misassignment.

### 3. No Augmentation = Lower Generalisation
This notebook is intentionally the "no augmentation" baseline. Real-world X-rays vary in exposure, patient positioning, and image quality. Without augmentation, the model is likely to:
- Perform well on images similar to the training set.
- Underperform on unseen orientations or contrast levels.
- Show a wider train-val gap compared to an augmented run.

### 4. Binary vs. Multi-class Segmentation
Merging L1–L5 into a single "Vertebra" class simplifies the task and typically yields higher Dice scores than per-vertebra segmentation. Results from this binary model should **not** be directly compared to multi-class segmentation benchmarks without noting this distinction.

### 5. CosineAnnealingWarmRestarts Prevents Premature Convergence
The LR schedule restarts periodically, which:
- Allows the model to explore different regions of the loss landscape.
- Reduces sensitivity to the choice of initial learning rate.
- Typically produces better final results than a simple step decay.

### 6. Deep Supervision Speeds Up Early Learning
The auxiliary loss heads at decoder stages 0 and 1 inject gradient signal earlier in the network, which helps the model produce meaningful feature maps in the encoder from earlier epochs rather than relying solely on the gradient flowing back from the final head.

---

## 📁 Project Structure

```
project_root/
│
├── ORIGINAL/                    # Input X-ray images (.jpg / .png)
├── JSON/                        # Per-image COCO annotation files (.json)
│
├── output_no_aug/
│   ├── checkpoints/
│   │   └── best_dense_unet.pth  # Best model checkpoint (saved at best val mIoU)
│   │
│   ├── plots/
│   │   ├── sample_<file_id>.png           # Raw sample visualisations (pre-training)
│   │   ├── training_curves.png            # Full training history dashboard
│   │   ├── live_dice_miou_curve.png       # Live-updated Dice & mIoU curve
│   │   ├── test_per_class_metrics.png     # Per-class IoU & Dice bar charts
│   │   ├── confusion_matrix.png           # Count & normalised confusion matrices
│   │   └── class_distribution.png        # Pixel class distribution per split
│   │
│   ├── predictions/
│   │   └── qualitative_predictions.png    # 8-sample test prediction grid
│   │
│   └── epoch_snapshots/
│       └── epoch_NNN/
│           └── <file_id>_miouXX.X_diceXX.X.png  # Per-epoch snapshot
│
└── BUU_binary_no_augmentation.ipynb       # Main notebook
```

---

## 🚀 How to Run

### Prerequisites

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install albumentations opencv-python-headless Pillow numpy matplotlib scikit-learn tqdm seaborn
```

### Setup

1. Place X-ray images in `./ORIGINAL/`
2. Place COCO JSON annotation files in `./JSON/` (one JSON per image)
3. Open and run the notebook top-to-bottom:

```bash
jupyter notebook BUU_binary_no_augmentation.ipynb
```

### Adjust These Settings (Section 1)

```python
IMAGE_DIR      = Path('./ORIGINAL')   # ← Change to your image folder
JSON_DIR       = Path('./JSON')       # ← Change to your annotation folder
TRAIN_SIZE     = 500                  # ← Number of training samples
VAL_SIZE       = 100                  # ← Number of validation samples
NUM_EPOCHS     = 80                   # ← Max training epochs
BATCH_SIZE     = 2                    # ← Increase if you have more VRAM
LEARNING_RATE  = 1e-3
```

---

## ⚙️ Configuration Reference

| Parameter | Value | Notes |
|-----------|-------|-------|
| `NUM_CLASSES` | 2 | Background + Vertebra |
| `IMAGE_SIZE` | 384×384 or 512×512 | Auto-selected based on VRAM |
| `TRAIN_SIZE` | 500 | Fixed; falls back to 70% if dataset is smaller |
| `VAL_SIZE` | 100 | Fixed; falls back to 15% if dataset is smaller |
| `BATCH_SIZE` | 2 | Increase to 4 or 8 if VRAM allows |
| `NUM_EPOCHS` | 80 | Max; early stopping at patience=20 |
| `LEARNING_RATE` | 1e-3 | Initial LR for AdamW |
| `WEIGHT_DECAY` | 1e-4 | L2 regularisation |
| `RANDOM_SEED` | 42 | Full reproducibility (dataset split, shuffling, etc.) |
| `USE_AMP` | True | Mixed precision (FP16) for faster training |
| `growth_rate` | 16 | Dense block growth rate (reduced for memory) |
| `init_features` | 32 | Stem output channels (reduced for memory) |
| `dense_layers` | (3,3,3,3,4) | Layers per block in encoder + bottleneck |
| `T_0` | 20 | First LR restart cycle length (epochs) |
| `T_mult` | 2 | Cycle length multiplier (20→40→80) |
| `EARLY_STOP_PATIENCE` | 20 | Epochs without improvement before stopping |

---

## 📚 Further Reading

- **Dense U-Net:** Combines [U-Net](https://arxiv.org/abs/1505.04597) with [DenseNet](https://arxiv.org/abs/1608.06993) dense connections for richer feature reuse.
- **Dice Loss for segmentation:** [Milletari et al., 2016 — V-Net](https://arxiv.org/abs/1606.04797)
- **Squeeze-and-Excitation Networks:** [Hu et al., 2018](https://arxiv.org/abs/1709.01507)
- **Deep Supervision:** [Lee et al., 2015](https://arxiv.org/abs/1409.5185)
- **Gradient Checkpointing:** [Chen et al., 2016](https://arxiv.org/abs/1604.06174)

---

*Generated for the BUU-LSpine Dense U-Net Binary Segmentation project — No Augmentation baseline.*
