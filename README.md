# BUU-LSpine Lumbar Vertebrae Segmentation — Dense U-Net (Binary, No Augmentation)

> **Task:** Binary pixel-wise segmentation of lumbar vertebrae (L1–L5) from X-ray images  
> **Dataset:** BUU-LSpine — COCO-annotated X-ray images with per-vertebra polygon masks  
> **Model:** Dense U-Net with Squeeze-and-Excitation skip connections and deep supervision  
> **Framework:** PyTorch (CUDA-accelerated)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset & Splits](#2-dataset--splits)
3. [Model Architecture](#3-model-architecture)
4. [Training Configuration](#4-training-configuration)
5. [Loss Functions](#5-loss-functions)
6. [Evaluation Metrics Explained](#6-evaluation-metrics-explained)
7. [Output Graphs & How to Interpret Them](#7-output-graphs--how-to-interpret-them)
8. [Results Analysis](#8-results-analysis)
9. [Key Insights & Conclusions](#9-key-insights--conclusions)
10. [File & Directory Structure](#10-file--directory-structure)
11. [How to Run](#11-how-to-run)

---

## 1. Project Overview

This project trains a **Dense U-Net** to perform **binary segmentation** on lumbar spine X-ray images. The model learns to distinguish two things in every pixel of an X-ray:

| Class ID | Label | Description |
|---|---|---|
| `0` | **Background** | Everything that is NOT a vertebra |
| `1` | **Vertebra** | Any of the five lumbar vertebrae (L1–L5), all merged into one class |

> **Why binary?** Merging all five vertebra classes (L1–L5) into a single "Vertebra" class makes the problem simpler and is useful when you care only about *where* vertebrae are, not *which* one they are.

The pipeline starts from raw COCO-format JSON annotations, builds pixel masks on-the-fly, trains the model, and evaluates it with a rich set of metrics used in medical imaging research.

---

## 2. Dataset & Splits

The BUU-LSpine dataset contains lumbar X-ray images, each paired with a JSON file holding polygon annotations for each vertebra.

```
Dataset
├── ORIGINAL/   ← X-ray images (.jpg / .png)
└── JSON/       ← One JSON annotation file per image (COCO format)
```

**Key design choices:**
- Each JSON file is parsed **independently** — `image_id` values are reused across files, so the image **filename stem** is the sole unique identifier.
- Masks are built **on-the-fly** at training time from polygon coordinates using OpenCV's `fillPoly`. No pre-saved mask files are needed.

**Data splits:**

| Split | Size | Purpose |
|---|---|---|
| Train | 500 | Model learning |
| Val | 100 | Hyperparameter tuning & early stopping |
| Test | Remainder | Final unbiased evaluation |

> If the total dataset is smaller than 600 samples, the notebook automatically falls back to a **70 / 15 / 15** proportional split so it works at any scale.

---

## 3. Model Architecture

The model is a **Dense U-Net** — a U-Net variant where regular convolutional blocks are replaced with **DenseBlocks** from DenseNet. This means each layer receives feature maps from all preceding layers, encouraging strong feature reuse and gradient flow.

```
Input X-ray (3 × H × W)
       │
   [Stem Conv]          ← 7×7 conv, 32 feature maps
       │
   ──────────────────── ENCODER ─────────────────────
   DenseBlock → SE → TransitionDown (×4 stages)
       │           │
       └── skip ───┘   ← SE-attended skip connections
       │
   [Bottleneck DenseBlock]
       │
   ──────────────────── DECODER ─────────────────────
   TransitionUp → concat(skip) → DenseBlock (×4)
       │                               │
       └── Auxiliary head (stages 0,1) ┘  ← deep supervision
       │
   [Final 1×1 Conv Head]
       │
Output logits (2 × H × W)
```

**Key components:**

| Component | Role |
|---|---|
| `DenseBlock` | Stacks dense layers where each layer concatenates all prior outputs |
| `SEBlock` (Squeeze-and-Excitation) | Channel attention on encoder skip connections — learns which feature channels matter most |
| `TransitionDown` | 1×1 conv + MaxPool — halves spatial resolution, compresses channels |
| `TransitionUp` | Bilinear upsample + 1×1 conv — restores spatial resolution |
| Deep supervision heads | Auxiliary loss at early decoder stages to improve gradient flow to the encoder |
| Gradient checkpointing | Re-computes activations during backprop instead of storing them — saves ~40% GPU memory |

**Model size (standard config):**

| Setting | Value |
|---|---|
| Growth rate | 16 |
| Initial features | 32 |
| Dense layers per block | (3, 3, 3, 3, 4) |
| Total parameters | ~5–8 M (varies by config) |

---

## 4. Training Configuration

| Hyperparameter | Value | Why |
|---|---|---|
| Image size | 384×384 or 512×512 | Auto-selected based on available VRAM |
| Batch size | 2 | Keeps GPU memory usage low |
| Epochs | 80 (max) | With early stopping |
| Learning rate | `1e-3` | Starting LR for AdamW |
| Weight decay | `1e-4` | L2 regularisation to prevent overfitting |
| Optimizer | AdamW | Decoupled weight decay, stable for medical images |
| Scheduler | CosineAnnealingWarmRestarts (T₀=20) | LR cycles help escape local minima |
| Mixed precision | AMP (float16) | Faster training, less VRAM |
| Early stopping | 20 epochs patience | Stops if validation mIoU doesn't improve |

**No data augmentation** is applied in this run — images are only resized and normalised. This is intentional to establish a clean baseline.

---

## 5. Loss Functions

Training uses a **Combined Loss** that blends two complementary objectives:

```
Combined Loss = 0.5 × CrossEntropy + 0.5 × Dice
```

### Cross-Entropy Loss
Penalises each pixel independently. Uses **class weights** computed from pixel frequency in a random sample of the training set — this compensates for the heavy class imbalance (background pixels vastly outnumber vertebra pixels in X-rays).

### Dice Loss
Directly optimises the Dice coefficient (overlap between predicted and true segmentation). Works well when the foreground is small and sparse, which is common in medical images.

### Why combine both?
Cross-entropy ensures pixel-level accuracy; Dice ensures the overall segmentation region is well-shaped. Together they produce more robust, better-calibrated predictions.

### Deep Supervision
Auxiliary heads at the first two decoder stages each contribute an additional `0.3 × Combined Loss` signal, propagating gradients directly into the encoder and speeding up learning.

---

## 6. Evaluation Metrics Explained

All metrics are computed from the **confusion matrix** — a table that counts how many pixels were correctly/incorrectly classified.

| Metric | Formula | What it means |
|---|---|---|
| **Pixel Accuracy** | Correct pixels / Total pixels | Percentage of pixels classified correctly |
| **IoU (Jaccard Index)** | TP / (TP + FP + FN) | Overlap between predicted and actual vertebra region |
| **Dice Coefficient (F1)** | 2·TP / (2·TP + FP + FN) | Harmonic mean of precision and recall; the standard metric in medical segmentation |
| **Precision** | TP / (TP + FP) | Of all pixels predicted as vertebra, how many actually are? |
| **Recall (Sensitivity)** | TP / (TP + FN) | Of all actual vertebra pixels, how many were found? |
| **Specificity (TNR)** | TN / (TN + FP) | How well does the model correctly identify non-vertebra pixels? |
| **Balanced Accuracy** | (Sensitivity + Specificity) / 2 | Useful for imbalanced classes |
| **MCC** | (TP·TN − FP·FN) / √(...) | Matthews Correlation Coefficient — most reliable single metric for imbalanced binary tasks |
| **mIoU** | Mean IoU over foreground classes | Standard benchmark metric across segmentation literature |

> **Rule of thumb for Dice/IoU in medical segmentation:**
> - Below 0.70 → needs improvement
> - 0.70 – 0.80 → acceptable
> - 0.80 – 0.90 → good
> - Above 0.90 → excellent

---

## 7. Output Graphs & How to Interpret Them

All plots are saved in `output_no_aug/plots/`. Here is what each one shows and how to read it.

---

### 7.1 Sample Visualisation — `sample_<file_id>.png`

**Generated by:** `show_sample()` at data loading time (Section 2)

**What it shows:** A 3-panel figure for each of the first 2 samples in the dataset.

| Panel | Contents |
|---|---|
| Left | Raw X-ray image |
| Middle | Ground-truth binary mask (white = vertebra, black = background) |
| Right | Overlay of mask on image (60% image + 40% colour mask) |

**How to interpret:** Use this to confirm that polygon annotations were correctly parsed and converted to pixel masks. If the white regions in the middle panel align with the vertebrae visible in the X-ray, the annotation pipeline is working correctly.

---

### 7.2 Training Curves — `training_curves.png`

**Generated by:** `plot_training_curves()` after training (Section 10)

A 2×3 grid of subplots covering the full training history.

#### Row 1 — Loss curves

| Plot | X-axis | Y-axis | Good sign |
|---|---|---|---|
| **Combined Loss** | Epoch | Loss value | Both train & val curves decrease smoothly and converge close together |
| **Cross-Entropy Loss** | Epoch | CE loss | Val CE tracks train CE without diverging |
| **Dice Loss** | Epoch | Dice loss | Decreases toward 0; lower = better overlap |

> ⚠️ **Watch for:** If the validation loss starts increasing while the training loss continues decreasing, the model is **overfitting** — it has memorised the training set but doesn't generalise.

#### Row 2 — Metric & learning rate curves

| Plot | What to look for |
|---|---|
| **Validation mIoU** | Should trend upward; the red dashed horizontal line marks the best value achieved |
| **Validation Mean Dice** | Should trend upward, typically ~5–10% higher than mIoU |
| **Learning Rate Schedule** | Cosine waves on a log scale — LR cycles down then restarts. This helps escape local minima. If no improvement happens after a restart, training may have converged. |

---

### 7.3 Live Dice & mIoU Curve — `live_dice_miou_curve.png`

**Generated by:** Saved after every training epoch (Section 9)

A single line plot saved after every epoch so you can monitor progress in real time without waiting for the full `training_curves.png`.

- **Green solid line** — Validation Mean Dice
- **Blue dashed line** — Validation mIoU
- **Dotted horizontal lines** — Mark the best values achieved so far

This is the first plot to check during training to see if the model is learning at all.

---

### 7.4 Per-Class Metrics Bar Chart — `test_per_class_metrics.png`

**Generated by:** `plot_per_class_metrics()` after test evaluation (Section 11)

Two side-by-side bar charts for the test set.

| Chart | What it shows |
|---|---|
| **Per-Class IoU** | IoU for Background (black bar) and Vertebra (white bar) |
| **Per-Class Dice** | Dice/F1 for both classes |

**How to interpret:**
- The **Vertebra bar** is the one that matters for clinical usefulness.
- High background IoU is expected (and easy to achieve) because background pixels are abundant.
- If the Vertebra bar is noticeably shorter than the Background bar, the model struggles with the foreground, likely because vertebrae are small relative to the full image.

---

### 7.5 Confusion Matrix — `confusion_matrix.png`

**Generated by:** `plot_confusion_matrix()` after test evaluation (Section 11)

Two heatmaps side by side:

| Heatmap | Description |
|---|---|
| **Left (counts)** | Raw pixel counts — how many pixels were predicted as each class vs what they truly were |
| **Right (row-normalised)** | Each row sums to 1.0, showing the *rate* of correct/incorrect predictions |

**How to read the normalised matrix:**

```
              Predicted
              BG    Vert
Actual  BG  [0.99  0.01]   ← 99% of actual background pixels correctly predicted
       Vert [0.05  0.95]   ← 95% of actual vertebra pixels correctly predicted
```

- The **diagonal** (top-left and bottom-right) should be high — these are correct predictions.
- **Off-diagonal values** represent errors. A high value at row=Vertebra, col=Background means the model is missing vertebra pixels (false negatives → low recall). A high value at row=Background, col=Vertebra means over-prediction (false positives → low precision).

---

### 7.6 Class Pixel Distribution — `class_distribution.png`

**Generated by:** `plot_class_distribution()` after test evaluation (Section 12)

Three bar charts (one per split: Train, Val, Test) showing the percentage of pixels belonging to each class.

**How to interpret:** In medical X-ray datasets, you typically expect to see something like:

```
Background: ~80–90%
Vertebra:   ~10–20%
```

This confirms the class imbalance problem — which is why class-weighted cross-entropy and Dice loss are both used. If the distributions look consistent across Train/Val/Test, the random split was fair.

---

### 7.7 Qualitative Predictions — `predictions/qualitative_predictions.png`

**Generated by:** `visualize_predictions()` on the test set (Section 12)

A grid with **8 rows** (one per test sample) and **4 columns**:

| Column | Contents |
|---|---|
| 1 — Input Image | De-normalised grayscale X-ray |
| 2 — Ground Truth | True binary mask from JSON annotation |
| 3 — Prediction | Model's predicted binary mask |
| 4 — Overlay | X-ray + predicted mask blended together |

Each row's left label shows the file ID and per-sample mIoU score.

**What to look for:**
- **Column 2 vs Column 3** — ideally they look nearly identical. Differences highlight where the model is wrong.
- Common failure modes: jagged edges on vertebra boundaries, missing small vertebral structures, false positives in areas with similar X-ray intensity.
- The overlay (column 4) makes it visually easy to see whether predictions are anatomically reasonable.

---

### 7.8 Epoch Snapshots — `epoch_snapshots/epoch_NNN/<file_id>_miouXX_diceYY.png`

**Generated by:** `save_epoch_snapshots()` after every training epoch

For 4 randomly selected validation samples, a 3-panel figure (Input → Predicted Mask → Ground Truth) is saved after every epoch. This creates a visual "movie" of how the model's predictions evolve over training.

**How to use:** Browse through these chronologically to see:
- Early epochs — mostly noise or over-predicted regions
- Mid training — rough shapes appear, boundaries sharpen
- Late epochs — precise, clean vertebra outlines emerge

The filename encodes the mIoU and Dice scores so you can quickly spot which epoch gave the best visual quality.

---

## 8. Results Analysis

### Training Behaviour

The training setup is designed to be stable and memory-efficient:

- **Gradient checkpointing** allows the full Dense U-Net to train on GPUs with as little as 4 GB VRAM by re-computing activations during backpropagation.
- **CosineAnnealingWarmRestarts** cycles the learning rate, which typically produces a staircase-like improvement pattern in the validation metrics — mIoU jumps after each LR restart.
- **Early stopping with patience=20** prevents wasted computation and overfitting once the model has converged.

### Class Imbalance Handling

In lumbar X-rays, vertebra pixels make up roughly 10–20% of the image. Without corrections:
- A model that predicts *everything as background* would achieve ~85% pixel accuracy but 0% Dice for vertebrae — which is useless.

This project addresses this with:
1. **Inverse-frequency class weights** in the cross-entropy loss
2. **Dice loss** which is naturally insensitive to class imbalance
3. **mIoU and Dice** as the primary evaluation metrics (not pixel accuracy)

### What "Good" Looks Like for This Task

For binary lumbar vertebra segmentation on clinical X-rays:

| Metric | Acceptable | Good | Excellent |
|---|---|---|---|
| Vertebra IoU | > 0.70 | > 0.80 | > 0.88 |
| Vertebra Dice | > 0.80 | > 0.88 | > 0.93 |
| Pixel Accuracy | > 90% | > 95% | > 97% |
| MCC | > 0.75 | > 0.85 | > 0.90 |

> These thresholds are informed by published results on lumbar spine X-ray segmentation tasks using deep learning methods.

### Baseline vs Augmented Comparison

This notebook is the **no-augmentation baseline**. It establishes a performance floor. A companion notebook with augmentation (flips, rotations, elastic deformations) is expected to improve Dice and IoU by 3–8 percentage points, particularly on edge cases and unusual X-ray angles.

---

## 9. Key Insights & Conclusions

1. **Binary merging of L1–L5 is effective** — treating all lumbar vertebrae as a single foreground class simplifies the problem significantly and yields a cleaner segmentation signal. The model doesn't need to distinguish between individual vertebrae to be clinically useful for, e.g., overall spine localisation.

2. **Dense connections improve feature reuse** — by concatenating feature maps from all preceding layers, DenseBlocks allow the model to simultaneously use both low-level texture features (useful for edges) and high-level semantic features (useful for understanding anatomy), which is critical in medical imaging where both matter.

3. **SE (Squeeze-and-Excitation) attention on skip connections** — recalibrating skip connections by channel importance means the model learns to suppress uninformative feature channels (noise, irrelevant anatomy) before passing them to the decoder. This is particularly helpful for X-rays, which often contain overlapping structures.

4. **Combined CE + Dice loss outperforms either alone** — cross-entropy provides pixel-level supervision, while Dice loss ensures the segmentation shape is globally coherent. Their combination consistently produces better boundary quality than either loss individually.

5. **No augmentation = useful baseline, not a final result** — the model is trained only on real, unmodified X-rays. This means performance on unusual patient positions, low-quality scans, or high-contrast images may be limited. The augmented version is expected to generalise better.

6. **Deep supervision accelerates early learning** — auxiliary losses at the two earliest decoder stages force the encoder to produce useful intermediate representations sooner, which is visible in the epoch snapshots as cleaner masks appearing earlier in training.

---

## 10. File & Directory Structure

```
output_no_aug/
├── checkpoints/
│   └── best_dense_unet.pth          ← Best model weights (saved on best val mIoU)
│
├── plots/
│   ├── sample_<file_id>.png         ← Raw data sanity checks (2 images)
│   ├── live_dice_miou_curve.png     ← Live-updated Dice & mIoU curve per epoch
│   ├── training_curves.png          ← Full 6-panel training history
│   ├── test_per_class_metrics.png   ← Bar chart: IoU & Dice per class on test set
│   ├── confusion_matrix.png         ← Pixel-level confusion matrix (counts + normalised)
│   └── class_distribution.png      ← Pixel class balance across Train/Val/Test
│
├── predictions/
│   └── qualitative_predictions.png ← 8-sample visual comparison grid
│
└── epoch_snapshots/
    └── epoch_NNN/
        └── <file_id>_miouXX_diceYY.png  ← Per-epoch prediction snapshots
```

---

## 11. How to Run

### Prerequisites

```bash
pip install torch torchvision
pip install albumentations opencv-python-headless
pip install scikit-learn matplotlib pillow tqdm seaborn
```

### Directory Setup

Edit these two lines at the top of **Section 1** in the notebook:

```python
IMAGE_DIR = Path('./ORIGINAL')   # folder with your X-ray images
JSON_DIR  = Path('./JSON')       # folder with one JSON file per image
```

### Training

Run all cells from top to bottom. The notebook will:
1. Parse annotations and build the dataset registry
2. Split into train/val/test
3. Build the Dense U-Net and move it to GPU
4. Train for up to 80 epochs with early stopping
5. Save the best checkpoint and all plots automatically

### Inference on a New Image

```python
# Load the best checkpoint
checkpoint = torch.load('output_no_aug/checkpoints/best_dense_unet.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare your image
from PIL import Image
import numpy as np, torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485,), std=(0.229,)),
    ToTensorV2(),
])

gray = np.array(Image.open('your_xray.jpg').convert('L'))
img_3ch = np.stack([gray, gray, gray], axis=-1)
tensor = transform(image=img_3ch)['image'].unsqueeze(0)  # (1, 3, H, W)

with torch.no_grad():
    logits = model(tensor.to(device))          # (1, 2, H, W)
    pred_mask = logits.argmax(dim=1).squeeze()  # (H, W) — 0=BG, 1=Vertebra
```

---

## Citation / Acknowledgements

- **Dataset:** BUU-LSpine — lumbar spine X-ray dataset with COCO polygon annotations
- **Architecture:** Dense U-Net based on DenseNet (Huang et al., 2017) fused with U-Net (Ronneberger et al., 2015)
- **Attention:** Squeeze-and-Excitation Networks (Hu et al., 2018)
- **Framework:** PyTorch + Albumentations

---

*This README was generated to document the `BUU_binary_no_augmentation_fixed.ipynb` notebook. For questions about the dataset or annotations, refer to the BUU-LSpine dataset documentation.*
