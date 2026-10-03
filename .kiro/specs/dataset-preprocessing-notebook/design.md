# Design Document: Dataset Preprocessing Notebook

## Overview

`notebooks/01_dataset_preprocessing.ipynb` is a single, self-contained Jupyter notebook that handles all dataset inspection, preprocessing, and feature engineering for a binary structural damage classification project (PEER Φ-Net Task 2). No `src/` modules are created — every function and transformation lives in notebook cells. The notebook prepares two parallel output pipelines: HOG features for a classical SVM and resized/scaled images for InceptionV3 fine-tuning.

Labels follow the PEER README convention: `0` = Damaged, `1` = Undamaged.

---

## Notebook Cell Architecture

The notebook is organized into 13 top-level Markdown sections, executed top-to-bottom.

### 1. Project Overview
A Markdown cell describing the project goal, dataset source (PEER Φ-Net Task 2), label conventions, and what this notebook produces. No code.

### 2. Imports
All library imports in one cell. Libraries: `numpy`, `matplotlib`, `skimage.feature` (HOG), `sklearn.preprocessing.StandardScaler`, `sklearn.model_selection.train_test_split`, `cv2`, `tensorflow`/`tf.keras.applications.inception_v3.preprocess_input`, `os`, `pathlib`.

### 3. Configuration
A single code cell defining all configurable parameters as named variables:

```python
SEED = 42
VAL_FRACTION = 0.15
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_ORIENTATIONS = 9
INCEPTION_SIZE = (299, 299)
BATCH_SIZE = 256
DATA_DIR = pathlib.Path("../data")
RESULTS_DIR = pathlib.Path("../results")
```

`RESULTS_DIR` is created if it does not exist.

### 4. Dataset Inspection
Implements the **Dataset_Inspector**:
- Checks existence of all 5 required `.npy` files; raises `RuntimeError` listing any missing paths.
- Loads `task2_X_train.npy` (Task 2_1) with `mmap_mode='r'`; loads all other arrays normally (they are small).
- Prints shape, dtype, min, max, file size in MB for each array.
- Validates pixel value ranges (`[0,255]` for uint8, `[0.0,1.0]` for float); prints a warning for unexpected ranges.
- Verifies that label array sample counts match image array sample counts.
- Computes and displays class counts, percentages, and imbalance ratio for train and test label arrays.

### 5. Task 2_1 vs Task 2_2 Comparison
Implements the **Dataset_Comparator**:
- Builds a comparison table covering: file size (MB), array shape, spatial dimensions, channels, dtype, pixel value range, and available split files.
- Prints the table as formatted text or a pandas DataFrame display.
- Ends with a "Selected Dataset" Markdown heading and a written justification referencing at least one attribute.
- Warns if spatial dimensions, channel counts, or label sets differ between the two tasks (combining would be inadvisable).

### 6. Class Distribution
Implements the **Visualiser** (class distribution):
- Bar chart with numeric labels showing sample counts per class (Damaged/Undamaged) for train, validation, and test sets.
- Minimum figure size 8×4 inches; displayed inline.
- Printed table of counts and percentages for each split.

### 7. Sample Image Visualization
Implements the **Visualiser** (image grid):
- Selects 3 images per class from the training pool using `np.random.default_rng(SEED)`.
- Reads images via `mmap_mode='r'` indexed access — no full array copy into RAM.
- Displays a 2×3 image grid annotated with class name and array index.
- Falls back gracefully if fewer than 3 samples exist for any class.

### 8. Train / Validation / Test Split
Implements the **Split_Manager**:
- Validates `VAL_FRACTION` is in `(0.05, 0.50)`; halts with an error message if not.
- Splits training pool indices using `sklearn.model_selection.train_test_split(indices, test_size=VAL_FRACTION, stratify=y_train, random_state=SEED)`.
- Stores only index arrays (`train_idx`, `val_idx`); the test set uses the official Task 2_1 test files unchanged.
- Prints sample count, per-class counts, and per-class percentages for all three splits.

### 9. Image Preprocessing
Implements the **Image_Preprocessor**:
- Processes images in batches of `BATCH_SIZE` so only one batch is in RAM at a time.
- Per image: cast to `float32`; normalize uint8 by dividing by `255.0`; verify 3 channels, replicating a single channel if needed; log and skip samples with >3 channels or unsupported dtypes; log any post-normalization pixel values outside `[0.0, 1.0]`.
- Does not apply augmentation.
- The original `data/` files are never modified.

### 10. Feature Engineering for SVM (HOG)
Implements the **HOG_Extractor**:
- Extracts HOG features batch-by-batch using `skimage.feature.hog` with:
  - `pixels_per_cell=HOG_PIXELS_PER_CELL` → `(8,8)`
  - `cells_per_block=HOG_CELLS_PER_BLOCK` → `(2,2)`
  - `orientations=HOG_ORIENTATIONS` → `9`
  - `channel_axis=-1` (multichannel, replaces deprecated `multichannel=True`)
- Prints HOG configuration and feature vector dimensionality.
- Includes a Markdown cell explaining HOG's suitability for crack and structural damage detection.
- Displays a HOG visualization image for one sample.
- Fits `StandardScaler` on training HOG features only; transforms validation and test features.
- Saves scaled arrays to `results/`:
  - `results/X_train_hog.npy`
  - `results/X_val_hog.npy`
  - `results/X_test_hog.npy`
- Also saves the scaler as `results/hog_scaler.pkl` (via `joblib.dump`) for downstream use.

### 11. InceptionV3 Preprocessing
Implements the **InceptionV3_Preparer**:
- Resizes each preprocessed image to `(299, 299)` using `cv2.resize(..., interpolation=cv2.INTER_LINEAR)`.
- Applies `tf.keras.applications.inception_v3.preprocess_input` to scale values to `[-1.0, 1.0]`.
- Processes in batches; does not load all resized images into RAM simultaneously.
- Includes a Markdown cell noting bilinear interpolation and the `[-1, 1]` scaling contract.
- Does not train, fine-tune, or run inference with InceptionV3.
- Does not apply augmentation.

### 12. Data Leakage Checks
Implements the **Leakage_Checker**:
- Asserts `set(train_idx) ∩ set(val_idx) == ∅`.
- Asserts test indices (implicit — the official test file) are not present in `train_idx` or `val_idx`.
- Verifies `StandardScaler` was fitted using only `train_idx` rows.
- Prints `"No data leakage detected"` if all checks pass; raises an error and halts dependent cells if any check fails.

### 13. Final Dataset Summary
Implements the **Summary_Reporter**:
- Prints: selected dataset name and selection justification; split sample counts and per-class breakdowns; image dimensions, channels, dtype, and pixel range; ordered list of preprocessing steps; HOG config and feature vector length; InceptionV3 input size, resize method, and pixel range.
- Ends with the message: `"Preprocessing complete. Data is ready for: (1) classical-ml branch (SVM + HOG features), (2) deep-learning branch (InceptionV3 fine-tuning)."`

---

## Key Technical Decisions

### Memory Safety
`task2_X_train.npy` from Task 2_1 is ~7 GB. Loading it fully into RAM would crash most development machines. Two strategies are applied:

- **mmap_mode='r'** — used during inspection and sample display. NumPy reads array metadata and only fetches indexed rows on demand; the full buffer never enters RAM.
- **Batch processing** — HOG extraction, image preprocessing, and InceptionV3 resizing all iterate over slices of size `BATCH_SIZE=256`. Only one batch is materialized in RAM at a time before writing results incrementally.

### HOG Parameters
| Parameter | Value | Rationale |
|---|---|---|
| `pixels_per_cell` | `(8, 8)` | Captures local gradient patterns at a scale appropriate for crack detection in ~128–256px images |
| `cells_per_block` | `(2, 2)` | Normalizes contrast locally; robust to lighting variation across structural images |
| `orientations` | `9` | Standard setting; balances gradient directional resolution with feature vector length |
| `channel_axis` | `-1` | Processes all 3 colour channels; colour cues (staining, concrete vs. rebar) contribute to damage detection |

### Stratified Split
`sklearn.model_selection.train_test_split(indices, test_size=0.15, stratify=y_train, random_state=42)` ensures the class ratio (Damaged/Undamaged) in the validation set mirrors the training pool. This matters because the dataset has an imbalance (detectable from the class distribution cell). The official test set is held out entirely — never touched during split creation.

### InceptionV3 Preprocessing
Images are resized to `(299, 299)` with `cv2.resize(..., interpolation=cv2.INTER_LINEAR)` (bilinear). `tf.keras.applications.inception_v3.preprocess_input` then linearly maps `[0.0, 1.0]` float32 values to `[-1.0, 1.0]`, which matches the scale the InceptionV3 weights expect.

### Feature Persistence
HOG arrays are saved as `.npy` files in `results/` after extraction. This decouples the slow extraction step (~minutes) from downstream SVM training notebooks. The `StandardScaler` is also persisted so the same transform can be applied to new inference data without re-fitting.

---

## Data Flow

```
data/task2_damage_state_1/
  task2_X_train.npy  (mmap)──┐
  task2_y_train.npy ─────────┼──► Inspect ──► Compare ──► Split indices
  task2_X_test.npy  ─────────┤              (Task 2_1     (train_idx,
  task2_y_test.npy  ─────────┘               vs 2_2)      val_idx)
data/task2_damage_state_2/                                    │
  task2_X_train.npy ─────────────────────────────────────────┘
                                                              │
                              ┌───────────────────────────────┘
                              ▼
                      Batch Preprocessing
                      (float32, normalize, channel check)
                              │
              ┌───────────────┴──────────────────┐
              ▼                                  ▼
      HOG Extraction                    InceptionV3 Resize
      (skimage.hog, batch)              (cv2 bilinear → 299×299)
              │                                  │
      StandardScaler                  preprocess_input → [-1, 1]
      (fit on train only)
              │
      Save to results/
        X_train_hog.npy
        X_val_hog.npy
        X_test_hog.npy
        hog_scaler.pkl
              │
      Leakage Checks ──► Summary
```

---

## Libraries

| Library | Usage |
|---|---|
| `numpy` | Array loading (`mmap_mode`), index operations, `.npy` save/load |
| `matplotlib` | Inline plots, image grids, bar charts |
| `scikit-image` (`skimage.feature.hog`) | HOG feature extraction |
| `scikit-learn` | `StandardScaler`, `train_test_split` |
| `opencv-python` (`cv2`) | Bilinear resize to `(299, 299)` |
| `tensorflow` / `tf.keras` | `inception_v3.preprocess_input` for `[-1, 1]` scaling |
| `joblib` | Persist `StandardScaler` to disk |
| `os`, `pathlib` | Path construction and directory creation |

---

## Configuration Variables

All defined in the cell immediately after Imports (Section 3):

| Variable | Value | Purpose |
|---|---|---|
| `SEED` | `42` | Fixed random seed for reproducibility |
| `VAL_FRACTION` | `0.15` | Fraction of training pool used for validation |
| `HOG_PIXELS_PER_CELL` | `(8, 8)` | HOG cell size in pixels |
| `HOG_CELLS_PER_BLOCK` | `(2, 2)` | HOG block normalization size |
| `HOG_ORIENTATIONS` | `9` | Number of gradient orientation bins |
| `INCEPTION_SIZE` | `(299, 299)` | Target spatial size for InceptionV3 |
| `BATCH_SIZE` | `256` | Images processed per batch (memory control) |
| `DATA_DIR` | `pathlib.Path("../data")` | Root path to raw dataset files |
| `RESULTS_DIR` | `pathlib.Path("../results")` | Output directory for saved feature arrays |
