# Implementation Tasks

## Tasks

- [x] 1. Bootstrap notebook and configuration
  - Create `notebooks/01_dataset_preprocessing.ipynb` with all 13 Markdown section headers
  - Add Section 2 (Imports) cell: numpy, matplotlib, skimage, sklearn, cv2, tensorflow, joblib, os, pathlib
  - Add Section 3 (Configuration) cell: SEED=42, VAL_FRACTION=0.15, HOG_PIXELS_PER_CELL=(8,8), HOG_CELLS_PER_BLOCK=(2,2), HOG_ORIENTATIONS=9, INCEPTION_SIZE=(299,299), BATCH_SIZE=256, DATA_DIR, RESULTS_DIR; create RESULTS_DIR if not exists
  - **Files**: `notebooks/01_dataset_preprocessing.ipynb`

- [x] 2. Dataset file existence and size check
  - Implement Section 4 cells: check all 5 .npy files exist; raise RuntimeError listing missing paths if any
  - Display each file's size in MB (2 decimal places)
  - **Files**: `notebooks/01_dataset_preprocessing.ipynb`

- [x] 3. Safe array shape and dtype inspection
  - Load task2_X_train.npy (Task 2_1) with mmap_mode='r'; load remaining arrays normally
  - Print shape, dtype, min, max for each array; derive and display H, W, C from shape
  - Warn with array name + detected min/max if pixel range is neither [0,255] nor [0,1]
  - Verify y sample count == X sample count for each split; display error if mismatch
  - **Files**: `notebooks/01_dataset_preprocessing.ipynb`

- [x] 4. Class distribution analysis
  - Compute and display class counts, percentages (2 dp), imbalance ratio for train and test label arrays
  - Handle edge case: skip ratio if only one class present; display warning
  - **Files**: `notebooks/01_dataset_preprocessing.ipynb`

- [x] 5. Task 2_1 vs Task 2_2 comparison
  - Build comparison table: file size MB, shape, spatial dims, channels, dtype, pixel range, available splits (X+y present = available)
  - Display table; add "Selected Dataset" heading with justification referencing ≥1 attribute value
  - Display warning if spatial dims, channels, or label sets differ between tasks
  - **Files**: `notebooks/01_dataset_preprocessing.ipynb`

- [x] 6. Train / validation / test split
  - Validate VAL_FRACTION in (0.05, 0.50); display error and halt if out of range
  - Split training pool indices with train_test_split(stratify=y_train, random_state=SEED, test_size=VAL_FRACTION); store train_idx and val_idx
  - Print sample count, per-class counts, and per-class percentages for train, val, test
  - **Files**: `notebooks/01_dataset_preprocessing.ipynb`

- [x] 7. Sample image visualization
  - Select 3 images per class from training pool using np.random.default_rng(SEED); access via mmap_mode='r' indexed read
  - Display 2×3 grid annotated with class name and array index; handle <3 samples gracefully
  - **Files**: `notebooks/01_dataset_preprocessing.ipynb`

- [x] 8. Class distribution bar chart
  - Bar chart with numeric bar labels for Damaged/Undamaged counts across train, val, test splits; min figure size 8×4 inches
  - **Files**: `notebooks/01_dataset_preprocessing.ipynb`

- [x] 9. Image preprocessing pipeline
  - Implement batch preprocessor (BATCH_SIZE=256): cast to float32; divide by 255 if uint8; replicate channel if grayscale; skip and log samples with >3 channels or unsupported dtype; log samples with post-norm values outside [0,1]
  - Process train_idx, val_idx, and test arrays
  - **Files**: `notebooks/01_dataset_preprocessing.ipynb`

- [x] 10. HOG feature extraction and scaling
  - Extract HOG features batch-by-batch using skimage.feature.hog with configured params and channel_axis=-1
  - Add Markdown justification (≥2 sentences) for HOG suitability for crack detection; print feature vector dimensionality
  - Fit StandardScaler on train HOG only; transform val and test
  - Display HOG visualization image for one sample
  - Save X_train_hog.npy, X_val_hog.npy, X_test_hog.npy and hog_scaler.pkl to results/
  - **Files**: `notebooks/01_dataset_preprocessing.ipynb`, `results/X_train_hog.npy`, `results/X_val_hog.npy`, `results/X_test_hog.npy`, `results/hog_scaler.pkl`

- [x] 11. InceptionV3 data preparation
  - Resize preprocessed images to (299,299) using cv2.INTER_LINEAR in batches
  - Apply tf.keras.applications.inception_v3.preprocess_input to get float32 in [-1,1]
  - Add Markdown cell documenting bilinear resize and [-1,1] scaling contract
  - **Files**: `notebooks/01_dataset_preprocessing.ipynb`

- [x] 12. Data leakage checks
  - Assert set(train_idx) ∩ set(val_idx) == ∅; assert test indices not in train_idx or val_idx
  - Verify StandardScaler fitted on training indices only
  - Print "No data leakage detected" if all pass; raise error and halt otherwise
  - **Files**: `notebooks/01_dataset_preprocessing.ipynb`

- [x] 13. Final preprocessing summary
  - Print: selected dataset + justification; split counts with per-class breakdown; image dims/channels/dtype/pixel range; ordered preprocessing steps; HOG config + feature dim; InceptionV3 size/resize/range
  - End with: "Preprocessing complete. Data is ready for: (1) classical-ml branch (SVM + HOG features), (2) deep-learning branch (InceptionV3 fine-tuning)."
  - **Files**: `notebooks/01_dataset_preprocessing.ipynb`
