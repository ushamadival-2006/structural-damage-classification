# Requirements Document

## Introduction

This feature is a Jupyter notebook (`notebooks/01_dataset_preprocessing.ipynb`) that performs all dataset analysis, preprocessing, and feature engineering for a binary structural damage classification project using the PEER Φ-Net Task 2 dataset. The notebook prepares two parallel data pipelines: one for a classical SVM classifier and one for an InceptionV3 deep learning model. It does not train any model. The notebook is self-contained — no separate Python modules in `src/` are created.

Labels follow the PEER README convention:
- `0` = Damaged state (D)
- `1` = Undamaged state (UD)

---

## Glossary

- **Notebook**: The Jupyter notebook file `notebooks/01_dataset_preprocessing.ipynb`
- **Dataset_Inspector**: The notebook cells responsible for loading and inspecting dataset metadata
- **Dataset_Comparator**: The notebook cells responsible for comparing Task 2_1 and Task 2_2
- **Split_Manager**: The notebook cells responsible for creating and validating train/validation/test splits
- **Image_Preprocessor**: The notebook cells responsible for dtype conversion, channel normalisation, and pixel normalisation
- **HOG_Extractor**: The notebook cells responsible for computing HOG features from preprocessed images
- **InceptionV3_Preparer**: The notebook cells responsible for resizing and scaling images for InceptionV3
- **Leakage_Checker**: The notebook cells responsible for verifying no data leakage exists across splits
- **Visualiser**: The notebook cells responsible for producing all plots and sample image displays
- **Summary_Reporter**: The notebook cells responsible for printing the final preprocessing summary
- **Task_2_1**: The dataset located at `data/task2_damage_state_1/` (contains X_train, y_train, X_test, y_test)
- **Task_2_2**: The dataset located at `data/task2_damage_state_2/` (contains only X_train)
- **Selected_Dataset**: The dataset chosen for use after the Task 2_1 vs Task 2_2 comparison
- **HOG**: Histogram of Oriented Gradients — a handcrafted feature descriptor for structural/crack pattern representation
- **mmap_mode**: NumPy memory-mapped file mode `'r'`; reads array data from disk without loading it fully into RAM
- **Stratified_Split**: A train/validation split that preserves the original class ratio in both subsets
- **Fixed_Seed**: A fixed integer value (e.g. `42`) supplied to random number generators to ensure reproducibility
- **Pixel_Range**: The numeric range of pixel intensity values after normalisation (expected: `[0.0, 1.0]`)
- **InceptionV3_Input_Size**: The spatial dimensions required by InceptionV3 (`299 × 299` pixels)
- **InceptionV3_Preprocessing**: Per-pixel scaling to `[-1.0, 1.0]` as specified by `tf.keras.applications.inception_v3.preprocess_input`

---

## Requirements

### Requirement 1: File Existence and Metadata Inspection

**User Story:** As a data scientist, I want to verify that all required dataset files exist and review their sizes, so that I can confirm the dataset is complete before any processing begins.

#### Acceptance Criteria

1. WHEN the Notebook is executed, THE Dataset_Inspector SHALL check for the existence of `task2_X_train.npy`, `task2_y_train.npy`, `task2_X_test.npy`, and `task2_y_test.npy` under `data/task2_damage_state_1/`, and `task2_X_train.npy` under `data/task2_damage_state_2/`.
2. IF any required file is missing, THEN THE Dataset_Inspector SHALL raise a `RuntimeError` that lists all missing file paths, preventing dependent cells from executing.
3. WHEN all required files exist, THE Dataset_Inspector SHALL display each file's size in megabytes rounded to two decimal places.

---

### Requirement 2: Safe Shape and dtype Inspection

**User Story:** As a data scientist, I want to inspect array shapes, dtypes, image dimensions, number of channels, and pixel value ranges without loading the ~7 GB training array into RAM, so that I can understand the data structure while keeping memory usage safe.

#### Acceptance Criteria

1. WHEN inspecting `task2_X_train.npy` from Task_2_1, THE Dataset_Inspector SHALL load the array using `mmap_mode='r'` so that array contents are read on demand from disk and the full array contents are not copied into RAM.
2. WHEN any `.npy` file is loaded, THE Dataset_Inspector SHALL display its shape, dtype, minimum pixel value, and maximum pixel value.
3. WHEN inspecting a 4-D image array of shape `(N, H, W, C)`, THE Dataset_Inspector SHALL display `N` as the sample count, `H` as spatial height, `W` as spatial width, and `C` as number of channels. WHEN inspecting a 3-D image array of shape `(N, H, W)`, THE Dataset_Inspector SHALL display `N` as sample count, `H` as height, `W` as width, and report channels as 1.
4. WHEN pixel values are inspected, THE Dataset_Inspector SHALL verify that values fall within either `[0, 255]` (uint8) or `[0.0, 1.0]` (float32/float64); IF neither range is detected, THEN THE Dataset_Inspector SHALL display a warning that includes the array name, detected minimum value, and detected maximum value.
5. IF a `.npy` file cannot be opened due to a missing file or read error, THEN THE Dataset_Inspector SHALL display the file path and the reason for failure and continue inspecting remaining files without terminating the notebook.

---

### Requirement 3: Class Distribution Analysis

**User Story:** As a data scientist, I want to know the class counts, class percentages, and imbalance ratio for both the training and test sets, so that I can account for class imbalance in model training.

#### Acceptance Criteria

1. WHEN `task2_y_train.npy` is loaded, THE Dataset_Inspector SHALL compute and display the count and percentage (rounded to two decimal places) of samples with label `0` (Damaged) and label `1` (Undamaged).
2. WHEN `task2_y_test.npy` is loaded, THE Dataset_Inspector SHALL compute and display the count and percentage (rounded to two decimal places) of samples with label `0` (Damaged) and label `1` (Undamaged).
3. THE Dataset_Inspector SHALL compute and display the imbalance ratio as the count of the majority class divided by the count of the minority class, rounded to two decimal places, independently for both the training set and the test set.
4. WHEN class label arrays are loaded, THE Dataset_Inspector SHALL verify that the number of samples in the label array equals the number of samples in the corresponding image array; IF they differ, THE Dataset_Inspector SHALL display an error and halt further analysis for that mismatched set.
5. IF either the training or test label array contains only one unique class label, THEN THE Dataset_Inspector SHALL display a warning that class imbalance ratio cannot be computed for that set and skip the ratio calculation to avoid division by zero.

---

### Requirement 4: Representative Sample Image Display

**User Story:** As a data scientist, I want to see representative images from each class, so that I can visually confirm the data content and label correctness.

#### Acceptance Criteria

1. WHEN the Notebook is executed, THE Visualiser SHALL select exactly 3 sample images per class using fixed-seed random sampling (Fixed_Seed) from the training pool indices, and display them for the Damaged class (label `0`) and the Undamaged class (label `1`).
2. WHEN sample images are displayed, THE Visualiser SHALL annotate each image with its class label name (`Damaged` or `Undamaged`) and its array index.
3. WHEN displaying sample images, THE Visualiser SHALL access the image array using `mmap_mode='r'` so that only the indexed sample rows are read from disk.
4. IF the training pool contains fewer than 3 samples for any class, THEN THE Visualiser SHALL display a warning stating the class name and available sample count, and display only the available samples without error.

---

### Requirement 5: Task 2_1 vs Task 2_2 Comparison

**User Story:** As a data scientist, I want a structured comparison of Task_2_1 and Task_2_2, so that I can make an informed decision about which dataset to use.

#### Acceptance Criteria

1. WHEN the Notebook is executed, THE Dataset_Comparator SHALL compare Task_2_1 and Task_2_2 on the following attributes: file size in MB, array shape, spatial image dimensions, number of channels, pixel value dtype, minimum and maximum pixel values, and available split files (a split is considered available only when both the X and y `.npy` files for that split are present).
2. THE Dataset_Comparator SHALL display the comparison as a structured table or formatted text block with column or field headers that identify each dataset by name (e.g., `Task_2_1` and `Task_2_2`) and one row or entry per attribute.
3. WHEN the comparison is complete, THE Dataset_Comparator SHALL display a Markdown heading that contains the text "Selected Dataset" and a written justification that references at least one attribute value from the comparison results.
4. IF the class labels present in Task_2_1 differ from those in Task_2_2, OR IF the spatial image dimensions or channel count of Task_2_1 differ from those of Task_2_2, THEN THE Dataset_Comparator SHALL display a warning stating that combining the datasets is not recommended.

---

### Requirement 6: Train / Validation / Test Split

**User Story:** As a data scientist, I want a reproducible train/validation/test split that never contaminates training data with test data, so that model evaluation remains unbiased.

#### Acceptance Criteria

1. THE Split_Manager SHALL use the official train and test arrays from the Selected_Dataset as the training pool and test set respectively; test data SHALL NOT be used for training or validation.
2. WHEN creating the validation set, THE Split_Manager SHALL apply a Stratified_Split on the training pool using a Fixed_Seed to produce a validation subset such that the per-class proportion in the validation set equals the per-class proportion in the full training pool, within the tolerance inherent to integer rounding.
3. WHEN the split is complete, THE Split_Manager SHALL display the sample count, per-class counts, and per-class percentages for the train, validation, and test sets.
4. THE Split_Manager SHALL accept a configurable validation fraction defined as a named variable (default `0.15`) in the configuration cell, with a valid range of `(0.05, 0.50)` exclusive.
5. IF the validation fraction is outside the valid range `(0.05, 0.50)`, THEN THE Split_Manager SHALL display an error message stating the invalid value and the expected range, and halt execution of dependent cells.

---

### Requirement 7: Image Preprocessing

**User Story:** As a data scientist, I want all images converted to a consistent dtype and normalised pixel range, so that both the SVM and InceptionV3 pipelines receive clean, uniform input.

#### Acceptance Criteria

1. WHEN preprocessing images, THE Image_Preprocessor SHALL convert image arrays to `float32` dtype.
2. WHEN preprocessing images, THE Image_Preprocessor SHALL verify that each image has exactly 3 channels; IF an image has 1 channel, THEN THE Image_Preprocessor SHALL convert it to 3 channels by replicating the single channel.
3. IF an image has more than 3 channels, THEN THE Image_Preprocessor SHALL log the sample index and channel count, and skip that sample with a warning.
4. WHEN preprocessing images whose source dtype is `uint8`, THE Image_Preprocessor SHALL normalise pixel values to `[0.0, 1.0]` by dividing by `255.0`.
5. IF the source dtype is `float64`, THEN THE Image_Preprocessor SHALL cast values to `float32` without rescaling, assuming values are already in `[0.0, 1.0]`. IF the source dtype is neither `uint8`, `float32`, nor `float64`, THEN THE Image_Preprocessor SHALL display a warning with the dtype and sample index and skip that sample.
6. WHEN preprocessing images, THE Image_Preprocessor SHALL detect and log the index of any sample whose pixel values remain outside `[0.0, 1.0]` after normalisation; the preprocessing pipeline SHALL continue for all other samples.
7. THE Image_Preprocessor SHALL process images in batches so that no more than one batch is held in RAM simultaneously; it SHALL NOT create a full in-memory copy of the ~7 GB training array before normalisation.
8. THE Image_Preprocessor SHALL NOT apply data augmentation (flips, rotations, colour jitter, etc.) to any split.

---

### Requirement 8: HOG Feature Extraction for SVM

**User Story:** As a data scientist, I want a compact, interpretable feature vector derived from each preprocessed image for use with a linear SVM, so that the classical ML pipeline is efficient and appropriate for detecting structural damage patterns.

#### Acceptance Criteria

1. THE HOG_Extractor SHALL extract HOG features from each preprocessed image in the training, validation, and test sets; each resulting feature vector SHALL contain only finite numeric values (no NaN or infinity).
2. WHEN HOG features are extracted, THE HOG_Extractor SHALL display the HOG configuration used — `pixels_per_cell` in the range `[4, 16]`, `cells_per_block` in the range `[2, 4]`, and `orientations` in the range `[6, 18]` — and SHALL include at least two sentences in a Markdown cell explaining why HOG is suitable for crack and structural damage pattern detection.
3. THE HOG_Extractor SHALL produce a 1-D feature vector per image; the feature vector length SHALL be identical across all images in all splits, and THE HOG_Extractor SHALL print the feature vector dimensionality as an integer in a notebook output cell.
4. WHEN HOG feature extraction is complete, THE HOG_Extractor SHALL save the scaled training, validation, and test HOG feature arrays as separate files (e.g., `.npy`) in the `results/` directory so that downstream SVM training cells can load them without re-running extraction.
5. WHEN scaling HOG features, THE HOG_Extractor SHALL fit a `StandardScaler` (zero mean, unit variance) exclusively on the training set HOG features, then apply the fitted scaler to the validation and test set HOG features.
6. THE HOG_Extractor SHALL NOT use deep CNN features, pretrained network embeddings, or any feature derived from InceptionV3.

---

### Requirement 9: InceptionV3 Data Preparation

**User Story:** As a data scientist, I want images resized and scaled to InceptionV3's expected input format, so that the deep learning pipeline is ready for fine-tuning without further preprocessing.

#### Acceptance Criteria

1. WHEN preparing images for InceptionV3, THE InceptionV3_Preparer SHALL resize each preprocessed image to `299 × 299` pixels using bilinear interpolation, producing an output of shape `(299, 299, 3)`.
2. WHEN preparing images for InceptionV3, THE InceptionV3_Preparer SHALL apply `tf.keras.applications.inception_v3.preprocess_input` to scale pixel values to `[-1.0, 1.0]`, producing a `float32` output array.
3. WHEN preparing images for InceptionV3, THE InceptionV3_Preparer SHALL document in a Markdown cell that bilinear interpolation is used for resizing.
4. THE InceptionV3_Preparer SHALL NOT train, fine-tune, or run inference with InceptionV3.
5. THE InceptionV3_Preparer SHALL NOT apply data augmentation to any split.

---

### Requirement 10: Data Leakage Prevention

**User Story:** As a data scientist, I want explicit checks that no sample appears in more than one split and that all fitting is done only on training data, so that evaluation metrics are trustworthy.

#### Acceptance Criteria

1. THE Leakage_Checker SHALL verify that the integer positional indices used for the validation set are disjoint from the integer positional indices used for the training set; IF overlap is detected, THEN THE Leakage_Checker SHALL display an error and halt all notebook cells that depend on the split output.
2. THE Leakage_Checker SHALL verify that the integer positional test set indices are not present in the training or validation indices; IF overlap is detected, THEN THE Leakage_Checker SHALL display an error and halt all notebook cells that depend on the split output.
3. WHERE HOG feature scaling is applied, WHEN the Leakage_Checker runs, THE Leakage_Checker SHALL verify that the `StandardScaler` mean and scale parameters were computed from training-set indices only; IF the scaler was fitted on validation or test data, THEN THE Leakage_Checker SHALL display an error.
4. WHERE HOG feature scaling is not applied, THE Leakage_Checker SHALL skip the scaler check and proceed without error.
5. IF all leakage checks pass, THEN THE Leakage_Checker SHALL display the confirmation message: "No data leakage detected".

---

### Requirement 11: Visualisations

**User Story:** As a data scientist, I want clear charts and image grids throughout the notebook, so that the dataset characteristics are easy to understand at a glance.

#### Acceptance Criteria

1. THE Visualiser SHALL produce a bar chart showing the sample count for each class (Damaged, Undamaged) with numeric labels on each bar, covering all three splits (train, validation, test).
2. THE Visualiser SHALL produce an image grid displaying exactly 3 representative Damaged images and exactly 3 representative Undamaged images selected via fixed-seed random sampling (Fixed_Seed), with class label annotations on each image.
3. WHERE HOG feature extraction is performed, THE Visualiser SHALL display a HOG visualisation image for at least one sample to illustrate what the extractor captures.
4. THE Visualiser SHALL display all plots inline within the Notebook using `matplotlib` with a minimum figure size of `8 × 4` inches per plot.

---

### Requirement 12: Final Preprocessing Summary

**User Story:** As a data scientist, I want a concise summary cell at the end of the notebook that documents every preprocessing decision, so that the notebook is self-documenting and reproducible.

#### Acceptance Criteria

1. THE Summary_Reporter SHALL display the name of the Selected_Dataset and the written justification for selecting it.
2. THE Summary_Reporter SHALL display the final sample counts for train, validation, and test sets alongside per-class counts and percentages for each set.
3. THE Summary_Reporter SHALL display the image dimensions, number of channels, dtype, and Pixel_Range of the preprocessed arrays.
4. THE Summary_Reporter SHALL display the list of preprocessing steps applied in the order they were applied.
5. THE Summary_Reporter SHALL display the SVM feature method name, HOG configuration parameters, and the resulting feature vector dimensionality.
6. THE Summary_Reporter SHALL display the InceptionV3 target input size, resize method, and pixel scaling range.
7. WHEN all preprocessing steps are complete, THE Summary_Reporter SHALL display the message: "Preprocessing complete. Data is ready for: (1) classical-ml branch (SVM + HOG features), (2) deep-learning branch (InceptionV3 fine-tuning)."

---

### Requirement 13: Notebook Organisation and Memory Safety

**User Story:** As a data scientist, I want the notebook structured with clear Markdown section headings and memory-safe loading patterns throughout, so that it runs reliably on machines with limited RAM.

#### Acceptance Criteria

1. THE Notebook SHALL contain the following top-level Markdown sections in order: `1. Project Overview`, `2. Imports`, `3. Dataset Paths`, `4. Dataset Inspection`, `5. Task 2_1 vs Task 2_2 Comparison`, `6. Class Distribution`, `7. Sample Image Visualization`, `8. Train/Validation/Test Split`, `9. Image Preprocessing`, `10. Feature Engineering for SVM`, `11. InceptionV3 Preprocessing`, `12. Data Leakage Checks`, `13. Final Dataset Summary`.
2. THE Notebook SHALL NOT read the full `task2_X_train.npy` array from Task_2_1 into RAM in a single call without `mmap_mode`.
3. THE Notebook SHALL NOT modify, move, delete, or overwrite any original `.npy` file in the `data/` directory.
4. THE Notebook SHALL NOT create any Python module files under `src/` or any other directory outside `notebooks/`.
5. THE Notebook SHALL define all configurable parameters (validation fraction, Fixed_Seed, HOG parameters, InceptionV3_Input_Size) as named variables in a dedicated configuration cell near the top of the notebook.
