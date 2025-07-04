# Brain Age Prediction Using 3D CNN

This project predicts brain age from MRI scans using 3D Convolutional Neural Networks (CNN).

## Features

* Train a 3D CNN model on single-modality MRI data.
* Supports multi-modal fusion of two MRI types (VBM and Quasiraw) by averaging predictions.
* Uses data generators to load MRI data batch-wise without resizing.
* Outputs training curves and prediction scatter plots.
* Saves the trained model.
* UI Interface.

## Requirements

* Python 3.9
* TensorFlow
* NumPy
* Pandas
* Scikit-image
* Matplotlib
* Scikit-learn

## How to Run

1. Prepare your MRI `.npy` data and index CSV files (`my_index.csv` for single modality, `my_index_vbm_quasiraw.csv` for multi-modal fusion).
2. Run single modality training:

```bash
python train_brain_age.py
```

3. Run multi-modal fusion training and testing:

```bash
python decision_level_fusion_generator.py
```

## Data Format

* MRI files: 3D numpy arrays (`.npy`).
* Index CSV: contains file paths and ages.
