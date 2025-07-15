# Brain Age Prediction Using 3D CNN

This project aims to predict brain age from MRI scans using advanced 3D Convolutional Neural Networks (CNNs). It provides a complete pipeline from data preprocessing and model training to user-friendly prediction and visualization tools.

## Features

- **3D CNN Model**: Train and evaluate a deep learning model for brain age prediction using 3D MRI data.
- **Multi-Modal Fusion**: Support for combining different MRI modalities (e.g., VBM and Quasiraw) to improve prediction accuracy.
- **Data Quality Control**: Tools for MRI data quality checking and suspect sample management.
- **Batch Data Generator**: Efficiently loads large MRI datasets without resizing, supporting scalable training.
- **User Interface**: Intuitive GUI for loading models, selecting MRI files, visualizing slices, and predicting brain age.
- **Visualization**: Training curves, scatter plots of predictions, and MRI slice viewers.
- **Early Stopping**: Prevents overfitting during training.

## Installation

### Requirements

- Python 3.9
- TensorFlow
- NumPy
- Pandas
- Scikit-image
- Matplotlib
- Scikit-learn

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation

- Prepare your MRI data as `.npy` files (3D NumPy arrays).
- Create an index CSV file (e.g., `my_index.csv`) with columns for file paths and ages.

### 2. Model Training

Train a single-modality model:
```bash
python train/train_brain_age.py
```

For multi-modal fusion (if supported):
```bash
python train/train_brain_age_multimodal.py
```

### 3. Model Evaluation

Evaluate the trained model:
```bash
python test/eval_model.py
```

### 4. Prediction and Visualization

Launch the GUI for brain age prediction:
```bash
python predict/predict_brain_age.py
```
- Load a trained model and MRI file.
- View MRI slices and predict brain age with a single click.
