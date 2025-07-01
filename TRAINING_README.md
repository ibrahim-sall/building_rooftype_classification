# Fine-Tuned VGG16 Training Script

This script trains a Fine-Tuned VGG16 model for building roof type classification with normalized confusion matrix evaluation and comprehensive model saving.

## Features

- **Two-Phase Training**: Initial training with frozen VGG16 base, followed by fine-tuning top layers
- **Data Augmentation**: Comprehensive augmentation for training data
- **Advanced Callbacks**: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, and CSV logging
- **Comprehensive Evaluation**: Confusion matrices, classification reports, and per-class accuracy
- **Multiple Save Formats**: .h5, SavedModel, weights-only, and training history

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

## Data Structure

Make sure your data is organized in the following structure:
```
output/
├── train/
│   ├── complex/
│   ├── flat/
│   ├── gable/
│   ├── halfhip/
│   ├── hip/
│   ├── L-shaped/
│   └── pyramid/
└── val/
    ├── complex/
    ├── flat/
    ├── gable/
    ├── halfhip/
    ├── hip/
    ├── L-shaped/
    └── pyramid/
```
### Orthos
Directory Structure Expected:
```
    input_dir/
    ├── orthophoto1.tif
    ├── orthophoto2.tif
    └── footprints/
        ├── orthophoto1_footprints.shp
        ├── orthophoto2_footprints.shp
        └── ...
    
    Optional DSM Structure:
    dsm_dir/ (or same as input_dir)
    ├── orthophoto1_dsm.tif
    ├── orthophoto2_dsm.tif
    └── ...
```
## Usage

Run the training script:
```bash
python train.py
```

## Output Files

The script will generate the following files:
- `fine_tuned_vgg16_final.h5` - Complete trained model
- `fine_tuned_vgg16_final/` - SavedModel format directory
- `fine_tuned_vgg16_final_weights.h5` - Model weights only
- `fine_tuned_vgg16_final_history.pkl` - Training history data
- `fine_tuned_vgg16_training_history.png` - Training curves visualization
- `fine_tuned_vgg16_confusion_matrices.png` - Confusion matrix plots
- `fine_tuned_vgg16_training.log` - Detailed training log
- `best_fine_tuned_vgg16.h5` - Best model checkpoint during training

## Model Architecture

- **Base Model**: VGG16 pre-trained on ImageNet (frozen initially)
- **Custom Classifier**: 
  - GlobalAveragePooling2D
  - Dense(512) + ReLU + Dropout(0.5)
  - Dense(256) + ReLU + Dropout(0.3)
  - Dense(128) + ReLU + Dropout(0.2)
  - Dense(7) + Softmax (output layer)

## Training Configuration

- **Image Size**: 140x140 pixels
- **Batch Size**: 64
- **Phase 1**: 20 epochs with frozen VGG16 base (lr=0.001)
- **Phase 2**: 30 epochs with fine-tuning top layers (lr=0.00001)
- **Classes**: 7 roof types (complex, flat, gable, halfhip, hip, L-shaped, pyramid)

## Hardware Requirements

- GPU recommended for faster training
- Minimum 8GB RAM
- At least 2GB free disk space for model files and outputs
