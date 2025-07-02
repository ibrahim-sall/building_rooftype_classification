# Building Roof Type Classification

Deep Learning based roof type classification using very high resolution aerial imagery. This repository contains the implementation and improvements for the paper accepted to ISPRS 2021 Congress.

## Summary

This repository provides a comprehensive deep learning pipeline for classifying building roof types from high-resolution aerial imagery. It features robust training scripts, flexible inference options for both single and multi-roof scenarios, and tools for GIS integration and visualization.

- [Features](#features)
- [Training Configuration](#training-configuration)
- [Hardware Requirements](#hardware-requirements)
- [File Structure](#file-structure)
- [Advanced Usage](#advanced-usage)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

## Features

### 🏗️ **Enhanced Training Pipeline** (`train.py`)
- **Fine-tuned VGG16** with two-phase training (frozen base → fine-tuning)
- **Smart Model Management**: Skip existing, continue training, or force retrain
- **Configurable Learning Rates** for each training phase
- **Resume Training**: Continue from existing models for additional epochs

### 🔍 **Two Inference Approaches**

#### 1. **Single Roof Classification** (`inference.py`)
- **Use Case**: Individual roof images (one roof per image)
- **Input**: Directory of pre-cropped roof images
- **Output**: One prediction per image file
- **Best For**: Validation sets, pre-segmented roof datasets

#### 2. **Multi-Roof Orthophoto Analysis** (`orthophoto_inference.py`)
- **Use Case**: Large orthophotos/aerial images with multiple roofs
- **Input**: Aerial/satellite imagery containing multiple buildings
- **Output**: Multiple detections with coordinates and classifications
- **Features**: 
  - Sliding window detection with NMS and confidence filtering
  - **Multi-channel support**: RGB, RGBA, RGBI, multi-spectral images
  - **Shapefile export**: GIS-compatible polygon data with attributes
  - **Mask generation**: Class and confidence raster masks
  - **Visualization**: Annotated images with bounding boxes
  - **Geographic coordinates**: Support for georeferenced imagery
- **Best For**: Urban planning, roof surveys, aerial imagery analysis, GIS workflows

## Training Configuration

- **Image Size**: 140x140 pixels
- **Batch Size**: 64
- **Phase 1**: 20 epochs with frozen VGG16 base (lr=0.001)
- **Phase 2**: 30 epochs with fine-tuning top layers (lr=0.00001)
- **Classes**: 7 roof types (complex, flat, gable, halfhip, hip, L-shaped, pyramid)

## Hardware Requirements

- GPU recommended for faster training
- Minimum 8GB RAM
- At least 1GB free disk space for model files and outputs

## File Structure

```
building_rooftype_classification/
├── train.py                    # Enhanced training script
├── inference.py               # Single roof inference
├── orthophoto_inference.py    # Multi-roof orthophoto analysis
├── compare_inference_approaches.py  # Usage comparison guide
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── TRAINING_README.md        # Detailed training guide
├── output/                   # Training data
│   ├── train/               # Training images by class
│   └── val/                # Validation images by class
├── test_orthophotos/        # Large aerial images for testing
└── models/                  # Saved model files
    ├── fine_tuned_vgg16_final.keras
    ├── fine_tuned_vgg16_final.h5
    └── fine_tuned_vgg16_final_savedmodel/
```

## Advanced Usage

### Custom Model Training
```bash
# Train with custom parameters
python train.py \
    --initial_epochs 15 \
    --fine_tune_epochs 25 \
    --batch_size 16 \
    --initial_lr 0.0005
```

### Batch Processing with Confidence Filtering
```bash
# Process multiple orthophotos with high confidence only
python orthophoto_inference.py \
    --input_dir aerial_survey/ \
    --confidence_threshold 0.85 \
    --nms_threshold 0.2 \
    --output_dir survey_results/ \
    --visualize
```

### Validation and Testing
```bash
# Test on validation set
python inference.py \
    --input_dir output/val/ \
    --output_csv validation_results.csv \
    --confidence_threshold 0.5

# Generate confusion matrix
python -c "from train import *; generate_confusion_matrix()"
```

## Results 

![alt text](results.jpg)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{roof_classification_2021,
  title={Deep Learning based roof type classification using very high resolution aerial imagery},
  author={[Authors]},
  journal={ISPRS Archives},
  year={2021},
  url={https://doi.org/10.5194/isprs-archives-XLIII-B3-2021-55-2021}
}
```

## License

[Add your license information here]