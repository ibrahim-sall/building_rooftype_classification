#!/usr/bin/env python3
"""
Fine-Tuned VGG16 Model Training Script - Refactored Version
==========================================================

This script implements a comprehensive training pipeline for building roof type classification
using a Fine-Tuned VGG16 model with proper function structure and logging.

Features:
1. **Smart Model Loading**: Automatically detects and loads existing models
2. **Configuration Options**: Easy configuration through constants
3. **Two-Phase Training**: Frozen → fine-tuned training approach
4. **Continue Training**: Additional training from existing models
5. **Comprehensive Evaluation**: Confusion matrix and classification reports
6. **Multiple Save Formats**: .keras, .h5, SavedModel, weights
7. **Proper Logging**: Structured logging throughout the process

Usage:
    python train.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.applications import VGG16
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

import seaborn as sns
import os
import logging
import pickle

from utils.model import *

# ============================================================================
def setup_logger():
    """Setup and return a logger instance."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

class Config:
    # Model behavior settings
    FORCE_RETRAIN = False
    AUTO_SKIP_IF_EXISTS = True
    CONTINUE_TRAINING = False
    ADDITIONAL_EPOCHS = 30
    
    # Training parameters
    CONTINUE_LEARNING_RATE = 0.0001
    FRESH_LEARNING_RATE = 0.001
    FINE_TUNE_LEARNING_RATE = 0.00001
    
    # Model parameters
    NUM_CLASSES = 7
    BATCH_SIZE = 64
    IMG_HEIGHT = 140
    IMG_WIDTH = 140
    CLASS_NAMES = ['complex', 'flat', 'gable', 'halfhip', 'hip', 'L-shaped', 'pyramid']
    
    # Directories
    TRAIN_DIR = 'output/train'
    VAL_DIR = 'output/val'
    MODELS_DIR = 'models'
    
    # Training phases
    EPOCHS_PHASE1 = 20
    EPOCHS_PHASE2 = 30
    FINE_TUNE_AT = 15

def check_directories(config, logger):
    """Check if required directories exist."""
    logger.info("Checking data directories...")
    
    if not os.path.exists(config.TRAIN_DIR):
        raise FileNotFoundError(f"Training directory '{config.TRAIN_DIR}' not found!")
    if not os.path.exists(config.VAL_DIR):
        raise FileNotFoundError(f"Validation directory '{config.VAL_DIR}' not found!")
    
    # Create models directory if it doesn't exist
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    logger.info("All directories found/created")

def create_data_generators(config, logger):
    """Create and return training and validation data generators."""
    logger.info("Creating data generators...")
    
    # Create data generators with augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Validation data generator (only rescaling)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )
    
    # Load validation data
    val_generator = val_datagen.flow_from_directory(
        config.VAL_DIR,
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        seed=42
    )
    
    logger.info(f"Training samples: {train_generator.samples}")
    logger.info(f"Validation samples: {val_generator.samples}")
    logger.info(f"Classes found: {list(train_generator.class_indices.keys())}")
    
    return train_generator, val_generator


def create_callbacks(config, logger):
    """Create and return training callbacks."""
    logger.info("Setting up training callbacks...")
    
    callbacks = [
        CSVLogger(os.path.join(config.MODELS_DIR, 'fine_tuned_vgg16_training.log'), separator=','),
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(config.MODELS_DIR, 'best_fine_tuned_vgg16.keras'),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    return callbacks

def train_phase1(model, train_generator, val_generator, config, callbacks, logger):
    """Train Phase 1: Frozen VGG16 base."""
    logger.info("=" * 50)
    logger.info("PHASE 1: Training with frozen VGG16 base")
    logger.info("=" * 50)
    
    steps_per_epoch = train_generator.samples // config.BATCH_SIZE
    validation_steps = val_generator.samples // config.BATCH_SIZE
    
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Validation steps: {validation_steps}")
    
    history_phase1 = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=config.EPOCHS_PHASE1,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    logger.info("Phase 1 training completed")
    return history_phase1

def train_phase2(model, vgg16_base, train_generator, val_generator, config, callbacks, history_phase1, logger):
    """Train Phase 2: Fine-tune top VGG16 layers."""
    logger.info("=" * 50)
    logger.info("PHASE 2: Fine-tuning top VGG16 layers")
    logger.info("=" * 50)
    
    # Unfreeze top layers of VGG16
    vgg16_base.trainable = True
    
    # Freeze layers before fine_tune_at
    for layer in vgg16_base.layers[:config.FINE_TUNE_AT]:
        layer.trainable = False
    
    # Use lower learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=config.FINE_TUNE_LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info(f"Fine-tuning from layer {config.FINE_TUNE_AT} onwards")
    print_model_info(model, logger, "Phase 2")
    
    # Show trainable layers
    logger.info("Trainable layers in VGG16 base:")
    for i, layer in enumerate(vgg16_base.layers):
        if layer.trainable:
            logger.info(f"  Layer {i}: {layer.name} - Trainable")
    
    steps_per_epoch = train_generator.samples // config.BATCH_SIZE
    validation_steps = val_generator.samples // config.BATCH_SIZE
    
    history_phase2 = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=config.EPOCHS_PHASE2,
        initial_epoch=history_phase1.epoch[-1],
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    logger.info("Phase 2 training completed")
    return history_phase2

def continue_training(model, train_generator, val_generator, config, callbacks, logger):
    """Continue training from existing model."""
    logger.info("=" * 50)
    logger.info("PHASE 3: CONTINUE TRAINING FROM EXISTING MODEL")
    logger.info("=" * 50)
    
    steps_per_epoch = train_generator.samples // config.BATCH_SIZE
    validation_steps = val_generator.samples // config.BATCH_SIZE
    
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Validation steps: {validation_steps}")
    logger.info(f"Additional epochs: {config.ADDITIONAL_EPOCHS}")
    
    # Use very low learning rate for continued training
    model.compile(
        optimizer=Adam(learning_rate=config.CONTINUE_LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print_model_info(model, logger, "Continue Training")
    
    history_continue = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=config.ADDITIONAL_EPOCHS,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    logger.info("Continue training completed")
    return history_continue

def combine_histories(hist1, hist2):
    """Combine two training histories."""
    combined_history = {}
    for key in hist1.history.keys():
        combined_history[key] = hist1.history[key] + hist2.history[key]
    return combined_history

def plot_training_history(history, config, title_suffix="", logger=None):
    """Plot and save training history."""
    if logger:
        logger.info(f"Plotting training history{title_suffix}...")
    
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Model Accuracy {title_suffix}', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'Model Loss {title_suffix}', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    history_plot_path = os.path.join(config.MODELS_DIR, f'fine_tuned_vgg16_training_history{title_suffix.lower().replace(" ", "_")}.png')
    plt.savefig(history_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    if logger:
        logger.info(f"Training history plot saved: {history_plot_path}")



def main():
    """Main training function."""
    logger = setup_logger()
    config = Config()
    
    # Set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    
    logger.info("=" * 60)
    logger.info("FINE-TUNED VGG16 TRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Number of classes: {config.NUM_CLASSES}")
    logger.info(f"Image size: {config.IMG_HEIGHT}x{config.IMG_WIDTH}")
    logger.info(f"Batch size: {config.BATCH_SIZE}")
    
    try:
        check_directories(config, logger)
        
        train_generator, val_generator = create_data_generators(config, logger)
        
        model, vgg16_base, skip_training = load_or_create_model(config, logger)
        
        print_model_info(model, logger)
        
        combined_history = None
        
        if not skip_training:
            callbacks = create_callbacks(config, logger)
            
            if config.CONTINUE_TRAINING and os.path.exists(os.path.join(config.MODELS_DIR, "fine_tuned_vgg16_final.keras")):
                history_continue = continue_training(model, train_generator, val_generator, config, callbacks, logger)
                combined_history = history_continue.history
            else:
                history_phase1 = train_phase1(model, train_generator, val_generator, config, callbacks, logger)
                
                history_phase2 = train_phase2(model, vgg16_base, train_generator, val_generator, config, callbacks, history_phase1, logger)
                
                combined_history = combine_histories(history_phase1, history_phase2)
            
            logger.info("=" * 50)
            logger.info("TRAINING RESULTS")
            logger.info("=" * 50)
            plot_training_history(combined_history, config, logger=logger)
        else:
            logger.info("=" * 50)
            logger.info("SKIPPING TRAINING - USING LOADED MODEL")
            logger.info("=" * 50)
        
        test_accuracy, test_loss = evaluate_model(model, val_generator, config, logger)
        
        if not skip_training:
            save_model(model, config, combined_history, logger)
        else:
            logger.info("=" * 50)
            logger.info("MODEL ALREADY EXISTS - SKIPPING SAVE")
            logger.info("=" * 50)
        
        # Final summary
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        logger.info("Files generated:")
        logger.info(f"  - fine_tuned_vgg16_final.keras (native Keras format)")
        logger.info(f"  - fine_tuned_vgg16_final.h5 (legacy HDF5 format)")
        logger.info(f"  - fine_tuned_vgg16_final_savedmodel/ (SavedModel format)")
        logger.info(f"  - fine_tuned_vgg16_final.weights.h5 (weights only)")
        logger.info(f"  - fine_tuned_vgg16_training_history.png")
        logger.info(f"  - fine_tuned_vgg16_confusion_matrices.png")
        logger.info(f"  - fine_tuned_vgg16_training.log")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()