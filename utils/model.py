import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, Model

def evaluate_model(model, val_generator, config, logger):
    """Evaluate the model and generate reports."""
    logger.info("=" * 50)
    logger.info("EVALUATING ON TEST SET")
    logger.info("=" * 50)
    
    validation_steps = val_generator.samples // config.BATCH_SIZE
    
    test_loss, test_accuracy = model.evaluate(val_generator, steps=validation_steps, verbose=1)
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"Test Loss: {test_loss:.4f}")
    
    # Generate predictions for confusion matrix
    logger.info("Generating predictions for confusion matrix...")
    val_generator.reset()
    test_predictions = model.predict(val_generator, steps=validation_steps, verbose=1)
    test_pred_classes = np.argmax(test_predictions, axis=1)
    
    val_generator.reset()
    test_labels = val_generator.classes

    min_len = min(len(test_pred_classes), len(test_labels))
    test_pred_classes = test_pred_classes[:min_len]
    test_labels = test_labels[:min_len]
    
    logger.info(f"Number of test samples: {len(test_labels)}")
    logger.info(f"Number of predictions: {len(test_pred_classes)}")
    
    # Generate confusion matrices
    cm = confusion_matrix(test_labels, test_pred_classes)
    cm_normalized = confusion_matrix(test_labels, test_pred_classes, normalize='true')
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES, ax=axes[0])
    axes[0].set_title('Confusion Matrix (Raw Counts)', fontsize=16)
    axes[0].set_xlabel('Predicted Label', fontsize=14)
    axes[0].set_ylabel('True Label', fontsize=14)
    

    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues', 
                xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES, ax=axes[1])
    axes[1].set_title('Normalized Confusion Matrix', fontsize=16)
    axes[1].set_xlabel('Predicted Label', fontsize=14)
    axes[1].set_ylabel('True Label', fontsize=14)
    
    plt.tight_layout()
    confusion_matrix_path = os.path.join(config.MODELS_DIR, 'fine_tuned_vgg16_confusion_matrices.png')
    plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Classification report
    logger.info("=" * 50)
    logger.info("CLASSIFICATION REPORT")
    logger.info("=" * 50)
    classification_report_str = classification_report(test_labels, test_pred_classes, target_names=config.CLASS_NAMES, digits=4)
    logger.info(classification_report_str)
    
    # Per-class accuracy
    per_class_accuracy = cm_normalized.diagonal()
    logger.info("=" * 50)
    logger.info("PER-CLASS ACCURACY")
    logger.info("=" * 50)
    for i, class_name in enumerate(config.CLASS_NAMES):
        logger.info(f"{class_name:12}: {per_class_accuracy[i]:.4f} ({per_class_accuracy[i]*100:.2f}%)")
    
    logger.info("Model evaluation completed")
    return test_accuracy, test_loss

def save_model(model, config, combined_history, logger):
    """Save the model in multiple formats."""
    logger.info("=" * 50)
    logger.info("SAVING MODEL")
    logger.info("=" * 50)
    
    model_name = "fine_tuned_vgg16_final"
    
    keras_path = os.path.join(config.MODELS_DIR, f"{model_name}.keras")
    model.save(keras_path)
    logger.info(f"Model saved as {keras_path}")
    
    savedmodel_path = os.path.join(config.MODELS_DIR, f"{model_name}_savedmodel")
    try:
        model.export(savedmodel_path)
        logger.info(f"Model exported as {savedmodel_path} (SavedModel format)")
    except AttributeError:
        model.save(savedmodel_path, save_format='tf')
        logger.info(f"Model saved as {savedmodel_path} (SavedModel format)")
    
    weights_path = os.path.join(config.MODELS_DIR, f"{model_name}.weights.h5")
    model.save_weights(weights_path)
    logger.info(f"Model weights saved as {weights_path}")
    
    if combined_history:
        history_path = os.path.join(config.MODELS_DIR, f"{model_name}_history.pkl")
        with open(history_path, 'wb') as f:
            pickle.dump(combined_history, f)
        logger.info(f"Training history saved as {history_path}")
    
    logger.info("Model saving completed")


def build_fine_tuned_vgg16(config, logger):
    """Build and return the fine-tuned VGG16 model."""
    logger.info("Building VGG16 model...")
    
    # Load pre-trained VGG16 model
    vgg16_base = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3)
    )
    
    vgg16_base.trainable = False
    
    inputs = vgg16_base.input
    x = vgg16_base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(config.NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    logger.info("Model built successfully")
    
    return model, vgg16_base


def load_or_create_model(config, logger):
    """Load existing model or create new one based on configuration."""
    
    model_path = os.path.join(config.MODELS_DIR, "fine_tuned_vgg16_final.keras")
    skip_training = False
    vgg16_base = None
    
    if os.path.exists(model_path) and not config.FORCE_RETRAIN:
        logger.info("=" * 50)
        logger.info("EXISTING MODEL FOUND")
        logger.info("=" * 50)
        
        try:
            model = tf.keras.models.load_model(model_path)
            logger.info(f"Successfully loaded existing model: {model_path}")
            
            if config.CONTINUE_TRAINING:
                skip_training = False
                logger.info(f"🔄 CONTINUE_TRAINING enabled - will train for {config.ADDITIONAL_EPOCHS} more epochs...")
            elif config.AUTO_SKIP_IF_EXISTS:
                skip_training = True
                logger.info("⏭️ Auto-skipping training - proceeding to evaluation...")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Creating new model instead...")
            model, vgg16_base = build_fine_tuned_vgg16(config, logger)
            skip_training = False
            
            model.compile(
                optimizer=Adam(learning_rate=config.FRESH_LEARNING_RATE),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
    else:
        if config.FORCE_RETRAIN and os.path.exists(model_path):
            logger.info("🔄 FORCE_RETRAIN is enabled - will overwrite existing model")
        else:
            logger.info(f"No existing model found at {model_path}")
        
        logger.info("Creating new model for training...")
        model, vgg16_base = build_fine_tuned_vgg16(config, logger)
        skip_training = False
        
        # Compile model for initial training
        model.compile(
            optimizer=Adam(learning_rate=config.FRESH_LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    return model, vgg16_base, skip_training

def print_model_info(model, logger, phase_name="Model"):
    """Print model information."""
    logger.info(f"\n{phase_name} Info:")
    logger.info(f"Total params: {model.count_params():,}")
    logger.info(f"Trainable params: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
    logger.info(f"Non-trainable params: {sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights]):,}")
