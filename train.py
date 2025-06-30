"""
Fine-Tuned VGG16 Model Training Script with Model Reloading
===========================================================

This script implements a comprehensive training pipeline for building roof type classification
using a Fine-Tuned VGG16 model with the following features:

1. **Smart Model Loading**: 
   - Automatically detects if a trained model exists
   - Loads existing model and skips training (configurable)
   - Supports force retraining with FORCE_RETRAIN flag
   - Supports continue training with CONTINUE_TRAINING flag

2. **Configuration Options**:
   - FORCE_RETRAIN: Set to True to retrain even if model exists
   - AUTO_SKIP_IF_EXISTS: Set to False for manual control
   - CONTINUE_TRAINING: Set to True to continue training from existing model
   - ADDITIONAL_EPOCHS: Number of additional epochs when continuing training

3. **Training Modes**:
   - **Fresh Training**: Two-phase training (frozen → fine-tuned)
   - **Continue Training**: Additional training from existing model with very low learning rate
   - **Skip Training**: Load model and proceed directly to evaluation

4. **Two-Phase Training** (Fresh):
   - Phase 1: Train custom classifier with frozen VGG16 base
   - Phase 2: Fine-tune top VGG16 layers with lower learning rate

5. **Continue Training** (Existing model):
   - Phase 3: Additional training with very low learning rate (0.00001)
   - Preserves existing knowledge while allowing incremental improvements

6. **Comprehensive Evaluation**:
   - Normalized confusion matrix
   - Classification report
   - Per-class accuracy breakdown

7. **Multiple Save Formats**:
   - .keras (recommended)
   - .h5 (legacy)
   - SavedModel (deployment)
   - Weights only
   - Training history

Usage Examples:
    # Fresh training (if no model exists)
    python train.py
    
    # Continue training from existing model for 10 more epochs
    CONTINUE_TRAINING = True
    ADDITIONAL_EPOCHS = 10
    python train.py
    
    # Force retrain (overwrite existing model)
    FORCE_RETRAIN = True
    python train.py
    
    # Just evaluate existing model
    AUTO_SKIP_IF_EXISTS = True  # (default)
    python train.py

The script will automatically:
- Load existing model if found (fine_tuned_vgg16_final.h5)
- Skip training and proceed to evaluation (default)
- Continue training if CONTINUE_TRAINING=True
- Or retrain if FORCE_RETRAIN=True
"""

# Fine-Tuned VGG16 Model Training with Normalized Confusion Matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.applications import VGG16
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import seaborn as sns
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# TRAINING CONFIGURATION - Modify these settings as needed
# ============================================================================

# Model behavior settings
FORCE_RETRAIN = False        # True = Always retrain from scratch
AUTO_SKIP_IF_EXISTS = True   # True = Auto-skip training if model exists  
CONTINUE_TRAINING = True   # True = Continue training from existing model
ADDITIONAL_EPOCHS = 30      # Number of additional epochs for continue training

# Training parameters
CONTINUE_LEARNING_RATE = 0.0001   # Moderate learning rate for continue training (10x higher)
FRESH_LEARNING_RATE = 0.001       # Standard learning rate for fresh training
FINE_TUNE_LEARNING_RATE = 0.00001 # Learning rate for fine-tuning phase

# Alternative learning rates (uncomment to use):
# CONTINUE_LEARNING_RATE = 0.00001  # Very conservative (original)
# CONTINUE_LEARNING_RATE = 0.0005   # More aggressive for significant improvements
# CONTINUE_LEARNING_RATE = 0.00005  # Slightly more aggressive than original

# ============================================================================
# END OF CONFIGURATION SECTION
# ============================================================================

# Configuration
num_classes = 7
batch_size = 64
img_height = 140
img_width = 140
class_names = ['complex', 'flat', 'gable', 'halfhip', 'hip', 'L-shaped', 'pyramid']

# Data directories
train_dir = 'output/train'
val_dir = 'output/val'

print("Starting Fine-Tuned VGG16 Training...")
print(f"Number of classes: {num_classes}")
print(f"Image size: {img_height}x{img_width}")
print(f"Batch size: {batch_size}")

# Check if data directories exist
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"Training directory '{train_dir}' not found!")
if not os.path.exists(val_dir):
    raise FileNotFoundError(f"Validation directory '{val_dir}' not found!")

# Data loading and preprocessing
print("\n" + "="*50)
print("LOADING AND PREPROCESSING DATA")
print("="*50)

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
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    seed=42
)

# Load validation data
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    seed=42
)

# We'll use the generators directly instead of converting to tf.data.Dataset
# This is more compatible and easier to work with

print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")
print(f"Classes found: {list(train_generator.class_indices.keys())}")

# Load pre-trained VGG16 model
vgg16_base = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(img_height, img_width, 3)
)

# Freeze base model initially
vgg16_base.trainable = False

# Build custom classifier on top
def build_fine_tuned_vgg16():
    inputs = vgg16_base.input
    x = vgg16_base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs)

# Create model or load existing one
models_dir = "models"
model_path = os.path.join(models_dir, "fine_tuned_vgg16_final.wheight.h5")
skip_training = False

# Ensure models directory exists
os.makedirs(models_dir, exist_ok=True)

if os.path.exists(model_path) and not FORCE_RETRAIN:
    print(f"\n{'='*50}")
    print("EXISTING MODEL FOUND")
    print("="*50)
    try:
        fine_tuned_vgg16 = tf.keras.models.load_model(model_path)
        print(f"✅ Successfully loaded existing model: {model_path}")
        
        if CONTINUE_TRAINING:
            skip_training = False
            print(f"🔄 CONTINUE_TRAINING enabled - will train for {ADDITIONAL_EPOCHS} more epochs...")
        elif AUTO_SKIP_IF_EXISTS:
            skip_training = True
            print("⏭️  Auto-skipping training - proceeding to evaluation...")
        else:
            print("\nWhat would you like to do?")
            print("1. Skip training and evaluate existing model")
            print("2. Continue training from loaded model")
            print("3. Start fresh training (overwrite existing model)")
            
            # For automated execution, default to evaluation
            choice = "1"  # You can modify this or add input() for interactive mode
            
            if choice == "1":
                skip_training = True
                print("⏭️  Skipping training - proceeding to evaluation...")
            elif choice == "2":
                skip_training = False
                print("🔄 Continuing training from loaded model...")
            else:
                skip_training = False
                print("🆕 Starting fresh training...")
                fine_tuned_vgg16 = build_fine_tuned_vgg16()
                
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Creating new model instead...")
        fine_tuned_vgg16 = build_fine_tuned_vgg16()
        skip_training = False
        
        # Compile model for training
        fine_tuned_vgg16.compile(
            optimizer=Adam(learning_rate=FRESH_LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
else:
    if FORCE_RETRAIN and os.path.exists(model_path):
        print(f"\n🔄 FORCE_RETRAIN is enabled - will overwrite existing model")
    else:
        print(f"\n📝 No existing model found at {model_path}")
    
    print("Creating new model for training...")
    fine_tuned_vgg16 = build_fine_tuned_vgg16()
    skip_training = False
    
    # Compile model for initial training
    fine_tuned_vgg16.compile(
        optimizer=Adam(learning_rate=FRESH_LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

print("\nModel Architecture:")
fine_tuned_vgg16.summary()

if not skip_training:
    print(f"\nPhase 1 - Frozen VGG16 base:")
    print(f"Total params: {fine_tuned_vgg16.count_params():,}")
    print(f"Trainable params: {sum([tf.keras.backend.count_params(w) for w in fine_tuned_vgg16.trainable_weights]):,}")
    print(f"Non-trainable params: {sum([tf.keras.backend.count_params(w) for w in fine_tuned_vgg16.non_trainable_weights]):,}")
else:
    print(f"\n📊 Loaded Model Info:")
    print(f"Total params: {fine_tuned_vgg16.count_params():,}")
    print(f"Trainable params: {sum([tf.keras.backend.count_params(w) for w in fine_tuned_vgg16.trainable_weights]):,}")
    print(f"Non-trainable params: {sum([tf.keras.backend.count_params(w) for w in fine_tuned_vgg16.non_trainable_weights]):,}")

# Setup callbacks (only if training)
if not skip_training:
    callbacks = [
        CSVLogger(os.path.join(models_dir, 'fine_tuned_vgg16_training.log'), separator=','),
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(models_dir, 'best_fine_tuned_vgg16.h5'),
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

# Phase 1: Train with frozen base (only if not skipping training)
if not skip_training:
    # Check if this is continue training or fresh training
    if CONTINUE_TRAINING and os.path.exists(model_path):
        print("\n" + "="*50)
        print("PHASE 3: CONTINUE TRAINING FROM EXISTING MODEL")
        print("="*50)
        
        # Calculate steps per epoch based on actual data
        steps_per_epoch = train_generator.samples // batch_size
        validation_steps = val_generator.samples // batch_size
        
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Validation steps: {validation_steps}")
        print(f"Additional epochs: {ADDITIONAL_EPOCHS}")
        
        # Use a very low learning rate for continued training
        fine_tuned_vgg16.compile(
            optimizer=Adam(learning_rate=CONTINUE_LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\n📊 Current Model State:")
        print(f"Total params: {fine_tuned_vgg16.count_params():,}")
        print(f"Trainable params: {sum([tf.keras.backend.count_params(w) for w in fine_tuned_vgg16.trainable_weights]):,}")
        print(f"Non-trainable params: {sum([tf.keras.backend.count_params(w) for w in fine_tuned_vgg16.non_trainable_weights]):,}")
        
        # Continue training with additional epochs
        history_continue = fine_tuned_vgg16.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=ADDITIONAL_EPOCHS,
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        # Use the continue training history for plotting
        combined_history = history_continue.history
        
    else:
        # Original training phases for fresh training
        print("\n" + "="*50)
        print("PHASE 1: Training with frozen VGG16 base")
        print("="*50)

        # Calculate steps per epoch based on actual data
        steps_per_epoch = train_generator.samples // batch_size
        validation_steps = val_generator.samples // batch_size
        epochs_phase1 = 20

        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Validation steps: {validation_steps}")

        history_phase1 = fine_tuned_vgg16.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs_phase1,
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )

        # Phase 2: Fine-tune top layers
        print("\n" + "="*50)
        print("PHASE 2: Fine-tuning top VGG16 layers")
        print("="*50)

        # Unfreeze top layers of VGG16
        vgg16_base.trainable = True

        # Fine-tune from this layer onwards
        fine_tune_at = 15

        # Freeze all the layers before the `fine_tune_at` layer
        for layer in vgg16_base.layers[:fine_tune_at]:
            layer.trainable = False

        # Use a lower learning rate for fine-tuning
        fine_tuned_vgg16.compile(
            optimizer=Adam(learning_rate=FINE_TUNE_LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        print(f"\nPhase 2 - Fine-tuning enabled:")
        print(f"Fine-tuning from layer {fine_tune_at} onwards")
        print(f"Total params: {fine_tuned_vgg16.count_params():,}")
        print(f"Trainable params: {sum([tf.keras.backend.count_params(w) for w in fine_tuned_vgg16.trainable_weights]):,}")
        print(f"Non-trainable params: {sum([tf.keras.backend.count_params(w) for w in fine_tuned_vgg16.non_trainable_weights]):,}")
        print(f"Number of trainable variables: {len(fine_tuned_vgg16.trainable_variables)}")

        # Show which layers are trainable
        print(f"\nTrainable layers in VGG16 base:")
        for i, layer in enumerate(vgg16_base.layers):
            if layer.trainable:
                print(f"  Layer {i}: {layer.name} - Trainable")

        # Continue training
        epochs_phase2 = 30
        total_epochs = epochs_phase1 + epochs_phase2

        history_phase2 = fine_tuned_vgg16.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs_phase2,
            initial_epoch=history_phase1.epoch[-1],
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )

        # Combine training histories
        def combine_histories(hist1, hist2):
            combined_history = {}
            for key in hist1.history.keys():
                combined_history[key] = hist1.history[key] + hist2.history[key]
            return combined_history

        combined_history = combine_histories(history_phase1, history_phase2)

    # Plot training results
    def plot_training_history(history, title_suffix=""):
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
        history_plot_path = os.path.join(models_dir, f'fine_tuned_vgg16_training_history{title_suffix.lower().replace(" ", "_")}.png')
        plt.savefig(history_plot_path, dpi=300, bbox_inches='tight')
        plt.show()

    print("\n" + "="*50)
    print("TRAINING RESULTS")
    print("="*50)

    plot_training_history(combined_history)
else:
    print("\n" + "="*50)
    print("SKIPPING TRAINING - USING LOADED MODEL")
    print("="*50)
    # Calculate steps for evaluation
    steps_per_epoch = train_generator.samples // batch_size
    validation_steps = val_generator.samples // batch_size

# Evaluate on test set
print("\n" + "="*50)
print("EVALUATING ON TEST SET")
print("="*50)

# For evaluation, we'll use the validation generator as test data
test_loss, test_accuracy = fine_tuned_vgg16.evaluate(val_generator, steps=validation_steps, verbose=1)
print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Generate predictions for confusion matrix
print("\nGenerating predictions for confusion matrix...")

# Reset the generator
val_generator.reset()

# Get predictions
test_predictions = fine_tuned_vgg16.predict(val_generator, steps=validation_steps, verbose=1)
test_pred_classes = np.argmax(test_predictions, axis=1)

# Get true labels from generator
val_generator.reset()
test_labels = val_generator.classes

# Ensure we have the right number of predictions
min_len = min(len(test_pred_classes), len(test_labels))
test_pred_classes = test_pred_classes[:min_len]
test_labels = test_labels[:min_len]

print(f"Number of test samples: {len(test_labels)}")
print(f"Number of predictions: {len(test_pred_classes)}")

# Generate confusion matrix
cm = confusion_matrix(test_labels, test_pred_classes)
cm_normalized = confusion_matrix(test_labels, test_pred_classes, normalize='true')

# Plot confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Non-normalized confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names, ax=axes[0])
axes[0].set_title('Confusion Matrix (Raw Counts)', fontsize=16)
axes[0].set_xlabel('Predicted Label', fontsize=14)
axes[0].set_ylabel('True Label', fontsize=14)

# Normalized confusion matrix
sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names, ax=axes[1])
axes[1].set_title('Normalized Confusion Matrix', fontsize=16)
axes[1].set_xlabel('Predicted Label', fontsize=14)
axes[1].set_ylabel('True Label', fontsize=14)

plt.tight_layout()
confusion_matrix_path = os.path.join(models_dir, 'fine_tuned_vgg16_confusion_matrices.png')
plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
plt.show()

# Classification report
print("\n" + "="*50)
print("CLASSIFICATION REPORT")
print("="*50)
print(classification_report(test_labels, test_pred_classes, 
                          target_names=class_names, digits=4))

# Calculate per-class accuracy
per_class_accuracy = cm_normalized.diagonal()
print("\n" + "="*50)
print("PER-CLASS ACCURACY")
print("="*50)
for i, class_name in enumerate(class_names):
    print(f"{class_name:12}: {per_class_accuracy[i]:.4f} ({per_class_accuracy[i]*100:.2f}%)")

# Save the final model (only if training was performed)
if not skip_training:
    print("\n" + "="*50)
    print("SAVING MODEL")
    print("="*50)

    # Save in different formats
    model_name = "fine_tuned_vgg16_final"

    # Save as .keras file (recommended native format)
    keras_path = os.path.join(models_dir, f"{model_name}.keras")
    fine_tuned_vgg16.save(keras_path)
    print(f"Model saved as {keras_path}")

    # Save in SavedModel format for deployment
    savedmodel_path = os.path.join(models_dir, f"{model_name}_savedmodel")
    try:
        fine_tuned_vgg16.export(savedmodel_path)
        print(f"Model exported as {savedmodel_path} (SavedModel format)")
    except AttributeError:
        # Fallback for older TensorFlow versions
        fine_tuned_vgg16.save(savedmodel_path, save_format='tf')
        print(f"Model saved as {savedmodel_path} (SavedModel format)")

    # Save weights only
    weights_path = os.path.join(models_dir, f"{model_name}.weights.h5")
    fine_tuned_vgg16.save_weights(weights_path)
    print(f"Model weights saved as {weights_path}")

    # Save training history
    import pickle
    history_path = os.path.join(models_dir, f"{model_name}_history.pkl")
    with open(history_path, 'wb') as f:
        pickle.dump(combined_history, f)
    print(f"Training history saved as {history_path}")
else:
    print("\n" + "="*50)
    print("MODEL ALREADY EXISTS - SKIPPING SAVE")
    print("="*50)
    model_name = "fine_tuned_vgg16_final"

print("\n" + "="*50)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("="*50)
print(f"Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Model saved as: {model_name}")
print("Files generated:")
print(f"  - {model_name}.keras (native Keras format - recommended)")
print(f"  - {model_name}.h5 (legacy HDF5 format)")
print(f"  - {model_name}_savedmodel/ (SavedModel format for deployment)")
print(f"  - {model_name}_weights.h5 (weights only)")
print(f"  - {model_name}_history.pkl (training history)")
print("  - fine_tuned_vgg16_training_history.png")
print("  - fine_tuned_vgg16_confusion_matrices.png")
print("  - fine_tuned_vgg16_training.log")

# OPTIONAL: Different training strategies (comment/uncomment as needed)

# Strategy 1: Train entire VGG16 from scratch (NOT recommended for small datasets)
# vgg16_base.trainable = True
# fine_tuned_vgg16.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Strategy 2: Fine-tune more layers from the beginning
# fine_tune_at = 10  # Instead of 15, unfreeze more layers
# for layer in vgg16_base.layers[:fine_tune_at]:
#     layer.trainable = False

# Strategy 3: Different fine-tuning point (current is 15)
# You can experiment with different values: 10, 12, 17, etc.

if __name__ == "__main__":
    # This ensures the script runs when executed directly
    print("Fine-Tuned VGG16 Training Script")
    print("Make sure your data is in the following structure:")
    print("output/")
    print("  train/")
    print("    complex/")
    print("    flat/")
    print("    gable/")
    print("    halfhip/")
    print("    hip/")
    print("    L-shaped/")
    print("    pyramid/")
    print("  val/")
    print("    complex/")
    print("    flat/")
    print("    gable/")
    print("    halfhip/")
    print("    hip/")
    print("    L-shaped/")
    print("    pyramid/")
    print("\nStarting training process...")