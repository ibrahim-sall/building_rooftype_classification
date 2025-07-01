"""
Building Roof Type Classification - Inference Script
===================================================

This script performs inference (testing only) on a directory containing full orthophotos
using the trained Fine-Tuned VGG16 model for building roof type classification.

Features:
1. **Flexible Input**: Process all images in a specified directory
2. **Multiple Output Formats**: 
   - Console output with predictions and confidence scores
   - CSV file with detailed results
   - Optional visualization of predictions
3. **Batch Processing**: Efficiently process large numbers of images
4. **Model Auto-Detection**: Automatically finds and loads the best available model
5. **Image Preprocessing**: Handles various image formats and sizes
6. **Confidence Filtering**: Option to filter predictions by confidence threshold

Usage Examples:
    # Basic inference on a directory
    python inference.py --input_dir /path/to/orthophotos
    
    # With confidence threshold and CSV output
    python inference.py --input_dir /path/to/orthophotos --confidence_threshold 0.8 --output_csv results.csv
    
    # With visualization
    python inference.py --input_dir /path/to/orthophotos --visualize --output_dir results/
    
    # Specify custom model path
    python inference.py --input_dir /path/to/orthophotos --model_path my_model.keras

Supported image formats: .jpg, .jpeg, .png, .bmp, .tiff, .tif
"""

import os
import sys
import argparse
import logging

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from utils.loader import load_trained_model

# Configuration
IMG_HEIGHT = 140
IMG_WIDTH = 140
CLASS_NAMES = ['complex', 'flat', 'gable', 'halfhip', 'hip', 'L-shaped', 'pyramid']
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.jp2'}


def get_image_files(directory):
    """Get all supported image files from directory."""
    image_files = []
    
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    for filename in os.listdir(directory):
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in SUPPORTED_FORMATS:
            image_files.append(os.path.join(directory, filename))
    
    if not image_files:
        raise ValueError(f"No supported image files found in {directory}")
    
    return sorted(image_files)

def preprocess_image(image_path, logger=None):
    """Preprocess a single image for prediction."""
    if logger is None:
        logger = logging.getLogger(__name__)
    try:
        # Load and resize image
        img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize to [0,1]
        return img_array, True
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        return None, False

def predict_single_image(model, image_path, confidence_threshold=0.0, logger=None):
    """Make prediction for a single image."""
    img_array, success = preprocess_image(image_path, logger)
    
    if not success:
        return None
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])
    predicted_class = CLASS_NAMES[predicted_class_idx]
    
    # Get all class probabilities
    class_probabilities = {CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))}
    
    result = {
        'image_path': image_path,
        'image_name': os.path.basename(image_path),
        'predicted_class': predicted_class,
        'confidence': confidence,
        'class_probabilities': class_probabilities,
        'meets_threshold': confidence >= confidence_threshold
    }
    
    return result

def save_results_to_csv(results, output_path, logger=None):
    """Save prediction results to CSV file."""
    if logger is None:
        logger = logging.getLogger(__name__)
    csv_data = []
    for result in results:
        if result is None:
            continue
            
        row = {
            'image_name': result['image_name'],
            'image_path': result['image_path'],
            'predicted_class': result['predicted_class'],
            'confidence': result['confidence'],
            'meets_threshold': result['meets_threshold']
        }
        
        # Add individual class probabilities
        for class_name in CLASS_NAMES:
            row[f'prob_{class_name}'] = result['class_probabilities'][class_name]
        
        csv_data.append(row)
    
    df = pd.DataFrame(csv_data)
    df.to_csv(output_path, index=False)
    logger.info(f"Results saved to CSV: {output_path}")
    return 

def visualize_predictions(results, output_dir, max_images=20, logger=None):
    """Create visualization of predictions."""
    if logger is None:
        logger = logging.getLogger(__name__)
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Filter successful predictions
    valid_results = [r for r in results if r is not None]
    
    valid_results.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Create visualization for top predictions
    n_images = min(len(valid_results), max_images)
    cols = 4
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(n_images):
        row = idx // cols
        col = idx % cols
        result = valid_results[idx]
    
        try:
            img = Image.open(result['image_path'])
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
            
            title = f"{result['image_name']}\n{result['predicted_class']} ({result['confidence']:.3f})"
            axes[row, col].set_title(title, fontsize=10)
            
        except Exception as e:
            axes[row, col].text(0.5, 0.5, f"Error loading\n{result['image_name']}", 
                              ha='center', va='center', transform=axes[row, col].transAxes)
            axes[row, col].axis('off')
    
    # Hide unused subplots
    for idx in range(n_images, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    viz_path = os.path.join(output_dir, 'predictions_visualization.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualization saved to: {viz_path}")

def print_summary_statistics(results):
    """Print summary statistics of predictions."""
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        print("No valid predictions to summarize")
        return
    
    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    
    # Overall statistics
    total_images = len(results)
    successful_predictions = len(valid_results)
    failed_predictions = total_images - successful_predictions
    
    print(f"Total images processed: {total_images}")
    print(f"Successful predictions: {successful_predictions}")
    print(f"Failed predictions: {failed_predictions}")
    
    if successful_predictions == 0:
        return
    
    # Class distribution
    class_counts = {}
    confidence_scores = []
    
    for result in valid_results:
        predicted_class = result['predicted_class']
        confidence = result['confidence']
        
        class_counts[predicted_class] = class_counts.get(predicted_class, 0) + 1
        confidence_scores.append(confidence)
    
    print(f"\nClass Distribution:")
    for class_name in CLASS_NAMES:
        count = class_counts.get(class_name, 0)
        percentage = (count / successful_predictions) * 100
        print(f"  {class_name:12}: {count:4d} ({percentage:5.1f}%)")
    
    # Confidence statistics
    confidence_scores = np.array(confidence_scores)
    print(f"\nConfidence Statistics:")
    print(f"  Mean confidence: {np.mean(confidence_scores):.3f}")
    print(f"  Std deviation:   {np.std(confidence_scores):.3f}")
    print(f"  Min confidence:  {np.min(confidence_scores):.3f}")
    print(f"  Max confidence:  {np.max(confidence_scores):.3f}")
    print(f"  Median:          {np.median(confidence_scores):.3f}")

def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(
        description="Building Roof Type Classification - Inference Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py --input_dir /path/to/orthophotos
  python inference.py --input_dir /path/to/orthophotos --confidence_threshold 0.8
  python inference.py --input_dir /path/to/orthophotos --output_csv results.csv
  python inference.py --input_dir /path/to/orthophotos --visualize --output_dir results/
        """
    )
    
    parser.add_argument('--input_dir', required=True,
                       help='Directory containing orthophotos to classify')
    parser.add_argument('--model_path', default=None,
                       help='Path to trained model (auto-detect if not specified)')
    parser.add_argument('--confidence_threshold', type=float, default=0.0,
                       help='Minimum confidence threshold for predictions (default: 0.0)')
    parser.add_argument('--output_csv', default=None,
                       help='Output CSV file path for detailed results')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization of predictions')
    parser.add_argument('--output_dir', default='inference_results',
                       help='Output directory for visualizations (default: inference_results)')
    parser.add_argument('--max_viz_images', type=int, default=20,
                       help='Maximum number of images to show in visualization (default: 20)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for processing (default: 32)')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("BUILDING ROOF TYPE CLASSIFICATION - INFERENCE")
    logger.info("="*60)
    
    # Load model
    model = load_trained_model(args.model_path)
    
    # Get image files
    logger.info(f"\nScanning directory: {args.input_dir}")
    image_files = get_image_files(args.input_dir)
    logger.info(f"Found {len(image_files)} image files")
    
    # Process images
    logger.info(f"\nProcessing images...")
    results = []
    successful_count = 0
    
    for i, image_path in enumerate(image_files, 1):
        logger.info(f"Processing {i}/{len(image_files)}: {os.path.basename(image_path)}")
        
        result = predict_single_image(model, image_path, args.confidence_threshold, logger)
        results.append(result)
        
        if result is not None:
            successful_count += 1
            confidence_emoji = "🟢" if result['confidence'] > 0.8 else "🟡" if result['confidence'] > 0.5 else "🔴"
            threshold_emoji = "✅" if result['meets_threshold'] else "❌"
            
            logger.info(f" → {result['predicted_class']} ({result['confidence']:.3f}) {confidence_emoji} {threshold_emoji}")
        else:
            logger.error(" → ❌ Failed")
    
    print_summary_statistics(results)
    
    if args.output_csv:
        save_results_to_csv(results, args.output_csv)
    
    if args.visualize:
        logger.info(f"\nCreating visualization...")
        visualize_predictions(results, args.output_dir, args.max_viz_images, logger)
    
    # Print final summary
    logger.info(f"\n" + "="*60)
    logger.info("INFERENCE COMPLETED")
    logger.info("="*60)
    logger.info(f"Total images: {len(image_files)}")
    logger.info(f"Successful predictions: {successful_count}")
    logger.info(f"Success rate: {(successful_count/len(image_files)*100):.1f}%")
    
    if args.confidence_threshold > 0:
        above_threshold = sum(1 for r in results if r and r['meets_threshold'])
        logger.info(f"Predictions above threshold ({args.confidence_threshold}): {above_threshold}")
    
    logger.info("\nFiles generated:")
    if args.output_csv:
        logger.info(f"  - {args.output_csv} (detailed results)")
    if args.visualize:
        logger.info(f"  - {args.output_dir}/predictions_visualization.png")

if __name__ == "__main__":
    main()
