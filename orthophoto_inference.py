"""
Building Roof Type Classification - Footprint-Based Inference
============================================================

This script performs roof type classification on existing building footprint shapefiles
by extracting roof areas from orthophotos and adding classification labels to the footprints.
Optionally, it can also extract height statistics from Digital Surface Models (DSM).

**WORKFLOW**:
1. Read building footprint shapefiles from 'footprints' directory
2. For each footprint, extract the corresponding roof area from orthophoto
3. Classify the roof type using the trained model
4. Optionally, extract height statistics from DSM if provided
5. Add classification labels, confidence scores, and height data to the footprint attributes
6. Save the labeled footprints as new shapefiles

**MULTI-CHANNEL SUPPORT**: This script handles various orthophoto formats including:
- RGB (3-channel)
- RGBA (4-channel with alpha/transparency)  
- RGBI (4-channel RGB + Infrared)
- Multi-spectral images (>4 channels - uses first 3)
- Grayscale (converted to RGB)

**DSM SUPPORT**: When DSM files are provided, the script extracts:
- Mean height of the building footprint
- Minimum and maximum heights
- Standard deviation of heights
- Number of valid height pixels

Features:
1. **Footprint-Based Classification**: Uses existing building footprints as input
2. **Multi-Channel Image Support**: Automatically handles 3, 4, or more channel images
3. **Geographic Coordinate Support**: Works with georeferenced orthophotos and footprints
4. **Height Extraction**: Optional DSM processing for building height statistics
5. **Confidence Scoring**: Provides classification confidence for each footprint
6. **Batch Processing**: Process multiple orthophotos and their corresponding footprints
7. **Labeled Output**: Creates new shapefiles with roof type classifications and heights

Directory Structure Expected:
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

Usage Examples:
    # Basic footprint classification
    python orthophoto_inference.py --input_dir test_orthophotos/
    
    # With DSM for height extraction
    python orthophoto_inference.py --input_dir test_orthophotos/ --dsm_dir test_dsm/
    
    # With custom confidence threshold
    python orthophoto_inference.py --input_dir test_orthophotos/ --confidence_threshold 0.8
    
    # With visualizations showing classified footprints
    python orthophoto_inference.py --input_dir test_orthophotos/ --visualize
    
    # Complete output with all options
    python orthophoto_inference.py --input_dir test_orthophotos/ --dsm_dir test_dsm/ --visualize --output_csv results.csv

Supported Image Formats:
    - Standard: .jpg, .jpeg, .png, .bmp, .tiff, .tif, .webp
    - Specialized: .jp2 (JPEG 2000), multi-channel TIFF
    - Channels: RGB, RGBA, RGBI, grayscale, multi-spectral

Requirements:
    pip install opencv-python geopandas shapely fiona rasterio

Note: This approach requires existing building footprint shapefiles and corresponding
georeferenced orthophotos. The footprints and orthophotos must be in the same
coordinate reference system for accurate roof extraction. DSM files are optional
but must also be in the same coordinate system if provided.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cv2
from skimage.util import view_as_windows
import warnings

# GIS libraries for shapefile export
try:
    import geopandas as gpd
    from shapely.geometry import Polygon, box
    import rasterio
    from rasterio.windows import from_bounds, Window
    from rasterio.transform import from_bounds as transform_from_bounds
    GIS_AVAILABLE = True
except ImportError:
    GIS_AVAILABLE = False
    print("⚠️  GIS libraries not available. Install with: pip install geopandas shapely fiona rasterio")

warnings.filterwarnings('ignore')

# Configuration
IMG_HEIGHT = 140
IMG_WIDTH = 140
CLASS_NAMES = ['complex', 'flat', 'gable', 'halfhip', 'hip', 'L-shaped', 'pyramid']
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.jp2', '.webp', '.gif'}

# Color map for different roof types (BGR format for OpenCV)
CLASS_COLORS = {
    'complex': (0, 255, 255),    # Yellow
    'flat': (255, 0, 0),         # Blue
    'gable': (0, 255, 0),        # Green
    'halfhip': (255, 165, 0),    # Orange
    'hip': (128, 0, 128),        # Purple
    'L-shaped': (255, 192, 203), # Pink
    'pyramid': (255, 0, 255)     # Magenta
}

def find_best_model():
    """Automatically find the best available model in the models directory."""
    models_dir = "models"
    model_candidates = [
        "fine_tuned_vgg16_final.keras",
        "best_fine_tuned_vgg16.keras",
        "fine_tuned_vgg16_final.h5",
        "best_fine_tuned_vgg16.h5",
        "fine_tuned_vgg16_final.wheight.h5",
        "fine_tuned_vgg16_final_savedmodel"
    ]
    
    # Check if models directory exists
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory '{models_dir}' not found!")
    
    for model_name in model_candidates:
        model_path = os.path.join(models_dir, model_name)
        if os.path.exists(model_path):
            return model_path
    
    raise FileNotFoundError(
        f"No trained model found in '{models_dir}' directory! Please ensure you have one of the following files:\n" +
        "\n".join([f"  - {models_dir}/{path}" for path in model_candidates])
    )

def load_trained_model(model_path=None):
    """Load the trained model."""
    if model_path is None:
        model_path = find_best_model()
    
    print(f"Loading model from: {model_path}")
    
    try:
        model = load_model(model_path)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def get_orthophoto_footprint_dsm_triplets(input_dir, dsm_dir=None):
    """
    Get triplets of orthophotos, their corresponding footprint shapefiles, and DSM files.
    
    Args:
        input_dir: Directory containing orthophotos and footprints subdirectory
        dsm_dir: Optional directory containing DSM files (if None, looks for DSM files in input_dir)
    
    Returns:
        List of tuples: (orthophoto_path, footprint_path, dsm_path or None)
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Directory not found: {input_dir}")
    
    footprints_dir = os.path.join(input_dir, 'footprints')
    if not os.path.exists(footprints_dir):
        raise FileNotFoundError(f"Footprints directory not found: {footprints_dir}")
    
    # Set DSM directory
    if dsm_dir is None:
        dsm_dir = input_dir
    elif not os.path.exists(dsm_dir):
        print(f"⚠️  DSM directory not found: {dsm_dir}, skipping DSM processing")
        dsm_dir = None
    
    triplets = []
    
    # Get all orthophoto files
    for filename in os.listdir(input_dir):
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in SUPPORTED_FORMATS:
            orthophoto_path = os.path.join(input_dir, filename)
            base_name = os.path.splitext(filename)[0]
            
            # Look for corresponding footprint shapefile
            possible_footprint_names = [
                f"{base_name}_footprints.shp",
                f"{base_name}.shp", 
                f"{base_name}_buildings.shp",
                f"{base_name}_emprise.shp"
            ]
            
            footprint_path = None
            for footprint_name in possible_footprint_names:
                potential_path = os.path.join(footprints_dir, footprint_name)
                if os.path.exists(potential_path):
                    footprint_path = potential_path
                    break
            
            # Look for corresponding DSM file if DSM directory is available
            dsm_path = None
            if dsm_dir:
                possible_dsm_names = [
                    f"{base_name}_dsm.tif",
                    f"{base_name}_DSM.tif",
                    f"{base_name}.tif",
                    f"{base_name}_height.tif",
                    f"{base_name}_heights.tif"
                ]
                
                for dsm_name in possible_dsm_names:
                    potential_dsm_path = os.path.join(dsm_dir, dsm_name)
                    if os.path.exists(potential_dsm_path):
                        dsm_path = potential_dsm_path
                        break
            
            if footprint_path:
                triplets.append((orthophoto_path, footprint_path, dsm_path))
                dsm_status = f" + DSM: {os.path.basename(dsm_path)}" if dsm_path else " (no DSM)"
                print(f"✅ Found: {filename} -> {os.path.basename(footprint_path)}{dsm_status}")
            else:
                print(f"⚠️  No footprint shapefile found for {filename}")
    
    if not triplets:
        raise ValueError(f"No orthophoto-footprint pairs found in {input_dir}")
    
    return triplets

def get_orthophoto_footprint_pairs(input_dir):
    """
    Get pairs of orthophotos and their corresponding footprint shapefiles.
    
    Args:
        input_dir: Directory containing orthophotos and footprints subdirectory
    
    Returns:
        List of tuples: (orthophoto_path, footprint_path)
    """
    # Use the new triplet function but only return pairs (ignore DSM)
    triplets = get_orthophoto_footprint_dsm_triplets(input_dir, dsm_dir=None)
    return [(ortho, footprint) for ortho, footprint, _ in triplets]

def extract_footprint_height(dsm_path, footprint_geometry):
    """
    Extract height statistics from DSM for a building footprint.
    
    Args:
        dsm_path: Path to the Digital Surface Model (DSM)
        footprint_geometry: Shapely geometry of the building footprint
    
    Returns:
        Dictionary with height statistics, or None if extraction fails
    """
    if not GIS_AVAILABLE:
        return None
    
    try:
        with rasterio.open(dsm_path) as dsm_src:
            # Get the bounds of the footprint
            minx, miny, maxx, maxy = footprint_geometry.bounds
            
            # Convert geographic coordinates to pixel coordinates
            transform = dsm_src.transform
            px1, py1 = ~transform * (minx, maxy)  # top-left
            px2, py2 = ~transform * (maxx, miny)  # bottom-right
            
            # Ensure proper order and bounds
            row_start = max(0, int(min(py1, py2)))
            row_end = min(dsm_src.height, int(max(py1, py2)) + 1)
            col_start = max(0, int(min(px1, px2)))
            col_end = min(dsm_src.width, int(max(px1, px2)) + 1)
            
            # Check if the area is valid
            if row_end <= row_start or col_end <= col_start:
                return None
            
            # Read the DSM data
            dsm_window = dsm_src.read(1, window=((row_start, row_end), (col_start, col_end)))
            
            # Remove NoData values
            if dsm_src.nodata is not None:
                valid_mask = dsm_window != dsm_src.nodata
                if not np.any(valid_mask):
                    return None
                dsm_values = dsm_window[valid_mask]
            else:
                # Filter out common NoData values and extreme outliers
                valid_mask = (dsm_window != -9999) & (dsm_window != -32768) & (~np.isnan(dsm_window))
                if not np.any(valid_mask):
                    return None
                dsm_values = dsm_window[valid_mask]
            
            # Calculate statistics
            if len(dsm_values) > 0:
                return {
                    'mean_height': float(np.mean(dsm_values)),
                    'min_height': float(np.min(dsm_values)),
                    'max_height': float(np.max(dsm_values)),
                    'std_height': float(np.std(dsm_values)),
                    'pixel_count': len(dsm_values)
                }
            else:
                return None
                
    except Exception as e:
        return None

def extract_footprint_image(orthophoto_path, footprint_geometry, buffer_pixels=10):
    """
    Extract image area corresponding to a building footprint from an orthophoto.
    
    Args:
        orthophoto_path: Path to the orthophoto
        footprint_geometry: Shapely geometry of the building footprint
        buffer_pixels: Buffer around footprint in pixels
    
    Returns:
        PIL Image of the extracted footprint area, or None if extraction fails
    """
    if not GIS_AVAILABLE:
        print("❌ GIS libraries required for footprint extraction")
        return None
    
    try:
        with rasterio.open(orthophoto_path) as src:
            # Get the bounds of the footprint
            minx, miny, maxx, maxy = footprint_geometry.bounds
            
            # Convert geographic coordinates to pixel coordinates
            transform = src.transform
            px1, py1 = ~transform * (minx, maxy)  # top-left
            px2, py2 = ~transform * (maxx, miny)  # bottom-right
            
            # Ensure proper order and add buffer
            row_start = max(0, int(min(py1, py2)) - buffer_pixels)
            row_end = min(src.height, int(max(py1, py2)) + buffer_pixels)
            col_start = max(0, int(min(px1, px2)) - buffer_pixels)
            col_end = min(src.width, int(max(px1, px2)) + buffer_pixels)
            
            # Check if the area is valid
            if row_end <= row_start or col_end <= col_start:
                print(f"⚠️  Invalid footprint bounds")
                return None
            
            # Read the image data using slicing instead of Window
            image_data = src.read(window=((row_start, row_end), (col_start, col_end)))
            
            # Handle different number of channels
            if image_data.shape[0] == 1:
                # Grayscale - convert to RGB
                image_array = np.stack([image_data[0]] * 3, axis=0)
            elif image_data.shape[0] >= 3:
                # Take first 3 channels (RGB)
                image_array = image_data[:3]
            else:
                print(f"⚠️  Unsupported number of channels: {image_data.shape[0]}")
                return None
            
            # Convert to HWC format and ensure uint8
            image_array = np.transpose(image_array, (1, 2, 0))
            if image_array.dtype != np.uint8:
                if image_array.max() <= 1.0:
                    image_array = (image_array * 255).astype(np.uint8)
                else:
                    image_array = image_array.astype(np.uint8)
            
            # Convert to PIL Image
            return Image.fromarray(image_array)
            
    except Exception as e:
        print(f"❌ Error extracting footprint from {os.path.basename(orthophoto_path)}: {e}")
        return None

def classify_footprint_image(model, footprint_image):
    """
    Classify a single footprint image using the trained model.
    
    Args:
        model: Trained roof classification model
        footprint_image: PIL Image of the footprint area
    
    Returns:
        Dictionary with classification results
    """
    try:
        # Resize to model input size
        resized_image = footprint_image.resize((IMG_WIDTH, IMG_HEIGHT), Image.LANCZOS)
        
        # Convert to array and normalize
        image_array = img_to_array(resized_image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = image_array / 255.0
        
        # Make prediction
        predictions = model.predict(image_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        
        # Get all class probabilities
        class_probabilities = {CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))}
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'class_probabilities': class_probabilities
        }
        
    except Exception as e:
        print(f"❌ Error classifying footprint: {e}")
        return None

def process_footprints(model, orthophoto_path, footprint_path, confidence_threshold=0.5, dsm_path=None):
    """
    Process building footprints by classifying roof types from orthophoto.
    
    Args:
        model: Trained roof classification model
        orthophoto_path: Path to orthophoto image
        footprint_path: Path to footprint shapefile
        confidence_threshold: Minimum confidence for valid classifications
        dsm_path: Optional path to Digital Surface Model for height extraction
    
    Returns:
        GeoDataFrame with classified footprints
    """
    if not GIS_AVAILABLE:
        print("❌ GIS libraries required for footprint processing")
        return None
    
    print(f"Processing footprints from: {os.path.basename(footprint_path)}")
    print(f"Using orthophoto: {os.path.basename(orthophoto_path)}")
    if dsm_path:
        print(f"Using DSM: {os.path.basename(dsm_path)}")
    
    try:
        # Load footprints shapefile
        footprints_gdf = gpd.read_file(footprint_path)
        print(f"  Loaded {len(footprints_gdf)} footprints")
        
        # Initialize classification columns
        footprints_gdf['roof_class'] = 'unknown'
        footprints_gdf['confidence'] = 0.0
        footprints_gdf['classified'] = False
        
        # Add probability columns for each class
        for class_name in CLASS_NAMES:
            footprints_gdf[f'prob_{class_name[:8]}'] = 0.0
        
        # Add height columns if DSM is provided
        if dsm_path:
            footprints_gdf['mean_height'] = np.nan
            footprints_gdf['min_height'] = np.nan
            footprints_gdf['max_height'] = np.nan
            footprints_gdf['std_height'] = np.nan
            footprints_gdf['height_px'] = 0  # Number of valid height pixels
        
        classified_count = 0
        height_extracted_count = 0
        
        # Process each footprint
        for idx, footprint in footprints_gdf.iterrows():
            try:
                # Extract height from DSM if provided
                if dsm_path:
                    height_stats = extract_footprint_height(dsm_path, footprint.geometry)
                    if height_stats:
                        footprints_gdf.at[idx, 'mean_height'] = height_stats['mean_height']
                        footprints_gdf.at[idx, 'min_height'] = height_stats['min_height']
                        footprints_gdf.at[idx, 'max_height'] = height_stats['max_height']
                        footprints_gdf.at[idx, 'std_height'] = height_stats['std_height']
                        footprints_gdf.at[idx, 'height_px'] = height_stats['pixel_count']
                        height_extracted_count += 1
                
                # Extract footprint area from orthophoto
                footprint_image = extract_footprint_image(
                    orthophoto_path, 
                    footprint.geometry,
                    buffer_pixels=5
                )
                
                if footprint_image is None:
                    continue
                
                # Classify the footprint
                classification = classify_footprint_image(model, footprint_image)
                
                if classification is None:
                    continue
                
                # Check confidence threshold
                if classification['confidence'] >= confidence_threshold:
                    footprints_gdf.at[idx, 'roof_class'] = classification['predicted_class']
                    footprints_gdf.at[idx, 'confidence'] = classification['confidence']
                    footprints_gdf.at[idx, 'classified'] = True
                    
                    # Add class probabilities
                    for class_name in CLASS_NAMES:
                        footprints_gdf.at[idx, f'prob_{class_name[:8]}'] = classification['class_probabilities'][class_name]
                    
                    classified_count += 1
                    
                    if classified_count % 10 == 0:
                        print(f"    Classified {classified_count} footprints...")
                
            except Exception as e:
                print(f"    ⚠️  Error processing footprint {idx}: {e}")
                continue
        
        print(f"  Successfully classified {classified_count}/{len(footprints_gdf)} footprints")
        if dsm_path:
            print(f"  Successfully extracted height for {height_extracted_count}/{len(footprints_gdf)} footprints")
        
        return footprints_gdf
        
    except Exception as e:
        print(f"❌ Error processing footprints: {e}")
        return None

def save_classified_footprints(footprints_gdf, output_path):
    """Save classified footprints to shapefile."""
    if footprints_gdf is None or len(footprints_gdf) == 0:
        print("⚠️  No footprints to save")
        return
    
    try:
        # Save to shapefile
        footprints_gdf.to_file(output_path, driver='ESRI Shapefile')
        print(f"🗺️  Classified footprints saved: {output_path}")
        
        # Also save as GeoJSON
        geojson_path = output_path.replace('.shp', '.geojson')
        footprints_gdf.to_file(geojson_path, driver='GeoJSON')
        print(f"🗺️  GeoJSON saved: {geojson_path}")
        
        # Print classification summary
        classified_footprints = footprints_gdf[footprints_gdf['classified'] == True]
        if len(classified_footprints) > 0:
            print(f"📊 Classification summary:")
            class_counts = classified_footprints['roof_class'].value_counts()
            for class_name, count in class_counts.items():
                percentage = (count / len(classified_footprints)) * 100
                print(f"    {class_name}: {count} ({percentage:.1f}%)")
        
    except Exception as e:
        print(f"❌ Error saving classified footprints: {e}")

def predict_windows(model, windows, batch_size=32):
    """
    Predict roof types for extracted windows in batches.
    
    Args:
        model: Trained model
        windows: List of (window_array, x, y, window_size) tuples
        batch_size: Batch size for prediction
    
    Returns:
        List of detection dictionaries
    """
    detections = []
    
    # Process in batches
    for i in range(0, len(windows), batch_size):
        batch_windows = windows[i:i+batch_size]
        batch_arrays = np.array([w[0] for w in batch_windows])
        
        # Predict batch
        predictions = model.predict(batch_arrays, verbose=0)
        
        # Process predictions
        for j, (window_array, x, y, window_size) in enumerate(batch_windows):
            pred_probs = predictions[j]
            predicted_class_idx = np.argmax(pred_probs)
            confidence = float(pred_probs[predicted_class_idx])
            predicted_class = CLASS_NAMES[predicted_class_idx]
            
            detection = {
                'x': x,
                'y': y,
                'width': window_size,
                'height': window_size,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'class_probabilities': {CLASS_NAMES[k]: float(pred_probs[k]) for k in range(len(CLASS_NAMES))}
            }
            
            detections.append(detection)
    
    return detections

def non_maximum_suppression(detections, nms_threshold=0.3):
    """
    Apply Non-Maximum Suppression to remove overlapping detections.
    
    Args:
        detections: List of detection dictionaries
        nms_threshold: IoU threshold for NMS
    
    Returns:
        Filtered list of detections
    """
    if not detections:
        return []
    
    # Sort by confidence (descending)
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    def calculate_iou(det1, det2):
        """Calculate Intersection over Union (IoU) between two detections."""
        x1 = max(det1['x'], det2['x'])
        y1 = max(det1['y'], det2['y'])
        x2 = min(det1['x'] + det1['width'], det2['x'] + det2['width'])
        y2 = min(det1['y'] + det1['height'], det2['y'] + det2['height'])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = det1['width'] * det1['height']
        area2 = det2['width'] * det2['height']
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    # Apply NMS
    filtered_detections = []
    
    for i, detection in enumerate(detections):
        keep = True
        
        for existing_detection in filtered_detections:
            iou = calculate_iou(detection, existing_detection)
            if iou > nms_threshold:
                keep = False
                break
        
        if keep:
            filtered_detections.append(detection)
    
    return filtered_detections

def visualize_detections(image_path, detections, output_path, confidence_threshold=0.0):
    """
    Create visualization of detected roofs on the original orthophoto.
    
    Args:
        image_path: Path to original image
        detections: List of detection dictionaries
        output_path: Path to save visualization
        confidence_threshold: Minimum confidence to display
    """
    # Load original image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Filter detections by confidence
    filtered_detections = [d for d in detections if d['confidence'] >= confidence_threshold]
    
    # Draw detections
    for detection in filtered_detections:
        x, y = detection['x'], detection['y']
        w, h = detection['width'], detection['height']
        predicted_class = detection['predicted_class']
        confidence = detection['confidence']
        
        # Get color for this class
        color = CLASS_COLORS.get(predicted_class, (255, 255, 255))
        
        # Draw rectangle
        cv2.rectangle(image_rgb, (x, y), (x + w, y + h), color, 3)
        
        # Draw label
        label = f"{predicted_class}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        
        # Background for text
        cv2.rectangle(image_rgb, (x, y - label_size[1] - 10), 
                     (x + label_size[0], y), color, -1)
        
        # Text
        cv2.putText(image_rgb, label, (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Save visualization
    plt.figure(figsize=(20, 16))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title(f'Detected Roofs: {os.path.basename(image_path)} ({len(filtered_detections)} detections)', 
              fontsize=16, pad=20)
    
    # Add legend
    legend_elements = []
    for class_name in CLASS_NAMES:
        bgr_color = CLASS_COLORS[class_name]
        rgb_color = (bgr_color[2]/255.0, bgr_color[1]/255.0, bgr_color[0]/255.0)  # Convert BGR to RGB and normalize
        legend_elements.append(patches.Patch(color=rgb_color, label=class_name))
    
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Visualization saved: {output_path}")

def process_orthophoto(model, image_path, window_sizes, stride_ratio, confidence_threshold, nms_threshold):
    """
    DEPRECATED: This function is no longer used in footprint-based processing.
    Kept for backward compatibility only.
    """
    print(f"⚠️  WARNING: process_orthophoto is deprecated. Use footprint-based processing instead.")
    return []

def save_detections_to_csv(all_detections, output_path):
    """Save all detections to CSV file."""
    csv_data = []
    
    for image_path, detections in all_detections.items():
        for detection in detections:
            row = {
                'image_name': os.path.basename(image_path),
                'image_path': image_path,
                'x': detection['x'],
                'y': detection['y'],
                'width': detection['width'],
                'height': detection['height'],
                'predicted_class': detection['predicted_class'],
                'confidence': detection['confidence']
            }
            
            # Add class probabilities
            for class_name in CLASS_NAMES:
                row[f'prob_{class_name}'] = detection['class_probabilities'][class_name]
            
            csv_data.append(row)
    
    if csv_data:
        df = pd.DataFrame(csv_data)
        df.to_csv(output_path, index=False)
        print(f"📊 Detections saved to CSV: {output_path}")
    else:
        print("⚠️  No detections to save to CSV")

def get_image_georeferencing(image_path):
    """
    Try to extract georeferencing information from the image.
    Returns transform and CRS if available, otherwise None.
    """
    if not GIS_AVAILABLE:
        return None, None, None, None
    
    try:
        with rasterio.open(image_path) as src:
            if src.crs is not None and src.transform is not None:
                return src.transform, src.crs, src.width, src.height
    except Exception as e:
        print(f"  ⚠️  Could not read georeferencing from {os.path.basename(image_path)}: {e}")
    
    return None, None, None, None

def pixel_to_geographic(x, y, transform):
    """Convert pixel coordinates to geographic coordinates using rasterio transform."""
    if transform is None:
        return x, y
    
    # Convert pixel coordinates to geographic coordinates
    geo_x, geo_y = transform * (x, y)
    return geo_x, geo_y

def save_detections_to_shapefile(image_path, detections, output_path, coordinate_system='pixel'):
    """
    Save detections from a single image to shapefile with polygon geometries.
    
    Args:
        image_path: Path to the source image
        detections: List of detection dictionaries for this image
        output_path: Path to output shapefile
        coordinate_system: 'pixel' for pixel coordinates, 'geographic' for geographic coordinates
    """
    if not GIS_AVAILABLE:
        print("❌ GIS libraries not available. Cannot save shapefile.")
        print("   Install with: pip install geopandas shapely fiona rasterio")
        return
    
    if not detections:
        print(f"⚠️  No detections to save for {os.path.basename(image_path)}")
        return
    
    geometries = []
    attributes = []
    
    # Try to get georeferencing info
    transform, crs, img_width, img_height = get_image_georeferencing(image_path)
    
    for detection in detections:
        x, y = detection['x'], detection['y']
        width, height = detection['width'], detection['height']
        
        if coordinate_system == 'geographic' and transform is not None:
            # Convert pixel coordinates to geographic coordinates
            x1_geo, y1_geo = pixel_to_geographic(x, y, transform)
            x2_geo, y2_geo = pixel_to_geographic(x + width, y + height, transform)
            
            # Create polygon in geographic coordinates
            polygon = box(x1_geo, y2_geo, x2_geo, y1_geo)  # Note: y coordinates are flipped
            coord_system = 'geographic'
        else:
            # Create polygon in pixel coordinates
            polygon = box(x, y, x + width, y + height)
            coord_system = 'pixel'
            crs = None  # No CRS for pixel coordinates
        
        geometries.append(polygon)
        
        # Prepare attributes
        attrs = {
            'image_name': os.path.basename(image_path),
            'roof_class': detection['predicted_class'],
            'confidence': round(detection['confidence'], 4),
            'coord_sys': coord_system,
            'x_pixel': x,
            'y_pixel': y,
            'width': width,
            'height': height
        }
        
        # Add class probabilities
        for class_name in CLASS_NAMES:
            attrs[f'prob_{class_name[:8]}'] = round(detection['class_probabilities'][class_name], 4)
        
        # Add geographic coordinates if available
        if coordinate_system == 'geographic' and transform is not None:
            center_x_geo, center_y_geo = pixel_to_geographic(x + width/2, y + height/2, transform)
            attrs['x_geo'] = round(center_x_geo, 6)
            attrs['y_geo'] = round(center_y_geo, 6)
        
        attributes.append(attrs)
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(attributes, geometry=geometries, crs=crs)
    
    try:
        # Save to shapefile
        gdf.to_file(output_path, driver='ESRI Shapefile')
        print(f"🗺️  Shapefile saved: {output_path}")
        print(f"   Image: {os.path.basename(image_path)}")
        print(f"   Features: {len(gdf)}")
        print(f"   Coordinate system: {coordinate_system}")
        if crs:
            print(f"   CRS: {crs}")
        
        # Also save as GeoJSON for broader compatibility
        geojson_path = output_path.replace('.shp', '.geojson')
        gdf.to_file(geojson_path, driver='GeoJSON')
        print(f"🗺️  GeoJSON saved: {geojson_path}")
        
    except Exception as e:
        print(f"❌ Error saving shapefile for {os.path.basename(image_path)}: {e}")

def create_detection_mask(image_path, detections, output_path, mask_type='class'):
    """
    Create a raster mask where each detected roof is labeled with its class or confidence.
    
    Args:
        image_path: Path to original image
        detections: List of detection dictionaries
        output_path: Path to save mask
        mask_type: 'class' for class labels, 'confidence' for confidence values
    """
    try:
        # Load original image to get dimensions
        with Image.open(image_path) as img:
            width, height = img.size
        
        # Create mask array
        if mask_type == 'class':
            # Create class mask (0 = background, 1-7 = class indices)
            mask = np.zeros((height, width), dtype=np.uint8)
            
            for detection in detections:
                x, y = detection['x'], detection['y']
                w, h = detection['width'], detection['height']
                class_idx = CLASS_NAMES.index(detection['predicted_class']) + 1
                
                # Fill detection area with class index
                mask[y:y+h, x:x+w] = class_idx
        
        elif mask_type == 'confidence':
            # Create confidence mask (0 = background, 0-255 = confidence * 255)
            mask = np.zeros((height, width), dtype=np.uint8)
            
            for detection in detections:
                x, y = detection['x'], detection['y']
                w, h = detection['width'], detection['height']
                confidence_value = int(detection['confidence'] * 255)
                
                # Fill detection area with confidence value
                mask[y:y+h, x:x+w] = confidence_value
        
        else:
            raise ValueError(f"Unknown mask_type: {mask_type}")
        
        # Save mask as PNG
        mask_image = Image.fromarray(mask, mode='L')
        mask_image.save(output_path)
        print(f"🎭 Detection mask saved: {output_path}")
        
        # Try to save as GeoTIFF if georeferencing is available
        if GIS_AVAILABLE:
            transform, crs, _, _ = get_image_georeferencing(image_path)
            if transform is not None and crs is not None:
                geotiff_path = output_path.replace('.png', '.tif')
                try:
                    with rasterio.open(
                        geotiff_path, 'w',
                        driver='GTiff',
                        height=height,
                        width=width,
                        count=1,
                        dtype=mask.dtype,
                        crs=crs,
                        transform=transform
                    ) as dst:
                        dst.write(mask, 1)
                    print(f"🌍 Georeferenced mask saved: {geotiff_path}")
                except Exception as e:
                    print(f"⚠️  Could not save georeferenced mask: {e}")
        
        return mask
        
    except Exception as e:
        print(f"❌ Error creating detection mask: {e}")
        return None

def save_class_legend(output_path):
    """Save a legend file explaining the class indices used in masks."""
    legend_content = """# Roof Type Classification - Class Legend
# 
# This file explains the class indices used in the detection masks.
# 
# Mask Values:
#   0 = Background (no roof detected)
#   1 = Complex roof
#   2 = Flat roof  
#   3 = Gable roof
#   4 = Halfhip roof
#   5 = Hip roof
#   6 = L-shaped roof
#   7 = Pyramid roof
#
# Class Details:
"""
    
    for i, class_name in enumerate(CLASS_NAMES, 1):
        legend_content += f"#   {i} = {class_name}\n"
    
    legend_content += f"""#
# Generated by: Building Roof Type Classification
# Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(output_path, 'w') as f:
        f.write(legend_content)
    
    print(f"📋 Class legend saved: {output_path}")

def print_summary(all_detections):
    """Print summary statistics."""
    total_images = len(all_detections)
    total_detections = sum(len(detections) for detections in all_detections.values())
    
    print("\n" + "="*60)
    print("ORTHOPHOTO INFERENCE SUMMARY")
    print("="*60)
    print(f"Total orthophotos processed: {total_images}")
    print(f"Total roof detections: {total_detections}")
    
    if total_detections > 0:
        print(f"Average detections per image: {total_detections/total_images:.1f}")
        
        # Class distribution
        class_counts = {}
        all_confidences = []
        
        for detections in all_detections.values():
            for detection in detections:
                predicted_class = detection['predicted_class']
                confidence = detection['confidence']
                
                class_counts[predicted_class] = class_counts.get(predicted_class, 0) + 1
                all_confidences.append(confidence)
        
        print(f"\nClass Distribution:")
        for class_name in CLASS_NAMES:
            count = class_counts.get(class_name, 0)
            percentage = (count / total_detections) * 100 if total_detections > 0 else 0
            print(f"  {class_name:12}: {count:4d} ({percentage:5.1f}%)")
        
        print(f"\nConfidence Statistics:")
        all_confidences = np.array(all_confidences)
        print(f"  Mean: {np.mean(all_confidences):.3f}")
        print(f"  Std:  {np.std(all_confidences):.3f}")
        print(f"  Min:  {np.min(all_confidences):.3f}")
        print(f"  Max:  {np.max(all_confidences):.3f}")

def create_footprint_visualization(orthophoto_path, classified_footprints, output_path):
    """Create visualization of classified footprints overlaid on orthophoto."""
    if not GIS_AVAILABLE:
        print("⚠️  GIS libraries required for footprint visualization")
        return
    
    try:
        # Load orthophoto
        with rasterio.open(orthophoto_path) as src:
            # Read RGB bands
            if src.count >= 3:
                rgb_data = src.read([1, 2, 3])
            else:
                # Grayscale - convert to RGB
                gray_data = src.read(1)
                rgb_data = np.stack([gray_data] * 3, axis=0)
            
            # Convert to HWC format
            rgb_image = np.transpose(rgb_data, (1, 2, 0))
            
            # Normalize to 0-255 if needed
            if rgb_image.dtype != np.uint8:
                if rgb_image.max() <= 1.0:
                    rgb_image = (rgb_image * 255).astype(np.uint8)
                else:
                    rgb_image = rgb_image.astype(np.uint8)
        
        # Create visualization
        plt.figure(figsize=(16, 12))
        plt.imshow(rgb_image)
        
        # Define colors for each class
        class_colors = {
            'complex': 'red',
            'flat': 'blue', 
            'gable': 'green',
            'halfhip': 'orange',
            'hip': 'purple',
            'L-shaped': 'cyan',
            'pyramid': 'yellow',
            'unknown': 'gray'
        }
        
        # Plot footprints
        with rasterio.open(orthophoto_path) as src:
            classified_count = 0
            unclassified_count = 0
            
            for idx, footprint in classified_footprints.iterrows():
                try:
                    # Convert geometry to pixel coordinates
                    geom = footprint.geometry
                    if geom.is_empty:
                        continue
                    
                    # Get exterior coordinates
                    if hasattr(geom, 'exterior'):
                        coords = list(geom.exterior.coords)
                    else:
                        coords = list(geom.coords)
                    
                    # Convert to pixel coordinates
                    pixel_coords = []
                    for lon, lat in coords:
                        col, row = src.index(lat, lon)
                        pixel_coords.append([col, row])
                    
                    pixel_coords = np.array(pixel_coords)
                    
                    # Determine color and style
                    if footprint['classified']:
                        color = class_colors.get(footprint['roof_class'], 'gray')
                        alpha = min(0.3 + footprint['confidence'] * 0.4, 0.8)
                        linewidth = 2
                        classified_count += 1
                    else:
                        color = 'lightgray'
                        alpha = 0.2
                        linewidth = 1
                        unclassified_count += 1
                    
                    # Plot footprint
                    from matplotlib.patches import Polygon
                    polygon = Polygon(pixel_coords.tolist(), fill=True, 
                                        facecolor=color, edgecolor=color,
                                        alpha=alpha, linewidth=linewidth)
                    plt.gca().add_patch(polygon)
                    
                except Exception as e:
                    continue
        
        # Add legend
        legend_elements = []
        from matplotlib.lines import Line2D
        for class_name, color in class_colors.items():
            if class_name != 'unknown':
                legend_elements.append(Line2D([0], [0], marker='s', color='w', 
                                                markerfacecolor=color, markersize=10, 
                                                label=class_name.capitalize()))
        
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
        
        plt.title(f'Classified Building Footprints\n'
                 f'Classified: {classified_count}, Unclassified: {unclassified_count}', 
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"📈 Footprint visualization saved: {output_path}")
        
    except Exception as e:
        print(f"❌ Error creating footprint visualization: {e}")
        import traceback
        traceback.print_exc()

def save_footprint_results_to_csv(results, output_path):
    """Save footprint classification results to CSV file."""
    if not results:
        print("⚠️  No results to save to CSV")
        return
    
    try:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"📊 CSV results saved: {output_path}")
        print(f"    Exported {len(results)} classified footprints")
        
    except Exception as e:
        print(f"❌ Error saving CSV results: {e}")

def print_footprint_summary(all_results):
    """Print summary statistics for footprint classifications."""
    if not all_results:
        print("\n" + "="*60)
        print("NO CLASSIFICATIONS COMPLETED")
        print("="*60)
        return
    
    total_footprints = len(all_results)
    
    print("\n" + "="*60)
    print("FOOTPRINT CLASSIFICATION SUMMARY")
    print("="*60)
    print(f"Total classified footprints: {total_footprints}")
    
    # Class distribution
    class_counts = {}
    confidences = []
    heights = []
    
    for result in all_results:
        roof_class = result['roof_class']
        confidence = result['confidence']
        
        class_counts[roof_class] = class_counts.get(roof_class, 0) + 1
        confidences.append(confidence)
        
        # Collect height data if available
        if 'mean_height' in result and not pd.isna(result['mean_height']):
            heights.append(result['mean_height'])
    
    print(f"\nClass Distribution:")
    for class_name in CLASS_NAMES:
        count = class_counts.get(class_name, 0)
        percentage = (count / total_footprints) * 100 if total_footprints > 0 else 0
        print(f"  {class_name:12}: {count:4d} ({percentage:5.1f}%)")
    
    if confidences:
        confidences = np.array(confidences)
        print(f"\nConfidence Statistics:")
        print(f"  Mean: {np.mean(confidences):.3f}")
        print(f"  Std:  {np.std(confidences):.3f}")
        print(f"  Min:  {np.min(confidences):.3f}")
        print(f"  Max:  {np.max(confidences):.3f}")
    
    if heights:
        heights = np.array(heights)
        print(f"\nHeight Statistics (from DSM):")
        print(f"  Buildings with height data: {len(heights)}/{total_footprints}")
        print(f"  Mean height: {np.mean(heights):.2f} m")
        print(f"  Std height:  {np.std(heights):.2f} m")
        print(f"  Min height:  {np.min(heights):.2f} m")
        print(f"  Max height:  {np.max(heights):.2f} m")

def main():
    parser = argparse.ArgumentParser(
        description="Building Roof Type Classification - Footprint-Based Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic footprint classification (requires footprints/ subdirectory)
  python orthophoto_inference.py --input_dir test_orthophotos/
  
  # With DSM for height extraction
  python orthophoto_inference.py --input_dir test_orthophotos/ --dsm_dir test_dsm/
  
  # With higher confidence threshold
  python orthophoto_inference.py --input_dir test_orthophotos/ --confidence_threshold 0.8
  
  # With visualization of classified footprints
  python orthophoto_inference.py --input_dir test_orthophotos/ --visualize
  
  # Save classification results to CSV
  python orthophoto_inference.py --input_dir test_orthophotos/ --output_csv footprint_results.csv
  
Directory Structure:
  test_orthophotos/
  ├── orthophoto1.tif
  ├── orthophoto2.tif
  └── footprints/
      ├── orthophoto1_footprints.shp  (or orthophoto1.shp)
      └── orthophoto2_footprints.shp  (or orthophoto2.shp)
  
  Optional DSM Structure:
  test_dsm/
  ├── orthophoto1_dsm.tif  (or orthophoto1_DSM.tif)
  └── orthophoto2_dsm.tif
        """
    )
    
    parser.add_argument('--input_dir', required=True,
                       help='Directory containing orthophotos and footprints/ subdirectory')
    parser.add_argument('--dsm_dir', default=None,
                       help='Directory containing DSM files for height extraction (optional)')
    parser.add_argument('--model_path', default=None,
                       help='Path to trained model (auto-detect if not specified)')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                       help='Minimum confidence threshold for classifications (default: 0.5)')
    parser.add_argument('--output_csv', default=None,
                       help='Output CSV file for classification results')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations of classified footprints')
    parser.add_argument('--output_dir', default='orthophoto_results',
                       help='Output directory for results (default: orthophoto_results)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("FOOTPRINT-BASED ROOF CLASSIFICATION")
    print("="*60)
    print(f"Confidence threshold: {args.confidence_threshold}")
    if args.dsm_dir:
        print(f"DSM directory: {args.dsm_dir}")
    
    # Check GIS availability
    if not GIS_AVAILABLE:
        print("❌ Error: GIS libraries (geopandas, rasterio, shapely) are required for footprint-based inference")
        print("Install them using: pip install geopandas rasterio shapely")
        sys.exit(1)
    
    # Load model
    model = load_trained_model(args.model_path)
    
    # Get orthophoto-footprint-DSM triplets
    print(f"\nScanning directory: {args.input_dir}")
    if args.dsm_dir:
        print(f"DSM directory: {args.dsm_dir}")
    try:
        triplets = get_orthophoto_footprint_dsm_triplets(args.input_dir, args.dsm_dir)
        print(f"Found {len(triplets)} orthophoto-footprint pairs")
        dsm_count = sum(1 for _, _, dsm in triplets if dsm is not None)
        if dsm_count > 0:
            print(f"DSM files available for {dsm_count} orthophotos")
    except Exception as e:
        print(f"❌ Error finding orthophoto-footprint pairs: {e}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each orthophoto-footprint-DSM triplet
    all_results = []
    
    for i, (orthophoto_path, footprint_path, dsm_path) in enumerate(triplets, 1):
        print(f"\n[{i}/{len(triplets)}] " + "="*50)
        
        # Process footprints for this orthophoto
        classified_footprints = process_footprints(
            model, orthophoto_path, footprint_path, args.confidence_threshold, dsm_path
        )
        
        if classified_footprints is not None:
            # Save classified footprints
            base_name = os.path.splitext(os.path.basename(orthophoto_path))[0]
            output_shapefile = os.path.join(args.output_dir, f"{base_name}_classified.shp")
            save_classified_footprints(classified_footprints, output_shapefile)
            
            # Create visualization if requested
            if args.visualize:
                viz_filename = f"{base_name}_classified_footprints.png"
                viz_path = os.path.join(args.output_dir, viz_filename)
                create_footprint_visualization(orthophoto_path, classified_footprints, viz_path)
            
            # Collect results for CSV export
            for idx, footprint in classified_footprints.iterrows():
                if footprint['classified']:
                    result = {
                        'orthophoto': os.path.basename(orthophoto_path),
                        'footprint_id': idx,
                        'roof_class': footprint['roof_class'],
                        'confidence': footprint['confidence']
                    }
                    
                    # Add height statistics if available
                    if dsm_path and not pd.isna(footprint.get('mean_height', np.nan)):
                        result['mean_height'] = footprint['mean_height']
                        result['min_height'] = footprint['min_height']
                        result['max_height'] = footprint['max_height']
                        result['std_height'] = footprint['std_height']
                        result['height_pixels'] = footprint['height_px']
                    
                    # Add class probabilities
                    for class_name in CLASS_NAMES:
                        result[f'prob_{class_name[:8]}'] = footprint[f'prob_{class_name[:8]}']
                    
                    all_results.append(result)
    
    # Save CSV results if requested
    if args.output_csv and all_results:
        csv_path = args.output_csv if os.path.dirname(args.output_csv) else os.path.join(args.output_dir, args.output_csv)
        save_footprint_results_to_csv(all_results, csv_path)
    
    # Print summary
    print_footprint_summary(all_results)
    
    print(f"\n" + "="*60)
    print("PROCESSING COMPLETED")
    print("="*60)
    
    print("\nFiles generated:")
    print(f"  🏗️  Classified shapefiles: {args.output_dir}/*_classified.shp")
    print(f"  🗺️  GeoJSON files: {args.output_dir}/*_classified.geojson")
    if args.output_csv:
        print(f"  📊 CSV results: {csv_path}")
    if args.visualize:
        print(f"  📈 Footprint visualizations: {args.output_dir}/*_classified_footprints.png")
    if args.dsm_dir:
        print(f"  📏 Height statistics included in outputs")

if __name__ == "__main__":
    main()
