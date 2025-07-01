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
- Mean height pixel value of the building footprint (0-255 range, not meters)
- Minimum and maximum height pixel values
- Standard deviation of height pixel values
- Number of valid height pixels

Features:
1. **Footprint-Based Classification**: Uses existing building footprints as input
2. **Multi-Channel Image Support**: Automatically handles 3, 4, or more channel images
3. **Geographic Coordinate Support**: Works with georeferenced orthophotos and footprints
4. **Height Extraction**: Optional DSM processing for building height statistics
5. **Confidence Scoring**: Provides classification confidence for each footprint
6. **Batch Processing**: Process multiple orthophotos and their corresponding footprints
7. **Labeled Output**: Creates new shapefiles with roof type classifications and heights

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
import logging
import argparse
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils.loader import *
from utils.saver import *
from utils.visualiser import *
from utils.footprints import *


import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

import cv2
from skimage.util import view_as_windows
from PIL import Image, ImageDraw, ImageFont

import geopandas as gpd
from shapely.geometry import Polygon, box
import rasterio
from rasterio.windows import from_bounds, Window
from rasterio.transform import from_bounds as transform_from_bounds



# Configuration
IMG_HEIGHT = 140
IMG_WIDTH = 140
CLASS_NAMES = ['complex', 'flat', 'gable', 'halfhip', 'hip', 'L-shaped', 'pyramid']
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.jp2', '.webp', '.gif'}

CLASS_COLORS = {
    'complex': (0, 255, 255),   
    'flat': (255, 0, 0),        
    'gable': (0, 255, 0),       
    'halfhip': (255, 165, 0),   
    'hip': (128, 0, 128),       
    'L-shaped': (255, 192, 203),
    'pyramid': (255, 0, 255)    
}



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
    
    logger.info("="*60)
    logger.info("FOOTPRINT-BASED ROOF CLASSIFICATION")
    logger.info("="*60)
    logger.info(f"Confidence threshold: {args.confidence_threshold}")
    if args.dsm_dir:
        logger.info(f"DSM directory: {args.dsm_dir}")
    
    
    # Load model
    model = load_trained_model(args.model_path)
    
    # Get orthophoto-footprint-DSM triplets
    logger.info(f"\nScanning directory: {args.input_dir}")
    if args.dsm_dir:
        logger.info(f"DSM directory: {args.dsm_dir}")
    try:
        triplets = get_orthophoto_footprint_dsm_triplets(args.input_dir, args.dsm_dir)
        logger.info(f"Found {len(triplets)} orthophoto-footprint pairs")
        dsm_count = sum(1 for _, _, dsm in triplets if dsm is not None)
        if dsm_count > 0:
            logger.info(f"DSM files available for {dsm_count} orthophotos")
    except Exception as e:
        logger.error(f"Error finding orthophoto-footprint pairs: {e}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each orthophoto-footprint-DSM triplet
    all_results = []
    
    for i, (orthophoto_path, footprint_path, dsm_path) in enumerate(triplets, 1):
        logger.info(f"\n[{i}/{len(triplets)}] " + "="*50)
        
        # Process footprints for this orthophoto
        classified_footprints = process_footprints(
            model, orthophoto_path, footprint_path, CLASS_NAMES, IMG_HEIGHT, IMG_WIDTH,
            confidence_threshold=args.confidence_threshold,
            dsm_path=dsm_path,
            logger=logger
        )
        
        if classified_footprints is not None:
            # Save classified footprints
            base_name = os.path.splitext(os.path.basename(orthophoto_path))[0]
            output_shapefile = os.path.join(args.output_dir, f"{base_name}_classified.shp")
            save_classified_footprints(classified_footprints, output_shapefile)
            
            if args.visualize:
                viz_filename = f"{base_name}_classified_footprints.png"
                viz_path = os.path.join(args.output_dir, viz_filename)
                create_footprint_visualization(orthophoto_path, classified_footprints, viz_path)
            
            for idx, footprint in classified_footprints.iterrows():
                if footprint['classified']:
                    result = {
                        'orthophoto': os.path.basename(orthophoto_path),
                        'footprint_id': idx,
                        'roof_class': footprint['roof_class'],
                        'confidence': footprint['confidence']
                    }
                    
                    if dsm_path and not pd.isna(footprint.get('mean_height', np.nan)):
                        result['mean_height'] = footprint['mean_height']
                        result['min_height'] = footprint['min_height']
                        result['max_height'] = footprint['max_height']
                        result['std_height'] = footprint['std_height']
                        result['height_pixels'] = footprint['height_px']
                    
                    for class_name in CLASS_NAMES:
                        result[f'prob_{class_name[:8]}'] = footprint[f'prob_{class_name[:8]}']
                    
                    all_results.append(result)

    if args.output_csv and all_results:
        csv_path = args.output_csv if os.path.dirname(args.output_csv) else os.path.join(args.output_dir, args.output_csv)
        save_footprint_results_to_csv(all_results, csv_path)
    
    print_footprint_summary(all_results, CLASS_NAMES)
    
    logger.info(f"\n" + "="*60)
    logger.info("PROCESSING COMPLETED")
    logger.info("="*60)
    
    logger.info("\nFiles generated:")
    logger.info(f"Classified shapefiles: {args.output_dir}/*_classified.shp")
    logger.info(f"GeoJSON files: {args.output_dir}/*_classified.geojson")
    if args.output_csv:
        logger.info(f"CSV results: {csv_path}")
    if args.visualize:
        logger.info(f"Footprint visualizations: {args.output_dir}/*_classified_footprints.png")
    if args.dsm_dir:
        logger.info(f"Height statistics included in outputs")


if __name__ == "__main__":
    
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)

    main()
