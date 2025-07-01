
import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd


import cv2
from skimage.util import view_as_windows
from PIL import Image, ImageDraw, ImageFont

import geopandas as gpd
from shapely.geometry import Polygon, box


def save_detections_to_csv(all_detections, output_path, CLASS_NAMES, logger=None):
    """Save all detections to CSV file."""
    if logger is None:
        logger = logging.getLogger(__name__)
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
        logger.info(f"Detections saved to CSV: {output_path}")
    else:
        logger.error("No detections to save to CSV")
        
        
def save_classified_footprints(footprints_gdf, output_path, logger=None):
    """Save classified footprints to shapefile."""
    if logger is None:
        logger = logging.getLogger(__name__)
        
    if footprints_gdf is None or len(footprints_gdf) == 0:
        logger.info(" No footprints to save")
        return
    
    try:
        footprints_gdf.to_file(output_path, driver='ESRI Shapefile')
        logger.info(f"Classified footprints saved: {output_path}")
        
        geojson_path = output_path.replace('.shp', '.geojson')
        footprints_gdf.to_file(geojson_path, driver='GeoJSON')
        logger.info(f"GeoJSON saved: {geojson_path}")
        
        # Print classification summary
        classified_footprints = footprints_gdf[footprints_gdf['classified'] == True]
        if len(classified_footprints) > 0:
            logger.info(f"lassification summary:")
            class_counts = classified_footprints['roof_class'].value_counts()
            for class_name, count in class_counts.items():
                percentage = (count / len(classified_footprints)) * 100
                print(f"    {class_name}: {count} ({percentage:.1f}%)")
        
    except Exception as e:
        logger.error(f"Error saving classified footprints: {e}")
        

def save_footprint_results_to_csv(results, output_path, logger=None):
    """Save footprint classification results to CSV file."""
    if logger is None:
        logger = logging.getLogger(__name__)
    if not results:
        logger.info("No results to save to CSV")
        return
    
    try:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        logger.info(f"CSV results saved: {output_path}")
        logger.info(f"    Exported {len(results)} classified footprints")
        
    except Exception as e:
        logger.error(f"Error saving CSV results: {e}")
