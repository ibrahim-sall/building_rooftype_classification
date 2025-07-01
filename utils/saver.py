
import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import cv2
from skimage.util import view_as_windows
from PIL import Image, ImageDraw, ImageFont

import geopandas as gpd
from shapely.geometry import Polygon, box
import rasterio
from rasterio.windows import from_bounds, Window
from rasterio.transform import from_bounds as transform_from_bounds


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


def save_classified_footprints(footprints_gdf, output_path, logger=None):
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
