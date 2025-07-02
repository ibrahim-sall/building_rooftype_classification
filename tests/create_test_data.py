#!/usr/bin/env python3
"""
Create Test Data for Orthophoto Inference

This script creates minimal test files to verify that orthophoto_inference.py
is working correctly without needing real data.
"""

import os
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
import rasterio
from rasterio.transform import from_bounds
from PIL import Image

def create_test_data(test_dir="test_data"):
    """Create minimal test data for orthophoto inference testing."""
    print(f"Creating test data in: {test_dir}")
    
    os.makedirs(test_dir, exist_ok=True)
    footprints_dir = os.path.join(test_dir, "footprints")
    dsm_dir = os.path.join(test_dir, "dsm")
    os.makedirs(footprints_dir, exist_ok=True)
    os.makedirs(dsm_dir, exist_ok=True)

    minx, miny, maxx, maxy = 563000, 5934000, 563500, 5934500
    
    orthophoto_path = os.path.join(test_dir, "test_orthophoto.tif")
    
    width, height = 500, 500
    rgb_data = np.random.randint(0, 255, size=(3, height, width), dtype=np.uint8)

    for i in range(5):
        x = np.random.randint(50, width-100)
        y = np.random.randint(50, height-100)
        w = np.random.randint(30, 80)
        h = np.random.randint(30, 80)
        color = np.random.randint(100, 200, size=3)
        for c in range(3):
            rgb_data[c, y:y+h, x:x+w] = color[c]
    
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    
    # Save georeferenced orthophoto
    with rasterio.open(
        orthophoto_path, 'w',
        driver='GTiff',
        height=height,
        width=width,
        count=3,
        dtype=rgb_data.dtype,
        crs='EPSG:25832',
        transform=transform
    ) as dst:
        dst.write(rgb_data)
    
    print(f"Created: {orthophoto_path}")
    
    # Create test footprints shapefile
    print("Creating test footprints...")
    footprints_path = os.path.join(footprints_dir, "test_orthophoto_footprints.shp")
    
    footprints = []
    for i in range(10):
        
        center_x = minx + np.random.random() * (maxx - minx)
        center_y = miny + np.random.random() * (maxy - miny)
        size = np.random.uniform(20, 50)

        footprint = Polygon([
            (center_x - size/2, center_y - size/2),
            (center_x + size/2, center_y - size/2),
            (center_x + size/2, center_y + size/2),
            (center_x - size/2, center_y + size/2)
        ])
        footprints.append(footprint)
    
    gdf = gpd.GeoDataFrame(
        {'building_id': range(len(footprints))},
        geometry=footprints,
        crs='EPSG:25832'
    )
    
    gdf.to_file(footprints_path)
    print(f"Created: {footprints_path}")
    
    print("Creating test DSM...")
    dsm_path = os.path.join(dsm_dir, "test_orthophoto_dsm.tif")
    
    dsm_data = np.random.randint(50, 150, size=(height, width), dtype=np.uint8)
    
    # Add some building heights
    for i in range(5):
        x = np.random.randint(50, width-100)
        y = np.random.randint(50, height-100)
        w = np.random.randint(30, 80)
        h = np.random.randint(30, 80)
        building_height = np.random.randint(180, 250)
        dsm_data[y:y+h, x:x+w] = building_height
    
    with rasterio.open(
        dsm_path, 'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=dsm_data.dtype,
        crs='EPSG:25832',
        transform=transform
    ) as dst:
        dst.write(dsm_data, 1)
    
    print(f"Created: {dsm_path}")
    
    print("Test data creation complete!")
    
if __name__ == "__main__":
    create_test_data()
