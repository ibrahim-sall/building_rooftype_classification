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
    
    # Create directories
    os.makedirs(test_dir, exist_ok=True)
    footprints_dir = os.path.join(test_dir, "footprints")
    dsm_dir = os.path.join(test_dir, "dsm")
    os.makedirs(footprints_dir, exist_ok=True)
    os.makedirs(dsm_dir, exist_ok=True)
    
    # Define a small test area (in EPSG:25832 coordinates)
    minx, miny, maxx, maxy = 563000, 5934000, 563500, 5934500
    
    # Create test orthophoto (500x500 pixels, RGB)
    print("Creating test orthophoto...")
    orthophoto_path = os.path.join(test_dir, "test_orthophoto.tif")
    
    # Create random RGB image data
    width, height = 500, 500
    rgb_data = np.random.randint(0, 255, size=(3, height, width), dtype=np.uint8)
    
    # Add some building-like patterns
    for i in range(5):
        x = np.random.randint(50, width-100)
        y = np.random.randint(50, height-100)
        w = np.random.randint(30, 80)
        h = np.random.randint(30, 80)
        color = np.random.randint(100, 200, size=3)
        for c in range(3):
            rgb_data[c, y:y+h, x:x+w] = color[c]
    
    # Create transform
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
    
    print(f"✅ Created: {orthophoto_path}")
    
    # Create test footprints shapefile
    print("Creating test footprints...")
    footprints_path = os.path.join(footprints_dir, "test_orthophoto_footprints.shp")
    
    # Create some building footprints
    footprints = []
    for i in range(10):
        # Random building footprint
        center_x = minx + np.random.random() * (maxx - minx)
        center_y = miny + np.random.random() * (maxy - miny)
        size = np.random.uniform(20, 50)  # Building size in meters
        
        # Create rectangular footprint
        footprint = Polygon([
            (center_x - size/2, center_y - size/2),
            (center_x + size/2, center_y - size/2),
            (center_x + size/2, center_y + size/2),
            (center_x - size/2, center_y + size/2)
        ])
        footprints.append(footprint)
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        {'building_id': range(len(footprints))},
        geometry=footprints,
        crs='EPSG:25832'
    )
    
    # Save shapefile
    gdf.to_file(footprints_path)
    print(f"✅ Created: {footprints_path}")
    
    # Create test DSM
    print("Creating test DSM...")
    dsm_path = os.path.join(dsm_dir, "test_orthophoto_dsm.tif")
    
    # Create height data (0-255 pixel values)
    dsm_data = np.random.randint(50, 150, size=(height, width), dtype=np.uint8)
    
    # Add some building heights
    for i in range(5):
        x = np.random.randint(50, width-100)
        y = np.random.randint(50, height-100)
        w = np.random.randint(30, 80)
        h = np.random.randint(30, 80)
        building_height = np.random.randint(180, 250)  # Taller buildings
        dsm_data[y:y+h, x:x+w] = building_height
    
    # Save georeferenced DSM
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
    
    print(f"✅ Created: {dsm_path}")
    
    # Create test runner script
    test_script_path = os.path.join(test_dir, "run_test.sh")
    with open(test_script_path, 'w') as f:
        f.write(f"""#!/bin/bash
# Test script for orthophoto inference

echo "Running basic test..."
python ../orthophoto_inference.py --input_dir . --confidence_threshold 0.3

echo ""
echo "Running test with DSM..."
python ../orthophoto_inference.py --input_dir . --dsm_dir dsm --confidence_threshold 0.3

echo ""
echo "Running test with visualization..."
python ../orthophoto_inference.py --input_dir . --dsm_dir dsm --confidence_threshold 0.3 --visualize --output_csv test_results.csv

echo ""
echo "Test completed! Check orthophoto_results/ directory for outputs."
""")
    
    os.chmod(test_script_path, 0o755)
    print(f"✅ Created: {test_script_path}")
    
    # Create Python test runner
    py_test_script = os.path.join(test_dir, "run_test.py")
    with open(py_test_script, 'w') as f:
        f.write("""#!/usr/bin/env python3
import subprocess
import sys
import os

def run_test():
    print("🔍 Testing orthophoto inference script...")
    
    tests = [
        {
            "name": "Basic test",
            "cmd": ["python", "../orthophoto_inference.py", "--input_dir", ".", "--confidence_threshold", "0.3"]
        },
        {
            "name": "Test with DSM",
            "cmd": ["python", "../orthophoto_inference.py", "--input_dir", ".", "--dsm_dir", "dsm", "--confidence_threshold", "0.3"]
        },
        {
            "name": "Test with visualization",
            "cmd": ["python", "../orthophoto_inference.py", "--input_dir", ".", "--dsm_dir", "dsm", "--confidence_threshold", "0.3", "--visualize", "--output_csv", "test_results.csv"]
        }
    ]
    
    for i, test in enumerate(tests, 1):
        print(f"\\n[{i}/{len(tests)}] {test['name']}...")
        try:
            result = subprocess.run(test['cmd'], check=True, capture_output=True, text=True)
            print(f"✅ {test['name']} passed")
        except subprocess.CalledProcessError as e:
            print(f"❌ {test['name']} failed:")
            print(f"   stdout: {e.stdout}")
            print(f"   stderr: {e.stderr}")
            return False
    
    print("\\n🎉 All tests passed!")
    print("📁 Check orthophoto_results/ directory for outputs.")
    return True

if __name__ == "__main__":
    if not run_test():
        sys.exit(1)
""")
    
    print(f"✅ Created: {py_test_script}")
    
    print("\n" + "="*60)
    print("TEST DATA CREATION COMPLETED")
    print("="*60)
    print(f"📁 Test directory: {test_dir}")
    print("📄 Files created:")
    print(f"   - test_orthophoto.tif (georeferenced orthophoto)")
    print(f"   - footprints/test_orthophoto_footprints.shp (building footprints)")
    print(f"   - dsm/test_orthophoto_dsm.tif (georeferenced DSM)")
    print(f"   - run_test.sh (bash test script)")
    print(f"   - run_test.py (python test script)")
    print("\n🚀 To run tests:")
    print(f"   cd {test_dir}")
    print("   python run_test.py")
    print("   # or")
    print("   bash run_test.sh")

if __name__ == "__main__":
    create_test_data()
