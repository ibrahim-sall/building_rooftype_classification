#!/usr/bin/env python3
"""
DSM Debug Script for orthophoto_inference.py

This script helps debug DSM (Digital Surface Model) integration issues,
particularly the "Invalid footprint bounds for DSM" error.
"""

import os
import sys
import numpy as np

try:
    import geopandas as gpd
    import rasterio
    from shapely.geometry import Polygon
    GIS_AVAILABLE = True
except ImportError:
    print("❌ GIS libraries not available. Install with: pip install geopandas rasterio shapely")
    sys.exit(1)

def debug_dsm_footprint_alignment(dsm_path, footprint_path):
    """Debug DSM and footprint alignment issues."""
    print("🔍 DSM-Footprint Debug Analysis")
    print("="*50)
    
    # Check if files exist
    if not os.path.exists(dsm_path):
        print(f"❌ DSM file not found: {dsm_path}")
        return
    
    if not os.path.exists(footprint_path):
        print(f"❌ Footprint file not found: {footprint_path}")
        return
    
    try:
        # Load DSM information
        print(f"\n📊 DSM Analysis: {os.path.basename(dsm_path)}")
        with rasterio.open(dsm_path) as dsm:
            print(f"  ✅ DSM loaded successfully")
            print(f"  📏 Dimensions: {dsm.width} x {dsm.height} pixels")
            print(f"  🗺️  CRS: {dsm.crs}")
            print(f"  📍 Bounds: {dsm.bounds}")
            print(f"  🔢 Data type: {dsm.dtypes[0]}")
            print(f"  ❌ NoData value: {dsm.nodata}")
            
            # Sample DSM data
            sample_data = dsm.read(1, window=((0, min(100, dsm.height)), (0, min(100, dsm.width))))
            valid_mask = sample_data != dsm.nodata if dsm.nodata is not None else np.ones_like(sample_data, dtype=bool)
            valid_data = sample_data[valid_mask]
            
            if len(valid_data) > 0:
                print(f"  📈 Pixel value range: {np.min(valid_data):.2f} - {np.max(valid_data):.2f}")
                print(f"  📊 Mean pixel value: {np.mean(valid_data):.2f}")
                print(f"  ⚠️  Note: These are pixel values (0-255), not heights in meters")
            else:
                print(f"  ⚠️  No valid height data found in sample")
    
    except Exception as e:
        print(f"❌ Error loading DSM: {e}")
        return
    
    try:
        # Load footprint information
        print(f"\n🏠 Footprint Analysis: {os.path.basename(footprint_path)}")
        footprints = gpd.read_file(footprint_path)
        print(f"  ✅ Footprints loaded successfully")
        print(f"  🔢 Number of footprints: {len(footprints)}")
        print(f"  🗺️  CRS: {footprints.crs}")
        
        # Get overall bounds
        total_bounds = footprints.total_bounds
        print(f"  📍 Bounds: ({total_bounds[0]:.6f}, {total_bounds[1]:.6f}, {total_bounds[2]:.6f}, {total_bounds[3]:.6f})")
        
        # Analyze first few footprints
        print(f"\n🔍 Individual Footprint Analysis:")
        for i, (idx, footprint) in enumerate(footprints.head(3).iterrows()):
            geom = footprint.geometry
            bounds = geom.bounds
            area = geom.area
            print(f"  Footprint {idx}:")
            print(f"    📍 Bounds: ({bounds[0]:.6f}, {bounds[1]:.6f}, {bounds[2]:.6f}, {bounds[3]:.6f})")
            print(f"    📏 Area: {area:.2f} square units")
            
    except Exception as e:
        print(f"❌ Error loading footprints: {e}")
        return
    
    # Check CRS compatibility
    print(f"\n🎯 CRS Compatibility Check:")
    with rasterio.open(dsm_path) as dsm:
        dsm_crs = dsm.crs
        footprint_crs = footprints.crs
        
        if dsm_crs == footprint_crs:
            print(f"  ✅ CRS match: {dsm_crs}")
        else:
            print(f"  ⚠️  CRS mismatch!")
            print(f"    DSM CRS: {dsm_crs}")
            print(f"    Footprint CRS: {footprint_crs}")
            print(f"    🔧 Footprints will be reprojected automatically")
    
    # Check spatial overlap
    print(f"\n🗺️  Spatial Overlap Check:")
    with rasterio.open(dsm_path) as dsm:
        dsm_bounds = dsm.bounds
        footprint_bounds = footprints.total_bounds
        
        # Convert footprint bounds to DSM CRS if needed
        if dsm.crs != footprints.crs and footprints.crs is not None:
            try:
                # Reproject footprints to DSM CRS for comparison
                reproj_footprints = footprints.to_crs(dsm.crs)
                footprint_bounds = reproj_footprints.total_bounds
                print(f"  🔄 Footprint bounds reprojected to DSM CRS")
            except Exception as e:
                print(f"  ⚠️  Could not reproject for comparison: {e}")
        
        print(f"  DSM bounds:       ({dsm_bounds.left:.6f}, {dsm_bounds.bottom:.6f}, {dsm_bounds.right:.6f}, {dsm_bounds.top:.6f})")
        print(f"  Footprint bounds: ({footprint_bounds[0]:.6f}, {footprint_bounds[1]:.6f}, {footprint_bounds[2]:.6f}, {footprint_bounds[3]:.6f})")
        
        # Check overlap
        overlap_x = max(0, min(dsm_bounds.right, footprint_bounds[2]) - max(dsm_bounds.left, footprint_bounds[0]))
        overlap_y = max(0, min(dsm_bounds.top, footprint_bounds[3]) - max(dsm_bounds.bottom, footprint_bounds[1]))
        
        if overlap_x > 0 and overlap_y > 0:
            print(f"  ✅ Spatial overlap detected: {overlap_x:.2f} x {overlap_y:.2f}")
        else:
            print(f"  ❌ No spatial overlap detected!")
            print(f"     This is likely the cause of 'Invalid footprint bounds for DSM' error")
    
    print(f"\n💡 Recommendations:")
    if dsm_crs != footprint_crs:
        print(f"  • The script will automatically handle CRS reprojection")
    if overlap_x <= 0 or overlap_y <= 0:
        print(f"  • Check that DSM and footprints cover the same geographic area")
        print(f"  • Verify coordinate systems are correctly defined")
        print(f"  • Consider using a DSM that covers the footprint area")
    else:
        print(f"  • Spatial setup looks correct")
        print(f"  • If still getting errors, check for individual footprint issues")

def main():
    if len(sys.argv) != 3:
        print("Usage: python debug_dsm.py <dsm_file> <footprint_file>")
        print("Example: python debug_dsm.py orthophoto1_dsm.tif footprints/orthophoto1_footprints.shp")
        sys.exit(1)
    
    dsm_path = sys.argv[1]
    footprint_path = sys.argv[2]
    
    debug_dsm_footprint_alignment(dsm_path, footprint_path)

if __name__ == "__main__":
    main()
