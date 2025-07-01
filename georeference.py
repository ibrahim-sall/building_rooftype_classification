#!/usr/bin/env python3
"""
Georeference DSM Script

This script adds georeferencing information to a DSM file based on footprint bounds.
"""

import os
import sys
import argparse
import numpy as np


import rasterio
from rasterio.transform import from_bounds
import geopandas as gpd
import logging


def setup_logger():
    """Setup and return a logger instance."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)


def georeference_dsm(dsm_path, footprint_path, output_path, target_crs="EPSG:25832", logger=None):
    """
    Add georeferencing to a DSM based on footprint bounds.
    
    Args:
        dsm_path: Path to the ungeoreferenced DSM
        footprint_path: Path to footprint shapefile (to get bounds)
        output_path: Path for the georeferenced DSM
        target_crs: Target coordinate reference system
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    

    footprints = gpd.read_file(footprint_path)
    logger.info(f"📍 Footprint bounds: {footprints.total_bounds}")
    
    minx, miny, maxx, maxy = footprints.total_bounds
    
    with rasterio.open(dsm_path) as src:

        dsm_data = src.read(1)
        
        transform = from_bounds(minx, miny, maxx, maxy, src.width, src.height)
        
        logger.info(f"📏 DSM dimensions: {src.width} x {src.height}")
        logger.info(f"🔢 DSM data type: {src.dtypes[0]}")
        
        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=src.height,
            width=src.width,
            count=1,
            dtype=src.dtypes[0],
            crs=target_crs,
            transform=transform,
            nodata=src.nodata
        ) as dst:
            dst.write(dsm_data, 1)
    

    logger.info(f"✅ Georeferenced DSM saved: {output_path}")
    logger.info(f"🗺️  CRS: {target_crs}")
    logger.info(f"📍 Bounds: ({minx:.3f}, {miny:.3f}, {maxx:.3f}, {maxy:.3f})")
    logger.info(f"📏 Pixel size: {(maxx-minx)/src.width:.3f} x {(maxy-miny)/src.height:.3f}")

def batch_georeference_dsm(dsm_dir, footprint_dir, output_dir, target_crs="EPSG:25832", logger=None):
    """
    Georeference all DSM files in a directory using corresponding footprint files.
    
    Args:
        dsm_dir: Directory containing DSM files
        footprint_dir: Directory containing footprint shapefiles
        output_dir: Directory for georeferenced DSM outputs
        target_crs: Target coordinate reference system
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    logger.info(f"🔍 Batch georeferencing DSMs from: {dsm_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    dsm_files = []
    for filename in os.listdir(dsm_dir):
        if filename.lower().endswith(('.tif', '.tiff')):
            dsm_files.append(filename)
    
    if not dsm_files:
        logger.error("No dsm files found in dir")
        return
    
    logger.info(f"📊 Found {len(dsm_files)} DSM files")
    
    processed = 0
    
    for dsm_filename in dsm_files:
        dsm_path = os.path.join(dsm_dir, dsm_filename)
        base_name = os.path.splitext(dsm_filename)[0]
        
        # Remove common DSM suffixes to find corresponding footprint
        footprint_base = base_name.replace('_dsm', '').replace('_DSM', '').replace('_height', '').replace('_heights', '')
        
        # Look for corresponding footprint file
        possible_footprint_names = [
            f"{footprint_base}_footprints.shp",
            f"{footprint_base}.shp",
            f"{footprint_base}_buildings.shp",
            f"{footprint_base}_emprise.shp"
        ]
        
        footprint_path = None
        for footprint_name in possible_footprint_names:
            potential_path = os.path.join(footprint_dir, footprint_name)
            if os.path.exists(potential_path):
                footprint_path = potential_path
                break
        
        if footprint_path:
            output_filename = f"{base_name}_georeferenced.tif"
            output_path = os.path.join(output_dir, output_filename)
            
            try:
                print(f"\n[{processed+1}/{len(dsm_files)}] Processing: {dsm_filename}")
                georeference_dsm(dsm_path, footprint_path, output_path, target_crs)
                processed += 1
            except Exception as e:
                logger.error(f"❌ Error processing {dsm_filename}: {e}")
        else:
            logger.warning(f"⚠️  No footprint file found for {dsm_filename}")
    
    logger.info(f"\n✅ Batch processing completed: {processed}/{len(dsm_files)} files processed")

def main():
    parser = argparse.ArgumentParser(
        description="Georeference DSM files based on footprint bounds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  python georeference_dsm.py dsm.tif footprint.shp output.tif
  
  # Batch process all DSMs in directory
  python georeference_dsm.py --batch dsm_dir/ footprint_dir/ output_dir/
        """
    )
    
    # Add batch processing option
    parser.add_argument('--batch', action='store_true', help='Batch process all DSM files in directory')
    parser.add_argument('path1', help='DSM file/directory path')
    parser.add_argument('path2', help='Footprint file/directory path')  
    parser.add_argument('path3', help='Output file/directory path')
    parser.add_argument('--crs', default='EPSG:25832', help='Target CRS (default: EPSG:25832)')
    
    args = parser.parse_args()
    
    logger = setup_logger()
    
    if args.batch:
        if not os.path.exists(args.path1):
            logger.error(f"DSM directory not found: {args.path1}")
            sys.exit(1)
        
        if not os.path.exists(args.path2):
            logger.error(f"Footprint directory not found: {args.path2}")
            sys.exit(1)
        
        batch_georeference_dsm(args.path1, args.path2, args.path3, args.crs, logger)
    else:
        if not os.path.exists(args.path1):
            logger.error(f"DSM file not found: {args.path1}")
            sys.exit(1)
        
        if not os.path.exists(args.path2):
            logger.error(f"Footprint file not found: {args.path2}")
            sys.exit(1)
        
        georeference_dsm(args.path1, args.path2, args.path3, args.crs, logger)

if __name__ == "__main__":
    main()
