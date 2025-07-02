#!/usr/bin/env python3
"""
Simple Test for Orthophoto Inference Components
"""

import sys
import os
sys.path.append('..')

# Test individual components without needing a trained model
def test_components():
    print("🔍 Testing individual components...")
    
    try:
        # Test imports
        print("1. Testing imports...")
        from utils.loader import get_orthophoto_footprint_dsm_triplets
        from utils.footprints import extract_footprint_image
        import geopandas as gpd
        import rasterio
        print("   ✅ All imports successful")
        
        # Test file discovery
        print("2. Testing file discovery...")
        SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.jp2', '.webp', '.gif'}
        triplets = get_orthophoto_footprint_dsm_triplets('.', SUPPORTED_FORMATS, 'dsm')
        print(f"   ✅ Found {len(triplets)} triplets")
        
        if triplets:
            orthophoto_path, footprint_path, dsm_path = triplets[0]
            print(f"   📷 Orthophoto: {os.path.basename(orthophoto_path)}")
            print(f"   🏠 Footprints: {os.path.basename(footprint_path)}")
            print(f"   📏 DSM: {os.path.basename(dsm_path) if dsm_path else 'None'}")
            
            # Test footprint loading
            print("3. Testing footprint loading...")
            footprints_gdf = gpd.read_file(footprint_path)
            print(f"   ✅ Loaded {len(footprints_gdf)} footprints")
            
            # Test orthophoto loading
            print("4. Testing orthophoto loading...")
            with rasterio.open(orthophoto_path) as src:
                print(f"   ✅ Orthophoto: {src.width}x{src.height}, {src.count} bands, CRS: {src.crs}")
            
            # Test DSM loading
            if dsm_path:
                print("5. Testing DSM loading...")
                with rasterio.open(dsm_path) as src:
                    print(f"   ✅ DSM: {src.width}x{src.height}, CRS: {src.crs}")
            
            # Test footprint extraction (without model)
            print("6. Testing footprint extraction...")
            footprint_geom = footprints_gdf.geometry.iloc[0]
            extracted_image = extract_footprint_image(orthophoto_path, footprint_geom)
            if extracted_image:
                print(f"   ✅ Extracted footprint image: {extracted_image.size}")
            else:
                print("   ⚠️  Could not extract footprint image")
        
        print("\n🎉 Component tests completed successfully!")
        print("💡 To run full test, ensure you have a trained model in ../models/")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_components()
