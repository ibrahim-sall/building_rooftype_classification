import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import rasterio



def create_footprint_visualization(orthophoto_path, classified_footprints, output_path):
    """Create visualization of classified footprints overlaid on orthophoto."""
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
