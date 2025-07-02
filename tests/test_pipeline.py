#!/usr/bin/env python3
"""
Full Pipeline Test with Mock Model

This creates a mock model to test the complete inference pipeline.
"""

import sys
import os
sys.path.append('..')

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers

def create_mock_model():
    """Create a simple mock model that mimics the roof classification model."""
    print("Creating mock model...")

    inputs = layers.Input(shape=(140, 140, 3))
    x = layers.GlobalAveragePooling2D()(inputs)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(7, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Save to models directory
    models_dir = "./models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "fine_tuned_vgg16_final_mock.keras")
    model.save(model_path)
    
    print(f"Mock model saved to: {model_path}")
    return model_path

def run_full_test():
    """Run the full inference pipeline with mock model."""
    print("Running full pipeline test...")
    
    # Create mock model
    model_path = create_mock_model()
    
    import subprocess
    
    cmd = [
        "python", "../orthophoto_inference.py",
        "--input_dir", ".",
        "--dsm_dir", "dsm",
        "--confidence_threshold", "0.1",
        "--visualize",
        "--output_csv", "test_results.csv"
    ]
    
    print("🚀 Running inference script...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Inference script completed successfully!")
        print("\n Script output:")
        print(result.stdout)
        
        # Check output files
        output_dir = "orthophoto_results"
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            print(f"\n Output files created ({len(files)} files):")
            for file in sorted(files):
                file_path = os.path.join(output_dir, file)
                size = os.path.getsize(file_path)
                print(f"   - {file} ({size} bytes)")
        
        # Check CSV results
        csv_path = "test_results.csv"
        if os.path.exists(csv_path):
            import pandas as pd
            df = pd.read_csv(csv_path)
            print(f"\n CSV results: {len(df)} classified footprints")
            if len(df) > 0:
                print("   Sample results:")
                print(df[['roof_class', 'confidence']].head().to_string(index=False))
        
        return True
        
    except subprocess.CalledProcessError as e:
        print("Inference script failed!")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False

def cleanup():
    """Clean up test files."""
    print("\n Cleaning up...")
    
    # Remove mock model
    model_path = "./models/fine_tuned_vgg16_final_mock.keras"
    if os.path.exists(model_path):
        os.remove(model_path)
        print(f"   Removed: {model_path}")

    print("(Keeping test results for inspection)")

if __name__ == "__main__":
    print("FULL PIPELINE TEST")
    print("=" * 50)
    
    try:
        success = run_full_test()
        cleanup()
        
        if success:
            print("\n FULL PIPELINE TEST PASSED!")
            print("Your orthophoto inference script is working correctly.")
        else:
            print("\n FULL PIPELINE TEST FAILED!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n Test crashed: {e}")
        import traceback
        traceback.print_exc()
        cleanup()
        sys.exit(1)
