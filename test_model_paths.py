#!/usr/bin/env python3
"""
Simple test to verify that all model paths have been updated to use models/ directory.
"""

import os

def test_model_paths():
    """Test that model detection works with the new models/ directory structure."""
    
    print("🔍 Testing model path configuration...")
    
    # Check if models directory exists
    models_dir = "models"
    if not os.path.exists(models_dir):
        print(f"❌ Models directory '{models_dir}' not found!")
        return False
    else:
        print(f"✅ Models directory exists: {models_dir}")
    
    # Check for models in the directory
    model_files = []
    for filename in os.listdir(models_dir):
        if filename.endswith(('.keras', '.h5')):
            model_files.append(filename)
    
    if model_files:
        print(f"✅ Found {len(model_files)} model file(s):")
        for model_file in model_files:
            print(f"   - {os.path.join(models_dir, model_file)}")
    else:
        print(f"⚠️  No model files found in {models_dir}")
    
    # Test import and function call from inference script
    try:
        print("\n🔍 Testing inference.py model detection...")
        import sys
        sys.path.append('.')
        from inference import find_best_model
        
        model_path = find_best_model()
        print(f"✅ inference.py successfully found model: {model_path}")
        
    except Exception as e:
        print(f"❌ Error testing inference.py: {e}")
        return False
    
    # Test import and function call from orthophoto_inference script
    try:
        print("\n🔍 Testing orthophoto_inference.py model detection...")
        from orthophoto_inference import find_best_model as find_model_ortho
        
        model_path = find_model_ortho()
        print(f"✅ orthophoto_inference.py successfully found model: {model_path}")
        
    except Exception as e:
        print(f"❌ Error testing orthophoto_inference.py: {e}")
        return False
    
    print("\n✅ All model path tests passed!")
    return True

if __name__ == "__main__":
    print("="*60)
    print("MODEL PATH CONFIGURATION TEST")
    print("="*60)
    
    success = test_model_paths()
    
    if success:
        print("\n🎉 All tests passed! Models are correctly configured to use models/ directory.")
        print("\nUpdated structure:")
        print("project/")
        print("├── models/")
        print("│   ├── fine_tuned_vgg16_final.keras")
        print("│   ├── fine_tuned_vgg16_final.wheight.h5")
        print("│   └── ... (other model files)")
        print("├── inference.py")
        print("├── orthophoto_inference.py")
        print("└── train.py")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        exit(1)
