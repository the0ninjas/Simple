#!/usr/bin/env python3
"""Debug script to check openface installation."""
import sys

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print("\nTrying to import openface...")

try:
    import openface
    print("✓ Successfully imported openface")
    print(f"✓ openface module location: {openface.__file__}")
    print(f"✓ openface package: {openface.__package__}")
    print(f"✓ openface.__version__: {getattr(openface, '__version__', 'N/A')}")
    
    # Try importing the specific modules
    try:
        from openface.face_detection import FaceDetector
        print("✓ Successfully imported FaceDetector")
    except Exception as e:
        print(f"✗ Failed to import FaceDetector: {e}")
    
    try:
        from openface.multitask_model import MultitaskPredictor
        print("✓ Successfully imported MultitaskPredictor")
    except Exception as e:
        print(f"✗ Failed to import MultitaskPredictor: {e}")
        
except ImportError as e:
    print(f"✗ Failed to import openface: {e}")
    print("\nTroubleshooting:")
    print(f"1. Check if openface-test is installed: pip list | grep openface-test")
    print(f"2. Check Python path: {sys.path}")
    sys.exit(1)

print("\n✓ All imports successful!")
