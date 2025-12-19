#!/usr/bin/env python3
"""Test script to check which imports are working"""

print("Testing imports...")

try:
    import streamlit
    print("✅ Streamlit:", streamlit.__version__)
except ImportError as e:
    print("❌ Streamlit:", e)

try:
    import numpy
    print("✅ NumPy:", numpy.__version__)
except ImportError as e:
    print("❌ NumPy:", e)

try:
    import cv2
    print("✅ OpenCV:", cv2.__version__)
except ImportError as e:
    print("❌ OpenCV:", e)

try:
    import PIL
    print("✅ PIL/Pillow:", PIL.__version__)
except ImportError as e:
    print("❌ PIL/Pillow:", e)

try:
    import librosa
    print("✅ Librosa:", librosa.__version__)
except ImportError as e:
    print("❌ Librosa:", e)

print("\nTesting MVP modules...")

try:
    from mvp_image import extract_face_features
    print("✅ mvp_image imported successfully")
except Exception as e:
    print("❌ mvp_image:", e)

try:
    from mvp_audio import extract_audio_features
    print("✅ mvp_audio imported successfully")
except Exception as e:
    print("❌ mvp_audio:", e)

try:
    from mvp_model import compute_hydration_score, build_recommendations
    print("✅ mvp_model imported successfully")
except Exception as e:
    print("❌ mvp_model:", e)

print("\nImport test complete!")
