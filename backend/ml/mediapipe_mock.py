"""
Mock mediapipe module for Python 3.13 compatibility
This allows the project to run without the actual mediapipe dependency
"""

import numpy as np
from typing import Any, List

class FaceMesh:
    def __init__(self, static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
    
    def process(self, image):
        # Return mock results
        class MockResults:
            def __init__(self):
                self.multi_face_landmarks = [MockLandmarks()]
        
        return MockResults()

class MockLandmarks:
    def __init__(self):
        # Create mock landmarks
        self.landmark = [MockLandmark() for _ in range(468)]  # MediaPipe face mesh has 468 landmarks

class MockLandmark:
    def __init__(self):
        self.x = 0.5
        self.y = 0.5
        self.z = 0.0

# Create the solutions module
class Solutions:
    def __init__(self):
        self.face_mesh = FaceMesh

# Create the main mediapipe module
class MediaPipe:
    def __init__(self):
        self.solutions = Solutions()

# Create the module
mp = MediaPipe()

