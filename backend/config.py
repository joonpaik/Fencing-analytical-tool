"""
Configuration settings for Fencing Analysis MVP
"""

import os
from pathlib import Path

class Config:
    # Project paths
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "output"
    MODELS_DIR = PROJECT_ROOT / "models"
    
    # OpenPose configuration
    OPENPOSE_MODEL_PATH = "/path/to/openpose/models/"  # Update this path
    OPENPOSE_NET_RESOLUTION = "368x368"
    OPENPOSE_OUTPUT_RESOLUTION = "-1x-1"
    USE_GPU = True
    
    # Video processing settings
    TARGET_FPS = 30
    TARGET_RESOLUTION = "1280x720"
    VIDEO_QUALITY = 23  # CRF value for FFmpeg
    
    # Frame preprocessing
    BLUR_KERNEL_SIZE = (5, 5)
    BRIGHTNESS_ALPHA = 1.2
    BRIGHTNESS_BETA = 30
    BLUR_THRESHOLD = 100  # Laplacian variance threshold
    
    # Feature extraction
    SMOOTHING_WINDOW = 5
    VELOCITY_THRESHOLD = 0.1
    ANGLE_CALCULATION_JOINTS = {
        'arm_angle': ['shoulder', 'elbow', 'wrist'],
        'leg_angle': ['hip', 'knee', 'ankle']
    }
    
    # Movement classification
    MOVEMENT_TYPES = ['advance', 'retreat', 'lunge', 'parry', 'riposte']
    SEQUENCE_LENGTH = 30  # Frames to consider for movement classification
    CONFIDENCE_THRESHOLD = 0.7
    
    # Output settings
    SAVE_ANNOTATED_VIDEO = True
    SAVE_INTERMEDIATE_FILES = True  # Set False for production
    REPORT_FORMAT = "html"  # html, json, pdf

