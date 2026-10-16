"""
SignBridge Application Settings and Configuration Constants.

Centralizes all project-wide settings, directory paths,
hyperparameters, inference configurations, and UI theme colors.
"""

import os

# ==============================================================================
# Path Configurations
# ==============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_PATH = (
    os.path.join(PROJECT_ROOT, 'dataset', 'cleaned_images')
    if os.path.isdir(os.path.join(PROJECT_ROOT, 'dataset', 'cleaned_images'))
    else os.path.join(os.path.dirname(PROJECT_ROOT), 'SignBridgeDataset', 'CollectedImages')
)
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'yolov5_sign_cls.pt')
CLASS_LABELS_PATH = os.path.join(PROJECT_ROOT, 'models', 'class_labels.json')
DATASET_REPORT_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'dataset_report.json')
YOLOV5_DIR = os.path.join(PROJECT_ROOT, 'yolov5')
YOLOV5_DATASET_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'yolov5_cls')

# ==============================================================================
# Model Training Settings
# ==============================================================================
IMG_SIZE = 224          # Input size for the model (224x224 RGB)
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 0.001
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# ==============================================================================
# Inference & Camera Settings
# ==============================================================================
CONFIDENCE_THRESHOLD = 0.40   # Minimum confidence to accept a prediction
INFERENCE_INTERVAL_MS = 200   # Milliseconds between inference runs
CAMERA_FPS = 30
CAMERA_INDEX = 0

# ==============================================================================
# Prediction Stabilization
# ==============================================================================
STABILIZATION_WINDOW = 5      # Number of recent frames to consider
STABILIZATION_AGREEMENT = 3   # Minimum agreeing frames in the window
COOLDOWN_FRAMES = 4           # Frames to wait before re-recognizing

# ==============================================================================
# UI Settings
# ==============================================================================
WINDOW_TITLE = 'SignBridge — Sign Language Recognition Assistant'
WINDOW_MIN_WIDTH = 1200
WINDOW_MIN_HEIGHT = 700

# ==============================================================================
# UI Color Palette (Hex)
# ==============================================================================
BG_PRIMARY = '#1a1a2e'        # Near-black navy
BG_SECONDARY = '#16213e'      # Dark panel
BG_CARD = '#1f2b47'           # Card background
ACCENT = '#00d4aa'            # Teal accent
ACCENT_HOVER = '#00f5c4'
ACCENT_DIM = '#007a63'
TEXT_PRIMARY = '#e8e8e8'       # Light text
TEXT_SECONDARY = '#8892a8'     # Muted text
TEXT_MUTED = '#5a6278'
BORDER_COLOR = '#2a3550'      # Subtle borders
SUCCESS = '#4ade80'
WARNING = '#fbbf24'
ERROR = '#f87171'
