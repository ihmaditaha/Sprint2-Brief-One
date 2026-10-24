from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
EXTRACT_DIR = PROJECT_ROOT / "extract"

# Google Drive Dataset Configuration
FILE_ID = "1PP09vNLPEsiFXFyKu4_8mC-bFJbxfflC"
DATASET_ZIP_NAME = "tumors_dataset.zip"

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
]
NUM_CLASSES = len(CLASS_NAMES)

IMAGE_SIZE = (224, 224)
IMAGE_CHANNELS = 3
BATCH_SIZE = 32
AUTOTUNE = "tf.data.AUTOTUNE"

RANDOM_STATE = 42

TRANSFER_LEARNING_EPOCHS = 10
TRANSFER_LEARNING_LR = 1e-3

FINE_TUNING_EPOCHS = 10
FINE_TUNING_LR = 1e-5
FINE_TUNE_LAST_LAYERS = 30

MLFLOW_EXPERIMENT_NAME = "NeuroVision-Drive-MobileNetV2"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"