# src/config.py
"""
Zentrale Konfigurationsdatei für die Schadensfrüherkennungs-Pipeline.
Enthält alle Pfade, physikalischen Parameter und Modell-Hyperparameter.
"""
import json
from pathlib import Path

# -- Basis-Pfade --
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
CACHE_STATUS_DIR = BASE_DIR / ".cache_status"
LOG_DIR = BASE_DIR / "logs"


# -- Datenpfade --
TRAIN_SOURCE_DATA_PATH = DATA_DIR / "Data_Challenge_PHM2023_training_data"
TRAIN_METADATA_PATH = DATA_DIR / "phm_metadata_train.csv"
TRAIN_PROCESSED_FEATURES_PATH = DATA_DIR / "processed_features_train.csv"
TEST_SOURCE_DATA_PATH = DATA_DIR / "test_data"
TEST_METADATA_PATH = DATA_DIR / "phm_metadata_test.csv"
TEST_PROCESSED_FEATURES_PATH = DATA_DIR / "processed_features_test.csv"


# -- Ergebnis-Verzeichnisse --
# Eigene Unterordner für jedes Modell, um die Ergebnisse sauber zu trennen
CLASSICAL_ML_DIR = RESULTS_DIR / "classical_ml"
CNN_DIR = RESULTS_DIR / "cnn_model"
TRANSFORMER_DIR = RESULTS_DIR / "transformer_model"
TRANSFER_LEARNING_DIR = RESULTS_DIR / "transfer_learning"


# -- Ergebnis- und Artefakt-Pfade --

# Klassisches ML (RandomForest)
SAVED_MODEL_PATH = CLASSICAL_ML_DIR / "final_model.joblib"
SAVED_PREPROCESSOR_PATH = CLASSICAL_ML_DIR / "preprocessor.joblib"
SAVED_MODEL_COLUMNS_PATH = CLASSICAL_ML_DIR / "model_columns.joblib"
FEATURE_IMPORTANCE_PATH = CLASSICAL_ML_DIR / "feature_importances.json"
VALIDATION_CLASSIFICATION_REPORT_PATH = CLASSICAL_ML_DIR / "validation_classification_report.txt"
VALIDATION_CLASSIFICATION_REPORT_JSON_PATH = CLASSICAL_ML_DIR / "validation_classification_report.json" # Fürs Dashboard
TRAIN_VISUALIZATION_PATH = CLASSICAL_ML_DIR / "feature_comparison_plot_train.png" # KORREKTUR: Diese Zeile wurde wieder hinzugefügt

# Deep Learning (1D-CNN)
SAVED_CNN_MODEL_PATH = CNN_DIR / "1d_cnn_model.pth"
CNN_CLASSIFICATION_REPORT_PATH = CNN_DIR / "cnn_classification_report.txt"
CNN_CLASSIFICATION_REPORT_JSON_PATH = CNN_DIR / "cnn_classification_report.json" # Fürs Dashboard
CNN_PREDICTIONS_PATH = CNN_DIR / "cnn_predictions.json"

# Deep Learning (Transformer)
SAVED_TRANSFORMER_MODEL_PATH = TRANSFORMER_DIR / "transformer_model.pth"
TRANSFORMER_CLASSIFICATION_REPORT_PATH = TRANSFORMER_DIR / "transformer_classification_report.txt"
TRANSFORMER_CLASSIFICATION_REPORT_JSON_PATH = TRANSFORMER_DIR / "transformer_classification_report.json" # Fürs Dashboard
TRANSFORMER_PREDICTIONS_PATH = TRANSFORMER_DIR / "transformer_predictions.json"

# Transfer Learning
PRETRAINED_MODEL_PATH = TRANSFER_LEARNING_DIR / "pretrained_cnn.pth"
FINETUNED_MODEL_PATH = TRANSFER_LEARNING_DIR / "finetuned_cnn.pth"
TRANSFER_LEARNING_REPORT_PATH = TRANSFER_LEARNING_DIR / "transfer_learning_report.txt"

# Logging
LOG_FILE_PATH = LOG_DIR / "pipeline.log"


# -- Notwendige Verzeichnisse erstellen --
# Stellt sicher, dass alle Ordner existieren, bevor etwas gespeichert wird
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
CACHE_STATUS_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
CLASSICAL_ML_DIR.mkdir(exist_ok=True)
CNN_DIR.mkdir(exist_ok=True)
TRANSFORMER_DIR.mkdir(exist_ok=True)
TRANSFER_LEARNING_DIR.mkdir(exist_ok=True)


# -- Physikalische Systemparameter --
SAMPLING_RATE = 20480  # Hz
INPUT_SHAFT_GEAR_TEETH = 32
OUTPUT_SHAFT_GEAR_TEETH = 56
GEAR_RATIO = OUTPUT_SHAFT_GEAR_TEETH / INPUT_SHAFT_GEAR_TEETH
GMF_ORDER = INPUT_SHAFT_GEAR_TEETH


# -- Feature Engineering Parameter --
DWT_WAVELET = 'db4'
DWT_LEVEL = 5
ORDER_RESOLUTION = 0.1
MAX_ORDER = GMF_ORDER * 3.5


# -- Klassisches Modelltraining Parameter --
VALIDATION_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 150
MAX_DEPTH = 15
DATA_FRACTION = 1.0  # NEU: Standardmäßig 100% der Daten verwenden


# -- Deep Learning Hyperparameter --
NUM_EPOCHS = 25
BATCH_SIZE = 32
LEARNING_RATE = 0.001
SIGNAL_LENGTH = 20480 
EARLY_STOPPING_PATIENCE = 5


# -- Transformer Hyperparameter --
PATCH_LENGTH = 128
NUM_PATCHES = SIGNAL_LENGTH // PATCH_LENGTH
D_MODEL = 128
N_HEADS = 8
N_LAYERS = 3
TRANSFORMER_DROPOUT = 0.2


# -- Transfer Learning Simulations-Parameter --
PRETRAIN_CLASSES = [0, 1, 2]
FINETUNE_CLASSES = [6, 8]
FINETUNE_SAMPLES_PER_CLASS = 10