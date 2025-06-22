# src/config.py
"""
Zentrale Konfigurationsdatei für die Schadensfrüherkennungs-Pipeline.
Enthält alle Pfade, physikalischen Parameter und Modell-Hyperparameter.
"""
from pathlib import Path

# -- Basis-Pfade und Verzeichnisse --
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
CACHE_STATUS_DIR = BASE_DIR / ".cache_status"

# NEU: Logging-Konfiguration
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE_PATH = LOG_DIR / "pipeline.log"

# -- Datenpfade --
# Trainingsdaten
TRAIN_SOURCE_DATA_PATH = DATA_DIR / "Data_Challenge_PHM2023_training_data"
TRAIN_METADATA_PATH = DATA_DIR / "phm_metadata_train.csv"
TRAIN_PROCESSED_FEATURES_PATH = DATA_DIR / "processed_features_train.csv"

# Testdaten (für die finale, ungesehene Evaluation)
TEST_SOURCE_DATA_PATH = DATA_DIR / "test_data"
TEST_METADATA_PATH = DATA_DIR / "phm_metadata_test.csv"
TEST_PROCESSED_FEATURES_PATH = DATA_DIR / "processed_features_test.csv"

# -- Ergebnis- und Modellpfade --
SAVED_MODEL_PATH = RESULTS_DIR / "final_model.joblib"
SAVED_PREPROCESSOR_PATH = RESULTS_DIR / "preprocessor.joblib"
SAVED_MODEL_COLUMNS_PATH = RESULTS_DIR / "model_columns.joblib" # Für Spaltennamen
TRAIN_VISUALIZATION_PATH = RESULTS_DIR / "feature_comparison_plot_train.png"
VALIDATION_CLASSIFICATION_REPORT_PATH = RESULTS_DIR / "validation_classification_report.txt"
TEST_CLASSIFICATION_REPORT_PATH = RESULTS_DIR / "final_test_classification_report.txt"


# Verzeichnisse erstellen
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
(BASE_DIR / ".cache_status").mkdir(exist_ok=True)


# -- Physikalische Systemparameter (NEU) --
# Diese Werte sind für die GMF- und Ordnungsanalyse entscheidend.
# Annahmen basierend auf typischen Getrieben; müssen für reale Systeme angepasst werden.
SAMPLING_RATE = 20480  # Hz, aus dem Bericht bestätigt
INPUT_SHAFT_GEAR_TEETH = 32  # Zähnezahl des Zahnrads auf der Eingangswelle
OUTPUT_SHAFT_GEAR_TEETH = 56 # Zähnezahl des Zahnrads auf der Ausgangswelle
GEAR_RATIO = OUTPUT_SHAFT_GEAR_TEETH / INPUT_SHAFT_GEAR_TEETH # ca. 1.75

# GMF-Ordnung = Zähnezahl des Zahnrads, dessen Welle als Referenz (1. Ordnung) dient.
# Wir nehmen die Eingangswelle als Referenz.
GMF_ORDER = INPUT_SHAFT_GEAR_TEETH

# -- Feature Engineering Parameter --
# DWT Konfiguration
DWT_WAVELET = 'db4'  # Daubechies 4, ein guter Allrounder für transiente Signale
DWT_LEVEL = 5        # Anzahl der Dekompositionsebenen

# Ordnungsanalyse Konfiguration
ORDER_RESOLUTION = 0.1  # Auflösung des Ordnungsspektrums
MAX_ORDER = GMF_ORDER * 3.5  # Maximale Ordnung, die analysiert wird

# -- Modelltraining Parameter --
VALIDATION_SIZE = 0.2 # 20% der Trainingsdaten für die Validierung
RANDOM_STATE = 42
N_ESTIMATORS = 150
MAX_DEPTH = 15