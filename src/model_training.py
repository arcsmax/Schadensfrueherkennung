# src/model_training.py
"""
Modul für das Training, die Evaluierung und die Anwendung des ML-Modells.
Strukturiert für einen professionellen Workflow.
"""
import logging
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from . import config
from .utils import plot_feature_distribution
from .feature_orchestrator import _process_single_file # Wiederverwendung der Worker-Funktion

def run_model_training_pipeline() -> bool:
    """
    Hauptfunktion für den Modelltrainings-Schritt. Führt den vollständigen
    Trainings-, Validierungs- und Speicher-Workflow durch.
    """
    logging.info("="*50)
    logging.info("Schritt 3: Modelltraining und Validierung")
    logging.info("="*50)
    
    try:
        features_df = pd.read_csv(config.TRAIN_PROCESSED_FEATURES_PATH)
    except FileNotFoundError:
        logging.error(f"Trainings-Feature-Datei nicht gefunden: {config.TRAIN_PROCESSED_FEATURES_PATH}")
        return False
    
    features_df.rename(columns={'pitting_level': 'label'}, inplace=True)
    
    metadata_cols = ['rpm_input', 'torque_output_nm', 'repetition', 'label']
    features_to_use = [col for col in features_df.columns if col not in metadata_cols]
    
    X = features_df[features_to_use]
    y = features_df['label']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=config.VALIDATION_SIZE, random_state=config.RANDOM_STATE, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    model = RandomForestClassifier(n_estimators=config.N_ESTIMATORS, max_depth=config.MAX_DEPTH, random_state=config.RANDOM_STATE, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    y_pred_val = model.predict(X_val_scaled)
    report = classification_report(y_val, y_pred_val)
    logging.info("--- Validierungs-Report (dient nur zur Info) ---")
    print(report) # Direkte Ausgabe ist hier für den Nutzer hilfreich
    
    # Korrigiert: Stelle sicher, dass der Report ein String ist, bevor er geschrieben wird
    with open(config.VALIDATION_CLASSIFICATION_REPORT_PATH, 'w') as f:
        f.write(str(report))

    logging.info("Trainiere finales Modell auf dem gesamten Trainingsdatensatz...")
    final_scaler = StandardScaler().fit(X)
    X_scaled_full = final_scaler.transform(X)
    
    final_model = RandomForestClassifier(n_estimators=config.N_ESTIMATORS, max_depth=config.MAX_DEPTH, random_state=config.RANDOM_STATE, n_jobs=-1).fit(X_scaled_full, y)
    
    joblib.dump(final_model, config.SAVED_MODEL_PATH)
    joblib.dump(final_scaler, config.SAVED_PREPROCESSOR_PATH)
    joblib.dump(features_to_use, config.SAVED_MODEL_COLUMNS_PATH)
    logging.info(f"Finales Modell, Preprocessor und Spaltennamen gespeichert in: {config.RESULTS_DIR}")

    # Korrigiert: Die Plot-Funktion wird korrekt aus utils importiert und aufgerufen
    plot_feature_distribution(features_df.copy(), config.TRAIN_VISUALIZATION_PATH)
    return True

def predict_single_instance(filepath: str) -> tuple:
    """Macht eine Vorhersage für eine einzelne Rohdatendatei."""
    logging.info(f"Starte Echtzeit-Vorhersage für: {filepath}")
    try:
        model = joblib.load(config.SAVED_MODEL_PATH)
        preprocessor = joblib.load(config.SAVED_PREPROCESSOR_PATH)
        model_columns = joblib.load(config.SAVED_MODEL_COLUMNS_PATH)
    except FileNotFoundError:
        logging.error("Modell-Artefakte nicht gefunden. Bitte zuerst die Trainings-Pipeline ausführen.")
        return None, None

    dummy_row = pd.Series({'filepath': filepath})
    features_dict = _process_single_file(dummy_row)
    if not features_dict: return None, None
        
    feature_df = pd.DataFrame([features_dict])[model_columns]
    scaled_features = preprocessor.transform(feature_df)
    
    prediction = model.predict(scaled_features)
    probabilities = model.predict_proba(scaled_features)
    return prediction[0], probabilities[0]

def evaluate_on_test_data() -> bool:
    """Lädt das Modell und evaluiert es auf dem ungesehenen Testdatensatz."""
    logging.info("Starte finale Evaluierung auf dem separaten Testdatensatz...")
    try:
        model = joblib.load(config.SAVED_MODEL_PATH)
        preprocessor = joblib.load(config.SAVED_PREPROCESSOR_PATH)
        model_columns = joblib.load(config.SAVED_MODEL_COLUMNS_PATH)
        test_features_df = pd.read_csv(config.TEST_PROCESSED_FEATURES_PATH)
    except FileNotFoundError as e:
        logging.error(f"Benötigte Datei für Test-Evaluierung nicht gefunden: {e}.")
        return False
        
    test_features_df.rename(columns={'pitting_level': 'label'}, inplace=True)
    X_test = test_features_df[model_columns]
    y_test = test_features_df['label']
    X_test_scaled = preprocessor.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    report = classification_report(y_test, y_pred)
    logging.info("--- Finaler Klassifikationsbericht für den TEST-Datensatz ---")
    print(report)
    with open(config.TEST_CLASSIFICATION_REPORT_PATH, 'w') as f:
        f.write(str(report))
    return True
