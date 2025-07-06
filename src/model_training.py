# src/model_training.py
"""
Modul für das Training, die Evaluierung und die Anwendung des klassischen ML-Modells.
Speichert alle notwendigen Artefakte für das Ergebnis-Dashboard und enthält
dedizierte Funktionen für Inferenz und Test-Evaluierung.
"""
import logging
import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from . import config
from .utils import plot_feature_distribution
# Importiere die einzelnen Extraktionsfunktionen für die Inferenz
from .feature_engineering.time_domain import extract_time_domain_features
from .feature_engineering.time_frequency import extract_dwt_features
from .feature_engineering.order_analysis import extract_order_analysis_features

def run_model_training_pipeline() -> bool:
    """
    Hauptfunktion für den Modelltrainings-Schritt. Führt den vollständigen
    Trainings-, Validierungs- und Speicher-Workflow durch.
    """
    logging.info("="*50)
    logging.info("Schritt 3: Modelltraining und Validierung (RandomForest)")
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

    # --- 1. Validierungs-Split ---
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=config.VALIDATION_SIZE, random_state=config.RANDOM_STATE, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = RandomForestClassifier(n_estimators=config.N_ESTIMATORS, max_depth=config.MAX_DEPTH, random_state=config.RANDOM_STATE, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    y_pred_val = model.predict(X_val_scaled)
    class_names = [f'Level {lvl}' for lvl in sorted(y.unique())] # Für schönere Labels im Report
    
    # Text-Report für die Konsole und .txt-Datei
    report_text = classification_report(y_val, y_pred_val, target_names=class_names, zero_division=0)
    logging.info("--- Validierungs-Report (RandomForest) ---")
    print(report_text)
    with open(config.VALIDATION_CLASSIFICATION_REPORT_PATH, 'w') as f:
        f.write(report_text)

    # NEU: Speichere den Report auch als JSON für das Dashboard
    report_dict = classification_report(y_val, y_pred_val, target_names=class_names, zero_division=0, output_dict=True)
    with open(config.VALIDATION_CLASSIFICATION_REPORT_JSON_PATH, 'w') as f:
        json.dump(report_dict, f, indent=4)
    logging.info(f"JSON-Report für Dashboard gespeichert: {config.VALIDATION_CLASSIFICATION_REPORT_JSON_PATH}")


    # --- 2. Finales Training auf dem GESAMTEN Datensatz ---
    logging.info("Trainiere finales Modell auf dem gesamten Trainingsdatensatz...")
    final_scaler = StandardScaler().fit(X)
    X_scaled_full = final_scaler.transform(X)

    final_model = RandomForestClassifier(n_estimators=config.N_ESTIMATORS, max_depth=config.MAX_DEPTH, random_state=config.RANDOM_STATE, n_jobs=-1).fit(X_scaled_full, y)

    # --- 3. Speichern aller Artefakte für das Dashboard ---
    joblib.dump(final_model, config.SAVED_MODEL_PATH)
    joblib.dump(final_scaler, config.SAVED_PREPROCESSOR_PATH)
    joblib.dump(features_to_use, config.SAVED_MODEL_COLUMNS_PATH)

    # Feature Importances als JSON speichern
    importances = pd.Series(final_model.feature_importances_, index=features_to_use)
    top_importances = importances.nlargest(15).to_dict()
    with open(config.FEATURE_IMPORTANCE_PATH, 'w') as f:
        json.dump(top_importances, f, indent=4)
    logging.info(f"Feature Importances gespeichert in: {config.FEATURE_IMPORTANCE_PATH}")

    logging.info(f"Finales Modell und alle Artefakte gespeichert in: {config.RESULTS_DIR}")

    plot_feature_distribution(features_df.copy(), config.TRAIN_VISUALIZATION_PATH)
    return True

def predict_single_instance(filepath: str) -> tuple:
    """
    Macht eine Vorhersage für eine einzelne Rohdatendatei.
    Dieser Prozess ist autark und lädt das Modell und die Extraktionslogik.
    """
    logging.info(f"Starte Echtzeit-Vorhersage für: {filepath}")
    try:
        model = joblib.load(config.SAVED_MODEL_PATH)
        preprocessor = joblib.load(config.SAVED_PREPROCESSOR_PATH)
        model_columns = joblib.load(config.SAVED_MODEL_COLUMNS_PATH)
    except FileNotFoundError:
        logging.error("Modell-Artefakte nicht gefunden. Bitte zuerst die Trainings-Pipeline ausführen.")
        return None, None

    try:
        data = pd.read_csv(filepath, sep=r'\s+', header=None, names=['acc_x', 'acc_y', 'acc_z', 'tacho'], engine='python')
        signal = data['acc_x'].values
        tacho_signal = data['tacho'].values

        time_features = extract_time_domain_features(signal)
        dwt_features = extract_dwt_features(signal)
        order_features = extract_order_analysis_features(signal, tacho_signal)
        
        features_dict = {**time_features, **dwt_features, **order_features}
        feature_df = pd.DataFrame([features_dict])
        
        feature_df_aligned = feature_df[model_columns]
        
        scaled_features = preprocessor.transform(feature_df_aligned)
        prediction = model.predict(scaled_features)
        probabilities = model.predict_proba(scaled_features)
        
        return prediction[0], probabilities[0]
    except Exception as e:
        logging.error(f"Fehler bei der Inferenz für '{filepath}': {e}")
        return None, None

def evaluate_on_test_data() -> bool:
    """
    Lädt das gespeicherte Modell und evaluiert es auf dem ungesehenen Testdatensatz.
    """
    logging.info("Starte finale Evaluierung auf dem separaten Testdatensatz...")
    try:
        model = joblib.load(config.SAVED_MODEL_PATH)
        preprocessor = joblib.load(config.SAVED_PREPROCESSOR_PATH)
        model_columns = joblib.load(config.SAVED_MODEL_COLUMNS_PATH)
        test_features_df = pd.read_csv(config.TEST_PROCESSED_FEATURES_PATH)
    except FileNotFoundError as e:
        logging.error(f"Benötigte Datei für Test-Evaluierung nicht gefunden: {e}. Bitte Trainings- und Feature-Pipelines ausführen.")
        return False
        
    test_features_df.rename(columns={'pitting_level': 'label'}, inplace=True)
    
    X_test = test_features_df[model_columns]
    y_test = test_features_df['label']
    
    X_test_scaled = preprocessor.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    report = classification_report(y_test, y_pred, zero_division=0)
    logging.info("--- Finaler Klassifikationsbericht für den TEST-Datensatz ---")
    print(report)
    with open(config.TEST_CLASSIFICATION_REPORT_PATH, 'w') as f:
        f.write(str(report))
    return True
