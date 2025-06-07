# src/model_training.py
"""
Modul überarbeitet für einen professionellen Workflow:
1. Skalierung der Daten.
2. Training mit Validierungs-Set zur Leistungsbewertung.
3. Speichern von Modell UND Preprocessor.
4. Bereitstellung einer Vorhersage-Funktion für Echtzeit-Anwendungen.
"""
# Notwendige Importe für dieses Modul
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Korrekter relativer Import der Konfigurationsdatei
from . import config

def train_model(features_df: pd.DataFrame):
    """
    Führt den vollständigen Trainings- und Validierungs-Workflow durch.
    """
    print("Starte Trainings- und Validierungs-Workflow...")

    # Vorbereitung der Daten
    X = features_df.drop(columns=['label'])
    y = features_df['label']

    # 1. Aufteilung in Trainings- und Validierungsset
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=config.VALIDATION_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )
    print(f"Trainingsdaten aufgeteilt: {len(X_train)} zum Trainieren, {len(X_val)} zum Validieren.")

    # 2. Skalierung der Merkmale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val) # Wichtig: Validierungsdaten nur transformieren!

    # Speichere den trainierten Scaler
    joblib.dump(scaler, config.SAVED_PREPROCESSOR_PATH)
    print(f"Preprocessor (Scaler) wurde unter '{config.SAVED_PREPROCESSOR_PATH}' gespeichert.")

    # 3. Training und Validierung
    model = RandomForestClassifier(
        n_estimators=config.N_ESTIMATORS,
        max_depth=config.MAX_DEPTH,
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    # Bewertung auf dem Validierungs-Set
    y_pred_val = model.predict(X_val_scaled)
    val_report = classification_report(y_val, y_pred_val, zero_division=0)
    print("\n--- Klassifikationsbericht für das VALIDIERUNGS-Set ---")
    print(val_report)
    with open(config.VALIDATION_CLASSIFICATION_REPORT_PATH, 'w') as f:
        f.write("Validierungs-Klassifikationsbericht\n" + "="*40 + "\n" + val_report)

    # 4. Finales Training und Speichern des endgültigen Modells
    print("\nTrainiere finales Modell auf dem *gesamten* Trainingsdatensatz...")
    final_scaler = StandardScaler().fit(X)
    X_scaled_full = final_scaler.transform(X)
    
    final_model = RandomForestClassifier(
        n_estimators=config.N_ESTIMATORS,
        max_depth=config.MAX_DEPTH,
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    ).fit(X_scaled_full, y)
    
    joblib.dump(final_model, config.SAVED_MODEL_PATH)
    joblib.dump(final_scaler, config.SAVED_PREPROCESSOR_PATH) # Finalen Scaler für Inferenz speichern
    joblib.dump(X.columns.to_list(), config.SAVED_MODEL_PATH.with_suffix('.columns'))
    print(f"Finales Modell und Preprocessor wurden unter '{config.SAVED_MODEL_PATH}' gespeichert.")


def predict_single_instance(feature_df: pd.DataFrame) -> tuple:
    """
    Macht eine Vorhersage für einen einzelnen, neuen Merkmalsvektor.
    """
    model = joblib.load(config.SAVED_MODEL_PATH)
    preprocessor = joblib.load(config.SAVED_PREPROCESSOR_PATH)
    train_columns = joblib.load(config.SAVED_MODEL_PATH.with_suffix('.columns'))
    
    feature_df_aligned = feature_df[train_columns]
    scaled_features = preprocessor.transform(feature_df_aligned)
    
    prediction = model.predict(scaled_features)
    probabilities = model.predict_proba(scaled_features)
    
    return prediction[0], probabilities[0]


def evaluate_on_test_data(test_features_df: pd.DataFrame):
    """
    Lädt das gespeicherte Modell und evaluiert es auf dem ungesehenen Testdatensatz.
    """
    print("\nStarte finale Evaluierung auf dem separaten Testdatensatz...")
    model = joblib.load(config.SAVED_MODEL_PATH)
    preprocessor = joblib.load(config.SAVED_PREPROCESSOR_PATH)
    train_columns = joblib.load(config.SAVED_MODEL_PATH.with_suffix('.columns'))
    
    X_test = test_features_df[train_columns]
    y_test = test_features_df['label']
    
    X_test_scaled = preprocessor.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    report = classification_report(y_test, y_pred, zero_division=0)
    print("--- Finaler Klassifikationsbericht für den TEST-Datensatz ---")
    print(report)
    with open(config.TEST_CLASSIFICATION_REPORT_PATH, 'w') as f:
        f.write("Finaler Test-Klassifikationsbericht\n" + "="*40 + "\n" + report)