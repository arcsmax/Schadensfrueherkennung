# main.py
"""
Hauptskript, das den neuen professionellen Workflow orchestriert:
1. Training und Validierung auf dem Trainingsdatensatz.
2. Demonstration einer Echtzeit-Vorhersage für eine einzelne Instanz.
3. Finale, unvoreingenommene Evaluierung auf dem separaten Testdatensatz.
"""
import pandas as pd
import numpy as np
from src.data_processing import create_metadata_file
from src.feature_engineering import extract_features_for_all_files, extract_features_for_single_instance
from src.model_training import train_model, evaluate_on_test_data, predict_single_instance
from src.utils import plot_feature_distribution
from src import config

def run_training_pipeline():
    """Führt alle Schritte aus, um das Modell zu trainieren und zu speichern."""
    print("===== Starte Trainings-Pipeline =====")
    
    # Schritt A & B: Daten und Merkmale für Trainingsdaten laden/erstellen
    if config.TRAIN_METADATA_MARKER.exists():
        train_metadata_df = pd.read_csv(config.TRAIN_METADATA_PATH)
    else:
        train_metadata_df = create_metadata_file(config.TRAIN_SOURCE_DATA_PATH, config.TRAIN_METADATA_PATH, config.TRAIN_METADATA_MARKER)

    if config.TRAIN_FEATURES_MARKER.exists():
        train_features_df = pd.read_csv(config.TRAIN_PROCESSED_FEATURES_PATH)
    else:
        train_features_df = extract_features_for_all_files(train_metadata_df, config.TRAIN_PROCESSED_FEATURES_PATH, config.TRAIN_FEATURES_MARKER)
    
    if 'pitting_level' in train_features_df.columns:
        train_features_df.rename(columns={'pitting_level': 'label'}, inplace=True)
    
    # Schritt C: Visualisierung
    plot_feature_distribution(train_features_df.copy(), config.TRAIN_VISUALIZATION_PATH)

    # Schritt D: Modelltraining mit Validierung
    columns_to_drop = ['rpm_input', 'torque_output_nm', 'repetition', 'status']
    model_input_df = train_features_df.drop(columns=columns_to_drop, errors='ignore')
    train_model(model_input_df)
    
    print("===== Trainings-Pipeline abgeschlossen =====")

def demonstrate_real_time_prediction():
    """
    Demonstriert, wie eine einzelne, neue Messung klassifiziert wird.
    """
    print("\n===== Demonstration: Echtzeit-Vorhersage =====")
    # Simuliere das Eintreffen neuer Sensordaten
    # Wir nehmen dafür einfach die erste Datei aus unserem Test-Set
    print("Simuliere Eintreffen einer neuen Messung...")
    test_meta_df = pd.read_csv(config.TEST_METADATA_PATH)
    sample_filepath = test_meta_df.iloc[0]['filepath']
    
    print(f"Lade Rohdaten von: {sample_filepath}")
    data = pd.read_csv(sample_filepath, sep=r'\s+', header=None,
                       names=['acc_x', 'acc_y', 'acc_z', 'tacho'], engine='python')
    signal = data['acc_x'].values
    tacho_signal = data['tacho'].values

    # 1. Extrahiere Merkmale für diese eine Messung
    print("Extrahiere Merkmale für die neue Messung...")
    feature_vector_df = extract_features_for_single_instance(signal, tacho_signal)
    
    # 2. Mache eine Vorhersage
    print("Mache Vorhersage mit dem geladenen Modell und Preprocessor...")
    prediction, probabilities = predict_single_instance(feature_vector_df)
    
    # 3. Gib das Ergebnis aus
    class_names = [f"Level {i}" for i in sorted(test_meta_df['pitting_level'].unique())] # Beispielhafte Namen
    print("\n--- Vorhersage-Ergebnis ---")
    print(f"Vorhergesagter Schadensgrad: Level {prediction}")
    print("Wahrscheinlichkeiten pro Klasse:")
    for i, prob in enumerate(probabilities):
        print(f"  - Klasse {i}: {prob:.2%}")
    print("--------------------------")


def run_testing_pipeline():
    # Diese Funktion bleibt für die finale Evaluierung des gesamten Test-Sets zuständig.
    print("\n===== Starte FINALE Test-Pipeline =====")
    if not config.TEST_PROCESSED_FEATURES_PATH.exists():
        print("Test-Features nicht gefunden. Erstelle sie zuerst...")
        test_metadata_df = create_metadata_file(config.TEST_SOURCE_DATA_PATH, config.TEST_METADATA_PATH, config.TEST_METADATA_MARKER)
        extract_features_for_all_files(test_metadata_df, config.TEST_PROCESSED_FEATURES_PATH, config.TEST_FEATURES_MARKER)

    test_features_df = pd.read_csv(config.TEST_PROCESSED_FEATURES_PATH)
    if 'pitting_level' in test_features_df.columns:
        test_features_df.rename(columns={'pitting_level': 'label'}, inplace=True)
    
    evaluate_on_test_data(test_features_df)
    print("===== Test-Pipeline abgeschlossen =====")

if __name__ == "__main__":
    run_training_pipeline()
    demonstrate_real_time_prediction()
    run_testing_pipeline()