# src/feature_orchestrator.py
"""
Orchestriert den gesamten Prozess der Merkmalsextraktion.
Dieses Modul liest die Metadaten-Datei und wendet die verschiedenen
Extraktionsfunktionen parallel auf alle Rohdatendateien an, um die
Verarbeitungszeit drastisch zu reduzieren.
"""
import logging
import pandas as pd
from joblib import Parallel, delayed

from . import config
from .feature_engineering.time_domain import extract_time_domain_features
from .feature_engineering.time_frequency import extract_dwt_features
from .feature_engineering.order_analysis import extract_order_analysis_features

def _process_single_file(row: pd.Series) -> dict:
    """
    Worker-Funktion zur Verarbeitung einer einzelnen Zeile aus der Metadaten-Datei.
    Wird von jedem parallelen Prozess ausgeführt.
    """
    try:
        data = pd.read_csv(row['filepath'], sep=r'\s+', header=None,
                           names=['acc_x', 'acc_y', 'acc_z', 'tacho'], engine='python')
        
        signal = data['acc_x'].values
        tacho_signal = data['tacho'].values

        time_features = extract_time_domain_features(signal)
        dwt_features = extract_dwt_features(signal)
        order_features = extract_order_analysis_features(signal, tacho_signal)
        
        combined_features = row.to_dict()
        combined_features.update(time_features)
        combined_features.update(dwt_features)
        combined_features.update(order_features)
        
        # 'pitting_level' ist in Testdaten nicht immer vorhanden.
        # Hier wird davon ausgegangen, dass diese Funktion nur für Trainingsdaten
        # mit bekanntem Label aufgerufen wird, aber eine Prüfung schadet nicht.
        if 'pitting_level' not in combined_features:
            combined_features['pitting_level'] = -1

        return combined_features

    except Exception as e:
        # Logge den Fehler, aber unterbreche nicht den gesamten Prozess
        logging.error(f"Fehler bei der Verarbeitung von {row['filepath']}: {e}")
        return None

def run_feature_engineering_pipeline(is_test_data: bool = False) -> bool:
    """
    Hauptfunktion für Schritt 2 der Pipeline. Führt die Merkmalsextraktion parallel durch.
    """
    context = "TEST" if is_test_data else "TRAININGS"
    logging.info(f"Starte paralleles Feature Engineering für {context}-Daten...")
    
    metadata_path = config.TEST_METADATA_PATH if is_test_data else config.TRAIN_METADATA_PATH
    output_path = config.TEST_PROCESSED_FEATURES_PATH if is_test_data else config.TRAIN_PROCESSED_FEATURES_PATH

    try:
        metadata_df = pd.read_csv(metadata_path)
    except FileNotFoundError:
        logging.error(f"Metadaten-Datei '{metadata_path}' nicht gefunden. Bitte den vorherigen Schritt ausführen.")
        return False

    # Parallele Ausführung mit joblib.
    # n_jobs=-1 bedeutet: Nutze alle verfügbaren CPU-Kerne.
    # verbose=10 gibt detaillierten Fortschritt aus.
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(_process_single_file)(row) for _, row in metadata_df.iterrows()
    )

    # Filtere alle 'None'-Ergebnisse heraus, die bei Fehlern entstanden sind
    all_features_list = [res for res in results if res is not None]

    if not all_features_list:
        logging.error("Merkmalsextraktion fehlgeschlagen. Keine Features konnten erstellt werden.")
        return False

    final_features_df = pd.DataFrame(all_features_list).drop(columns=['filepath'], errors='ignore')
    final_features_df.to_csv(output_path, index=False)
    logging.info(f"Feature Engineering erfolgreich. {len(final_features_df)} Feature-Vektoren gespeichert in: {output_path}")
    
    return True
