# src/feature_engineering/main_extractor.py
"""
Haupt-Orchestrierungsmodul, erweitert um eine Funktion zur
Verarbeitung einzelner Instanzen für die Echtzeit-Vorhersage.
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from .time_domain import extract_time_domain_features
from .time_frequency import extract_dwt_features
from .order_analysis import extract_order_analysis_features
from .. import config

def extract_features_for_single_instance(signal: np.ndarray, tacho_signal: np.ndarray) -> pd.DataFrame:
    """
    Führt die gesamte Kette der Merkmalsextraktion für ein einzelnes
    Rohsignal-Array aus. Dies simuliert eine Echtzeit-Anwendung.

    Args:
        signal (np.ndarray): Das 1D-Vibrationssignal (z.B. aus der x-Achse).
        tacho_signal (np.ndarray): Das zugehörige Tachosignal.

    Returns:
        pd.DataFrame: Ein DataFrame mit einer Zeile, die alle extrahierten Merkmale enthält.
    """
    # Extrahiere Merkmale aus allen Domänen
    time_features = extract_time_domain_features(signal)
    dwt_features = extract_dwt_features(signal)
    order_features = extract_order_analysis_features(signal, tacho_signal)
    
    # Kombiniere alle Merkmale zu einem einzigen Dictionary
    combined_features = {}
    combined_features.update(time_features)
    combined_features.update(dwt_features)
    combined_features.update(order_features)
    
    # Konvertiere das Dictionary in einen DataFrame mit einer Zeile
    return pd.DataFrame([combined_features])


def extract_features_for_all_files(metadata_df: pd.DataFrame, output_path, marker_path) -> pd.DataFrame:
    """
    Bestehende Funktion für die Batch-Verarbeitung (unverändert).
    Iteriert durch die Metadaten, lädt jede Datei einzeln, extrahiert das
    vollständige Set an Merkmalen und fasst sie zusammen.
    """
    print(f"\nStarte Batch-Merkmalsextraktion. Ziel: {output_path}")
    marker_path.unlink(missing_ok=True)

    all_features_list = []
    
    for _, row in tqdm(metadata_df.iterrows(), total=metadata_df.shape[0], desc="Extrahiere Merkmale"):
        try:
            data = pd.read_csv(row['filepath'], sep=r'\s+', header=None,
                               names=['acc_x', 'acc_y', 'acc_z', 'tacho'], engine='python')
            
            signal = data['acc_x'].values
            tacho_signal = data['tacho'].values
            
            # Hier könnten wir die neue Single-Instance-Funktion wiederverwenden,
            # aber zur Klarheit behalten wir die separate Logik.
            time_features = extract_time_domain_features(signal)
            dwt_features = extract_dwt_features(signal)
            order_features = extract_order_analysis_features(signal, tacho_signal)
            
            combined_features = row.to_dict()
            combined_features.update(time_features)
            combined_features.update(dwt_features)
            combined_features.update(order_features)
            
            all_features_list.append(combined_features)

        except Exception as e:
            print(f"WARNUNG: Datei {row['filepath']} konnte nicht verarbeitet werden. Fehler: {e}")

    final_features_df = pd.DataFrame(all_features_list).drop(columns=['filepath'], errors='ignore')
    final_features_df.to_csv(output_path, index=False)
    marker_path.touch()
    print(f"Merkmals-Datei wurde unter '{output_path}' gespeichert.")
    
    return final_features_df