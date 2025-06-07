# src/feature_engineering.py
"""
Modul zur iterativen und speichereffizienten Merkmalsextraktion.
Liest die Metadaten und verarbeitet jede Rohsignal-Datei einzeln,
um den Arbeitsspeicher zu schonen.
"""
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.stats import kurtosis, skew
from tqdm import tqdm  # Fortschrittsanzeige für lange Prozesse
from . import config

def calculate_fft(signal: np.ndarray, sampling_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Berechnet die Fast-Fourier-Transformation (FFT) eines Signals."""
    n_points = len(signal)
    freqs = fftfreq(n_points, 1 / sampling_rate)[:n_points // 2]
    fft_vals = fft(signal)
    amplitudes = 2.0 / n_points * np.abs(fft_vals[0:n_points // 2])
    return freqs, amplitudes

def extract_features_from_signal(signal: np.ndarray) -> dict:
    """Extrahiert einen Satz von Merkmalen aus einem einzelnen Signal-Array."""
    features = {
        'mean': np.mean(signal),
        'std': np.std(signal),
        'skew': skew(signal),
        'kurtosis': kurtosis(signal),
        'peak': np.max(np.abs(signal)),
        'rms': np.sqrt(np.mean(signal**2)) # Root Mean Square
    }
    
    freqs, amplitudes = calculate_fft(signal, config.SAMPLING_RATE)
    for band_name, (low, high) in config.FREQUENCY_BANDS.items():
        band_mask = (freqs >= low) & (freqs <= high)
        features[f'energy_{band_name}'] = np.sum(amplitudes[band_mask]**2)
        
    return features

def create_feature_dataset(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Iteriert durch die Metadaten, lädt jede Datei einzeln, extrahiert Merkmale
    und fasst die Ergebnisse in einem finalen DataFrame zusammen.
    """
    print("\nStarte Schritt B: Iterative Merkmalsextraktion...")
    config.FEATURE_EXTRACTION_SUCCESS_MARKER.unlink(missing_ok=True)

    all_features_list = []
    
    # tqdm sorgt für eine schöne Fortschrittsanzeige in der Konsole
    for index, row in tqdm(metadata_df.iterrows(), total=metadata_df.shape[0], desc="Verarbeite Dateien"):
        try:
            # Lade das Rohsignal nur für diese eine Iteration
            signal_df = pd.read_csv(row['filepath'], sep=r'\s+', header=None, usecols=[0, 1, 2],
                                    names=['acc_x', 'acc_y', 'acc_z'], engine='python')
            
            # Merkmale für jede Achse extrahieren
            # Wir nehmen hier als Beispiel die x-Achse. In der Praxis könnte man
            # für jede Achse Merkmale extrahieren und sie benennen (z.B. 'mean_x', 'mean_y').
            signal_array = signal_df['acc_x'].values
            
            features = extract_features_from_signal(signal_array)
            
            # Füge die Metadaten zu den extrahierten Merkmalen hinzu
            features['pitting_level'] = row['pitting_level']
            features['rpm_input'] = row['rpm_input']
            features['torque_output_nm'] = row['torque_output_nm']
            
            all_features_list.append(features)

        except Exception as e:
            print(f"WARNUNG: Konnte Datei {row['filepath']} nicht verarbeiten. Fehler: {e}")

    if not all_features_list:
        raise RuntimeError("Merkmalsextraktion fehlgeschlagen. Keine Features konnten erstellt werden.")

    # Kombiniere die Liste von Dictionaries in einen finalen DataFrame
    final_features_df = pd.DataFrame(all_features_list)
    final_features_df.to_csv(config.PROCESSED_FEATURES_PATH, index=False)
    config.FEATURE_EXTRACTION_SUCCESS_MARKER.touch()

    print(f"\nSchritt B abgeschlossen: Merkmale für {len(final_features_df)} Signale extrahiert.")
    print(f"Feature-Datei wurde unter '{config.PROCESSED_FEATURES_PATH}' gespeichert.")
    
    return final_features_df