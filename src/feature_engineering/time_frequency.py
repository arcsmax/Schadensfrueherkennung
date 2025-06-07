# src/feature_engineering/time_frequency.py
"""
Modul zur Extraktion von Zeit-Frequenz-Merkmalen mittels Diskreter Wavelet Transformation (DWT).
"""
import numpy as np
import pywt
from .. import config # Importiere Konfiguration aus dem übergeordneten src-Paket

def extract_dwt_features(signal: np.ndarray) -> dict:
    """
    Führt eine DWT durch und extrahiert Energie- und Statistik-Merkmale
    aus den resultierenden Koeffizienten.

    Args:
        signal (np.ndarray): Das 1D-Vibrationssignal.

    Returns:
        dict: Ein Dictionary mit den DWT-Merkmalen pro Ebene.
    """
    try:
        # Führe die Wavelet-Dekomposition durch
        coeffs = pywt.wavedec(signal, config.DWT_WAVELET, level=config.DWT_LEVEL)
        
        features = {}
        # Iteriere durch die Koeffizienten jeder Ebene (Approximation und Details)
        for i, a in enumerate(coeffs):
            level_name = f'dwt_level_{i}'
            features[f'{level_name}_energy'] = np.sum(a**2)
            features[f'{level_name}_mean'] = np.mean(a)
            features[f'{level_name}_std'] = np.std(a)
            
        return features
    except Exception as e:
        print(f"WARNUNG: DWT-Extraktion fehlgeschlagen: {e}")
        return {} # Gib ein leeres Dict zurück, um die Pipeline nicht zu stoppen