# src/feature_engineering/time_domain.py
"""
Modul zur Extraktion von Zeitbereichs-Merkmalen.
Basierend auf der Empfehlung des Berichts: RMS, Kurtosis, Crest-Faktor.
"""
import numpy as np
from scipy.stats import kurtosis

def extract_time_domain_features(signal: np.ndarray) -> dict:
    """
    Berechnet ein Set von statistischen Merkmalen direkt aus dem Zeitsignal.

    Args:
        signal (np.ndarray): Das 1D-Vibrationssignal.

    Returns:
        dict: Ein Dictionary mit den berechneten Merkmalen.
    """
    if signal.size == 0:
        return {'td_rms': 0, 'td_kurtosis': 0, 'td_crest_factor': 0}

    rms = np.sqrt(np.mean(signal**2))
    peak = np.max(np.abs(signal))
    
    # Crest-Faktor: Verhältnis von Spitze zu RMS. Sehr sensitiv für Impulse.
    # Vermeide Division durch Null bei einem leeren oder Null-Signal.
    crest_factor = peak / rms if rms > 0 else 0
    
    features = {
        'td_rms': rms,
        'td_kurtosis': kurtosis(signal), # Misst die "Spitzigkeit" der Verteilung
        'td_crest_factor': crest_factor
    }
    return features