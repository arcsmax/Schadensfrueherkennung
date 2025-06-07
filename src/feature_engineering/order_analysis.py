# src/feature_engineering/order_analysis.py
"""
Modul zur Durchführung der Ordnungsanalyse unter Nutzung eines Tachosignals.
Dies ist die robusteste Methode für variable Drehzahlen.
"""
import numpy as np
from scipy.fft import fft
from scipy.interpolate import interp1d
from .. import config

def get_angle_vector_from_tacho(tacho_signal: np.ndarray, fs: int) -> np.ndarray:
    """
    Wandelt ein Tacho-Signal (Pulse pro Umdrehung) in einen kontinuierlichen
    Phasenwinkel-Vektor um.
    """
    # Finde die Indizes der ansteigenden Flanken der Tacho-Pulse
    # Annahme: Ein Puls pro Umdrehung
    tacho_pulses = np.where((tacho_signal[:-1] < 0.5) & (tacho_signal[1:] >= 0.5))[0]
    
    if len(tacho_pulses) < 2:
        # Nicht genügend Pulse für eine verlässliche Analyse
        return None

    # Lineare Interpolation der Phasenwinkel (0 bis 2*pi) zwischen den Pulsen
    pulse_times = tacho_pulses / fs
    pulse_angles = np.arange(len(tacho_pulses)) * 2 * np.pi
    
    # Erstelle einen Zeitvektor für das gesamte Signal
    time_vector = np.arange(len(tacho_signal)) / fs
    
    # Interpoliere, um für jeden Zeitpunkt den Winkel zu erhalten
    angle_function = interp1d(pulse_times, pulse_angles, kind='linear', fill_value="extrapolate")
    angle_vector = angle_function(time_vector)
    
    return angle_vector

def resample_to_angle_domain(signal: np.ndarray, angle_vector: np.ndarray, angle_resolution_rad: float):
    """
    Tastet das Zeitsignal auf ein äquidistantes Winkelgitter um (dynamisches Resampling).
    """
    # Erstelle ein neues, gleichmäßiges Winkelgitter
    max_angle = np.max(angle_vector)
    angle_grid = np.arange(0, max_angle, angle_resolution_rad)
    
    # Interpoliere die Signalwerte auf das neue Winkelgitter
    resampling_function = interp1d(angle_vector, signal, kind='linear', bounds_error=False, fill_value=0)
    resampled_signal = resampling_function(angle_grid)
    
    return resampled_signal

def extract_order_analysis_features(signal: np.ndarray, tacho_signal: np.ndarray) -> dict:
    """
    Führt die vollständige Ordnungsanalyse durch und extrahiert die empfohlenen Merkmale.
    """
    angle_vector = get_angle_vector_from_tacho(tacho_signal, config.SAMPLING_RATE)
    if angle_vector is None:
        return {} # Tachosignal unbrauchbar

    # Resampling in den Winkelbereich
    angle_resolution_rad = np.deg2rad(config.ORDER_RESOLUTION)
    angle_resampled_signal = resample_to_angle_domain(signal, angle_vector, angle_resolution_rad)
    
    # FFT auf dem winkel-resampelten Signal, um das Ordnungsspektrum zu erhalten
    N = len(angle_resampled_signal)
    order_spectrum = np.abs(fft(angle_resampled_signal)[:N//2]) * 2 / N
    
    # Erstelle die Ordnungsachse
    order_step = 1 / (angle_resolution_rad * N) * (2 * np.pi)
    orders = np.arange(N//2) * order_step
    
    # Extrahiere Merkmale aus dem Ordnungsspektrum
    features = {}
    
    # 1. Amplitude der GMF-Ordnung und ihrer Harmonischen
    for i in range(1, 4): # GMF und 2 Harmonische
        target_order = config.GMF_ORDER * i
        # Finde den Index, der der Zielordnung am nächsten ist
        idx = np.argmin(np.abs(orders - target_order))
        features[f'oa_gmf_harmonic_{i}_amp'] = order_spectrum[idx]

    # 2. Energie in den Seitenband-Regionen
    # Seitenbänder im Abstand der 1. Ordnung (Referenzwelle)
    sideband_width_order = 0.5 # Breite des zu analysierenden Bandes
    for i in range(1, 3): # Seitenbänder um GMF und 2. Harmonische
        gmf_harmonic = config.GMF_ORDER * i
        # Oberes Seitenband
        upper_band_mask = (orders >= gmf_harmonic + 1 - sideband_width_order) & (orders <= gmf_harmonic + 1 + sideband_width_order)
        features[f'oa_gmf_h{i}_upper_sb_energy'] = np.sum(order_spectrum[upper_band_mask]**2)
        # Unteres Seitenband
        lower_band_mask = (orders >= gmf_harmonic - 1 - sideband_width_order) & (orders <= gmf_harmonic - 1 + sideband_width_order)
        features[f'oa_gmf_h{i}_lower_sb_energy'] = np.sum(order_spectrum[lower_band_mask]**2)
        
    return features