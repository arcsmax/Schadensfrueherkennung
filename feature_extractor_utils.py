# feature_extractor_utils.py

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fft import fft, fftfreq

# Diese Datei enthält die Funktionen zur Merkmalsextraktion,
# die von der Haupt-Notebook-Datei aufgerufen werden, um die Parallelisierung
# robust und plattformunabhängig zu gestalten.

def get_time_domain_features(segment: pd.Series) -> dict:
    """
    Berechnet statistische Merkmale im Zeitbereich für ein gegebenes Signal-Segment.
    """
    features = {}
    keys = ['mean', 'std', 'rms', 'skewness', 'kurt', 'peak', 'peak_to_peak', 'crest_factor', 'shape_factor', 'clearance_factor']
    for k in keys: # Initialisiere alle Merkmale mit NaN für Robustheit
        features[k] = np.nan

    if segment.empty or not pd.api.types.is_numeric_dtype(segment) or segment.isnull().all():
        return features

    data = segment.dropna().values
    if data.size == 0:
        return features
        
    features['mean'] = np.mean(data)
    features['std'] = np.std(data)
    features['rms'] = np.sqrt(np.mean(data**2))
    features['skewness'] = skew(data)
    features['kurt'] = kurtosis(data) # Exzess-Kurtosis (Kurtosis von Normalverteilung = 0)
    
    peak_val = np.max(np.abs(data)) if data.size > 0 else 0
    features['peak'] = peak_val
    features['peak_to_peak'] = np.max(data) - np.min(data) if data.size > 0 else 0
    
    if features['rms'] != 0 and not np.isnan(features['rms']):
        features['crest_factor'] = peak_val / features['rms']
    
    mean_abs = np.mean(np.abs(data))
    if mean_abs != 0 and not np.isnan(mean_abs):
        if not np.isnan(features['rms']):
             features['shape_factor'] = features['rms'] / mean_abs
        
        # Für Clearance Factor, stelle sicher, dass der Nenner nicht Null oder NaN ist
        sqrt_abs_mean_sq = np.mean(np.sqrt(np.abs(data)))**2
        if sqrt_abs_mean_sq != 0 and not np.isnan(sqrt_abs_mean_sq):
            features['clearance_factor'] = peak_val / sqrt_abs_mean_sq
            
    return features

def get_frequency_domain_features(
    segment: pd.Series,
    sampling_rate: float,
    char_frequencies_for_segment: dict = None
) -> dict:
    """
    Berechnet Merkmale im Frequenzbereich für ein gegebenes Signal-Segment.
    """
    features = {}
    keys = ['spectral_centroid', 'spectral_std', 'gmf_band_energy_rel']
    for k in keys: # Initialisiere mit NaN
        features[k] = np.nan

    N = len(segment)
    if N < 2 or not pd.api.types.is_numeric_dtype(segment) or segment.isnull().all():
        return features

    data = segment.dropna().values
    if data.size < 2:
        return features
    
    N_actual = len(data)
    yf = fft(data)
    magnitude_spectrum = 2.0/N_actual * np.abs(yf[0:N_actual//2])
    frequencies = fftfreq(N_actual, 1/sampling_rate)[0:N_actual//2]

    if frequencies.size == 0:
        return features

    sum_magnitude_spectrum = np.sum(magnitude_spectrum)
    if sum_magnitude_spectrum == 0 or np.isnan(sum_magnitude_spectrum):
        features['spectral_centroid'] = 0.0 if sum_magnitude_spectrum == 0 else np.nan
        features['spectral_std'] = 0.0 if sum_magnitude_spectrum == 0 else np.nan
        return features
        
    features['spectral_centroid'] = np.sum(frequencies * magnitude_spectrum) / sum_magnitude_spectrum
    
    weighted_variance = np.sum(((frequencies - features['spectral_centroid'])**2) * magnitude_spectrum) / sum_magnitude_spectrum
    features['spectral_std'] = np.sqrt(weighted_variance) if weighted_variance >= 0 else np.nan
        
    total_energy = np.sum(magnitude_spectrum**2)
    if char_frequencies_for_segment and 'gmf_hz' in char_frequencies_for_segment and total_energy > 0 and not np.isnan(total_energy):
        gmf = char_frequencies_for_segment.get('gmf_hz')
        if pd.notna(gmf) and gmf > 0:
            band_width_hz = max(10.0, 0.05 * gmf)
            lower_bound = gmf - band_width_hz / 2
            upper_bound = gmf + band_width_hz / 2
            
            idx_in_band = np.where((frequencies >= lower_bound) & (frequencies <= upper_bound))[0]
            if idx_in_band.size > 0:
                gmf_band_energy = np.sum(magnitude_spectrum[idx_in_band]**2)
                features['gmf_band_energy_rel'] = gmf_band_energy / total_energy
            else:
                features['gmf_band_energy_rel'] = 0.0 # Band gefunden, aber keine Energie darin
        else:
             features['gmf_band_energy_rel'] = 0.0 # GMF war nicht gültig
            
    return features

def process_single_file_features_external(
    filename: str,
    full_raw_data_df: pd.DataFrame,
    vibration_cols: list,
    sampling_rate: float,
    char_freqs_dict_for_file: dict
) -> dict:
    """
    Extrahiert Merkmale für eine einzelne Datei. Diese Funktion ist die "Worker"-Funktion,
    die von jedem parallelen Prozess ausgeführt wird.
    """
    try:
        file_segment_df = full_raw_data_df[full_raw_data_df['filename'] == filename]
        if file_segment_df.empty:
            return {'filename': filename, 'error': 'Segment not found in DataFrame'}

        meta_row = file_segment_df.iloc[0]
        current_features = {
            'filename': filename,
            'pitting_level': meta_row['pitting_level'],
            'rpm_input': meta_row['rpm_input'],
            'torque_output_nm': meta_row['torque_output_nm'],
            'repetition': meta_row.get('repetition', np.nan)
        }

        for sensor_col in vibration_cols:
            sensor_segment = file_segment_df[sensor_col]
            
            time_feats = get_time_domain_features(sensor_segment)
            for key, value in time_feats.items():
                current_features[f'{sensor_col}_time_{key}'] = value
            
            freq_feats = get_frequency_domain_features(sensor_segment, sampling_rate, char_freqs_dict_for_file)
            for key, value in freq_feats.items():
                current_features[f'{sensor_col}_freq_{key}'] = value
        
        return current_features
    except Exception as e:
        # Gib einen Fehler im Ergebnis-Dictionary zurück, anstatt den Prozess abstürzen zu lassen
        return {'filename': filename, 'error': str(e)}