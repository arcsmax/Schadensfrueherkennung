# src/data_processing.py
"""
Modul zur Erfassung und Validierung von Metadaten aus der PHM-Datenstruktur.
Dieses Modul scannt die Rohdatenverzeichnisse und erstellt eine zentrale
Metadaten-Datei, die als Grundlage für die nachfolgende Merkmalsextraktion dient.
"""
import os
import re
import pandas as pd
from . import config

def parse_filename(filename: str) -> dict:
    match = re.match(r'V(\d+)_(\d+)N_(\d+)\.txt', filename, re.IGNORECASE)
    if match: return {'rpm_input': int(match.group(1)), 'torque_output_nm': int(match.group(2)), 'repetition': int(match.group(3))}
    return {}

def parse_folder_name(folder_name: str) -> int:
    match = re.search(r'_level_(\d+)', folder_name, re.IGNORECASE)
    if match: return int(match.group(1))
    return -1

def create_metadata_file(source_path, output_path, marker_path) -> pd.DataFrame:
    """
    Scannt ein Quellverzeichnis, extrahiert Metadaten und speichert sie.
    Funktioniert jetzt für beliebige Pfade.
    """
    print(f"Starte Metadaten-Erfassung für Pfad: {source_path}")
    marker_path.unlink(missing_ok=True)

    if not source_path.is_dir():
        raise FileNotFoundError(f"Das angegebene Quellverzeichnis existiert nicht: {source_path}")

    metadata_records = []
    for folder_name in sorted(os.listdir(source_path)):
        folder_path = os.path.join(source_path, folder_name)
        if not os.path.isdir(folder_path):
            continue

        pitting_level = parse_folder_name(folder_name)
        if pitting_level == -1:
            print(f"INFO: Ordner '{folder_name}' wird übersprungen (Schädigungsgrad nicht erkannt).")
            continue

        print(f"  -> Scanne Ordner für Level {pitting_level}: {folder_name}")
        for filename in sorted(os.listdir(folder_path)):
            if not filename.lower().endswith('.txt'):
                continue
            
            file_info = parse_filename(filename)
            if file_info:
                record = {
                    'pitting_level': pitting_level,
                    'rpm_input': file_info['rpm_input'],
                    'torque_output_nm': file_info['torque_output_nm'],
                    'repetition': file_info['repetition'],
                    'filepath': os.path.join(folder_path, filename)
                }
                metadata_records.append(record)

    if not metadata_records:
        raise RuntimeError(f"Keine validen .txt-Dateien im Quellverzeichnis gefunden: {source_path}")

    metadata_df = pd.DataFrame(metadata_records)
    metadata_df.to_csv(output_path, index=False)
    marker_path.touch()
    print(f"Metadaten-Datei wurde unter '{output_path}' gespeichert.")
    return metadata_df