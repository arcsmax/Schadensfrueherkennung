# src/data_processing.py
"""
Modul zur Erfassung und Validierung von Metadaten aus der PHM-Datenstruktur.
Dieses Modul scannt die Rohdatenverzeichnisse und erstellt eine zentrale
Metadaten-Datei, die als Grundlage für die nachfolgende Merkmalsextraktion dient.
"""
import os
import re
import logging
import pandas as pd
from pathlib import Path

# Korrekter relativer Import der Konfigurationsdatei und Hilfsfunktionen
from . import config
from .utils import clear_cache

def _parse_filename(filename: str) -> dict:
    """Extrahiert Metadaten (Drehzahl, Drehmoment, Wiederholung) aus dem Dateinamen."""
    # KORREKTUR: Regex mit einfachen Backslashes in einem Raw-String
    match = re.match(r'V(\d+)_(\d+)N_(\d+)\.txt', filename, re.IGNORECASE)
    if match:
        return {
            'rpm_input': int(match.group(1)),
            'torque_output_nm': int(match.group(2)),
            'repetition': int(match.group(3))
        }
    return {}

def _parse_folder_name(folder_name: str) -> int:
    """Extrahiert das Schädigungslevel aus dem Ordnernamen."""
    match = re.search(r'_level_(\d+)', folder_name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return -1

def _create_metadata_for_path(source_path: Path, output_path: Path) -> bool:
    """
    Worker-Funktion: Scannt ein Quellverzeichnis, extrahiert Metadaten und speichert sie.
    Gibt True bei Erfolg und False bei einem Fehler zurück.
    """
    logging.info(f"Starte Metadaten-Erfassung für den Pfad: {source_path}")

    if not source_path.is_dir():
        logging.error(f"Das angegebene Quellverzeichnis existiert nicht: {source_path}")
        return False

    metadata_records = []
    # Verwende Path.iterdir() für eine saubere, OS-unabhängige Iteration
    for folder_path in sorted(source_path.iterdir()):
        if not folder_path.is_dir():
            continue

        pitting_level = _parse_folder_name(folder_path.name)
        if pitting_level == -1:
            logging.warning(f"Ordner '{folder_path.name}' wird übersprungen (Schädigungsgrad nicht erkannt).")
            continue

        logging.debug(f"Scanne Ordner für Level {pitting_level}: {folder_path.name}")
        for file_path in sorted(folder_path.glob('*.txt')):
            file_info = _parse_filename(file_path.name)
            if file_info:
                record = {
                    'pitting_level': pitting_level,
                    'rpm_input': file_info['rpm_input'],
                    'torque_output_nm': file_info['torque_output_nm'],
                    'repetition': file_info['repetition'],
                    'filepath': str(file_path.resolve()) # Speichere den absoluten Pfad
                }
                metadata_records.append(record)

    if not metadata_records:
        logging.error(f"Keine validen .txt-Dateien im Quellverzeichnis gefunden: {source_path}")
        return False

    metadata_df = pd.DataFrame(metadata_records)
    
    # Stelle sicher, dass das Zielverzeichnis existiert
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    metadata_df.to_csv(output_path, index=False)
    logging.info(f"Metadaten erfolgreich erstellt und in '{output_path}' gespeichert. {len(metadata_df)} Einträge gefunden.")
    return True

def run_data_processing_pipeline() -> bool:
    """
    Hauptfunktion zur Orchestrierung der Datenverarbeitungsschritte.
    Erstellt die Metadaten-Dateien für Trainings- und Testdaten.
    """
    logging.info("="*50)
    logging.info("Schritt 1: Datenverarbeitung (Metadaten-Erstellung)")
    logging.info("="*50)
    
    # Wenn die Daten neu verarbeitet werden, müssen alle nachfolgenden Schritte
    # ebenfalls neu ausgeführt werden. Wir löschen daher ihre Cache-Marker.
    logging.info("Lösche abhängige Cache-Marker für 'feature_engineering' und 'model_training'.")
    clear_cache('feature_engineering')
    clear_cache('model_training')
    
    # Erstelle Metadaten für das Trainings-Set
    logging.info("Verarbeite Trainingsdaten...")
    train_success = _create_metadata_for_path(
        source_path=config.TRAIN_SOURCE_DATA_PATH,
        output_path=config.TRAIN_METADATA_PATH
    )
    
    # Optional: Erstelle Metadaten für das Test-Set, falls es existiert
    test_success = True # Standardmäßig auf True, falls kein Testset vorhanden
    if config.TEST_SOURCE_DATA_PATH.exists():
        logging.info("Verarbeite Testdaten...")
        test_success = _create_metadata_for_path(
            source_path=config.TEST_SOURCE_DATA_PATH,
            output_path=config.TEST_METADATA_PATH
        )
    else:
        logging.warning(f"Testdaten-Verzeichnis '{config.TEST_SOURCE_DATA_PATH}' nicht gefunden. Überspringe Metadaten-Erstellung für Testdaten.")
        
    is_successful = train_success and test_success
    
    if is_successful:
        logging.info("Datenverarbeitungs-Pipeline erfolgreich abgeschlossen.")
    else:
        logging.error("Fehler in der Datenverarbeitungs-Pipeline. Bitte Logs überprüfen.")
        
    return is_successful

# Dieser Block ermöglicht es, das Skript bei Bedarf auch direkt auszuführen,
# um schnell zu testen, ob die Metadaten-Erstellung funktioniert.
if __name__ == '__main__':
    from .utils import setup_logging
    setup_logging()
    run_data_processing_pipeline()
