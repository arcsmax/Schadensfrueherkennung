# src/data_processing.py
"""
Modul zur Erfassung und Validierung von Metadaten aus der PHM-Datenstruktur.
Dieses Modul scannt die Rohdatenverzeichnisse und erstellt eine zentrale
Metadaten-Datei, die als Grundlage für die nachfolgende Merkmalsextraktion dient.
"""
import re
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split # NEU: Import für stratifiziertes Sampling

# Korrekter relativer Import der Konfigurationsdatei und Hilfsfunktionen
from . import config
from .utils import clear_downstream_caches

def _parse_filename(filepath: Path) -> dict:
    """Extrahiert Metadaten (Drehzahl, Drehmoment, Wiederholung) aus dem Dateinamen."""
    match = re.match(r'V(\d+)_(\d+)N_(\d+)\.txt', filepath.name, re.IGNORECASE)
    if match:
        return {
            'rpm_input': int(match.group(1)),
            'torque_output_nm': int(match.group(2)),
            'repetition': int(match.group(3))
        }
    return None

def _parse_folder_name(folder_path: Path) -> int:
    """Extrahiert das Schädigungslevel aus dem Ordnernamen."""
    match = re.search(r'_level_(\d+)', folder_path.name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return -1

def run_data_processing_pipeline() -> bool:
    """
    Sammelt Metadaten aus den Rohdaten-Dateinamen und -Ordnern,
    wendet optional ein STRATIFIZIERTES Sampling an und speichert sie als CSV.
    """
    logging.info("="*50)
    logging.info("Schritt 1: Datenverarbeitung und Metadaten-Erstellung")
    logging.info("="*50)

    clear_downstream_caches('data_processing')

    source_path = config.TRAIN_SOURCE_DATA_PATH
    if not source_path.is_dir():
        logging.error(f"Das angegebene Quellverzeichnis existiert nicht: {source_path}")
        return False

    all_files = list(source_path.glob('**/*.txt'))
    if not all_files:
        logging.error(f"Keine .txt-Dateien im Quellverzeichnis gefunden: {source_path}")
        return False
        
    logging.info(f"Insgesamt {len(all_files)} Rohdatendateien gefunden.")

    # Schritt 1: Erstelle zuerst die VOLLSTÄNDIGEN Metadaten
    metadata_records = []
    for file_path in tqdm(all_files, desc="Analysiere Dateipfade"):
        file_info = _parse_filename(file_path)
        if file_info:
            pitting_level = _parse_folder_name(file_path.parent)
            if pitting_level != -1:
                file_info['pitting_level'] = pitting_level
                file_info['filepath'] = str(file_path.resolve())
                metadata_records.append(file_info)

    if not metadata_records:
        logging.error("Konnte keine validen Metadaten aus den Dateipfaden extrahieren.")
        return False
        
    metadata_df = pd.DataFrame(metadata_records)

    # Schritt 2: Wende STRATIFIZIERTES Sampling auf dem DataFrame an
    if config.DATA_FRACTION < 1.0:
        logging.warning(
            f"Reduziere Datensatz auf {config.DATA_FRACTION * 100:.0f}% "
            f"durch stratifiziertes Sampling, um die Klassenverteilung beizubehalten."
        )
        
        # Wir nutzen train_test_split, um eine stratifizierte Teilmenge zu erhalten.
        # Wir ignorieren den "Test"-Teil und behalten nur den "Train"-Teil.
        stratified_sample_df, _ = train_test_split(
            metadata_df,
            train_size=config.DATA_FRACTION,
            stratify=metadata_df['pitting_level'],
            random_state=config.RANDOM_STATE
        )
        metadata_df = stratified_sample_df.copy() # Verwende die Teilmenge für den Rest des Prozesses
        logging.info(f"Größe des Datensatzes nach Sampling: {len(metadata_df)} Einträge.")


    # Schritt 3: Speichere die finalen Metadaten (entweder voll oder als Teilmenge)
    metadata_df.to_csv(config.TRAIN_METADATA_PATH, index=False)
    logging.info(f"Metadaten erfolgreich in '{config.TRAIN_METADATA_PATH}' gespeichert.")
    
    return True

# Dieser Block ermöglicht es, das Skript bei Bedarf auch direkt auszuführen.
if __name__ == '__main__':
    from .utils import setup_logging
    setup_logging()
    run_data_processing_pipeline()