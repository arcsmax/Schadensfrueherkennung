# src/main.py
"""
Zentraler Orchestrator für die Schadensfrüherkennungs-Pipeline.
Dieses Skript steuert den sequenziellen Ablauf über Kommandozeilen-Argumente.

BEISPIEL-AUFRUFE:
- Gesamte Trainings-Pipeline ausführen:
  python src/main.py train

- Training erzwingen (Cache ignorieren):
  python src/main.py train --force

- Vorhersage für eine einzelne Datei:
  python src/main.py predict --file ./data/Data_Challenge_PHM2023_training_data/pitting_level_1/V1000_0N_1.txt

- Finale Evaluierung auf dem Test-Set:
  python src/main.py evaluate
"""
import logging
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from src import utils, config
    from src.data_processing import run_data_processing_pipeline
    from src.feature_orchestrator import run_feature_engineering_pipeline
    from src.model_training import run_model_training_pipeline, predict_single_instance, evaluate_on_test_data
except ImportError as e:
    print(f"Fehler beim Importieren der Module: {e}")
    sys.exit(1)

def handle_train_command(force_rerun: bool):
    """Führt die gesamte Trainings-Pipeline Schritt für Schritt aus."""
    if force_rerun:
        logging.info("Option --force erkannt. Lösche gesamten Cache.")
        utils.clear_cache()

    if not utils.check_cache('data_processing'):
        if not run_data_processing_pipeline():
            logging.error("Pipeline-Abbruch: Fehler in der Datenverarbeitung."); return
        utils.update_cache('data_processing')
    else: logging.info("Schritt 1 (Datenverarbeitung) wird aufgrund des Caches übersprungen.")

    if not utils.check_cache('feature_engineering'):
        if not run_feature_engineering_pipeline(is_test_data=False):
            logging.error("Pipeline-Abbruch: Fehler im Feature Engineering."); return
        utils.update_cache('feature_engineering')
    else: logging.info("Schritt 2 (Feature Engineering) wird aufgrund des Caches übersprungen.")

    if not utils.check_cache('model_training'):
        if not run_model_training_pipeline():
            logging.error("Pipeline-Abbruch: Fehler im Modelltraining."); return
        utils.update_cache('model_training')
    else: logging.info("Schritt 3 (Modelltraining) wird aufgrund des Caches übersprungen.")

def handle_predict_command(filepath: str):
    """Führt eine Vorhersage für eine einzelne Datei aus."""
    prediction, probabilities = predict_single_instance(filepath)
    if prediction is not None:
        logging.info("--- Vorhersage-Ergebnis ---")
        logging.info(f"  Vorhergesagter Schadensgrad: Level {prediction}")
        logging.info("  Wahrscheinlichkeiten pro Klasse:")
        for i, prob in enumerate(probabilities):
            logging.info(f"    - Klasse {i}: {prob:.2%}")
        logging.info("--------------------------")

def handle_evaluate_command():
    """Führt die Evaluierung auf dem separaten Testdatensatz durch."""
    # Stelle sicher, dass die Test-Features existieren
    if not config.TEST_PROCESSED_FEATURES_PATH.exists():
        logging.info("Test-Features nicht gefunden. Erstelle sie zuerst...")
        # Nutze die bestehende Pipeline-Logik zur Erstellung
        if not run_feature_engineering_pipeline(is_test_data=True):
             logging.error("Konnte Test-Features nicht erstellen. Abbruch.")
             return
    evaluate_on_test_data()

def main():
    parser = argparse.ArgumentParser(description="Orchestrator für die Schadensfrüherkennungs-Pipeline.")
    subparsers = parser.add_subparsers(dest='command', required=True, help="Verfügbare Befehle")

    train_parser = subparsers.add_parser('train', help="Führt die gesamte Trainings-Pipeline aus.")
    train_parser.add_argument('--force', action='store_true', help="Erzwingt die Neuausführung aller Schritte.")

    predict_parser = subparsers.add_parser('predict', help="Macht eine Vorhersage für eine einzelne Rohdatendatei.")
    predict_parser.add_argument('--file', type=str, required=True, help="Pfad zur .txt-Rohdatendatei.")

    subparsers.add_parser('evaluate', help="Evaluiert das Modell auf dem separaten Test-Set.")
    
    args = parser.parse_args()
    
    utils.setup_logging()
    logging.info("="*50)
    logging.info("STARTE SCHADENSFRÜHERKENNUNGS-ORCHESTRATOR")
    
    if args.command == 'train':
        handle_train_command(args.force)
    elif args.command == 'predict':
        handle_predict_command(args.file)
    elif args.command == 'evaluate':
        handle_evaluate_command()

    logging.info("ORCHESTRATOR-LAUF BEENDET")
    logging.info("="*50)

if __name__ == "__main__":
    main()
