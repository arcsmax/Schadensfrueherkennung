# src/main.py
"""
Zentraler Orchestrator für die Schadensfrüherkennungs-Pipeline.

Dieses Skript dient als Haupt-Einstiegspunkt und steuert den gesamten
Workflow über eine professionelle Kommandozeilenschnittstelle (CLI).
Es ermöglicht das getrennte Ausführen verschiedener Pipelines, das Treffen
von Vorhersagen und die Generierung des finalen Ergebnis-Dashboards.

-----------------------------------------------------------------------------
BEFEHLE & BEISPIEL-AUFRUFE:
-----------------------------------------------------------------------------

1.  **Klassisches ML-Modell trainieren:**
    Führt die gesamte Pipeline mit manuellem Feature Engineering und
    RandomForest-Modell aus. Nutzt Caching, um erledigte Schritte zu überspringen.
    
    `python src/main.py train`

2.  **Training erzwingen (Cache ignorieren):**
    Wie 'train', aber löscht zuerst alle Cache-Dateien. Nützlich nach Code-Änderungen.
    
    `python src/main.py train --force`

3.  **Deep Learning Modelle trainieren:**
    Startet das Training für die End-to-End Deep Learning Modelle.

    - Für das 1D-CNN:
      `python src/main.py train_cnn`
    - Für das Transformer-Modell:
      `python src/main.py train_transformer`

4.  **Transfer Learning demonstrieren:**
    Führt den simulierten Workflow für Transfer Learning aus.
    
    `python src/main.py run_transfer_learning`

5.  **Interaktives Dashboard generieren:**
    Liest alle Ergebnis-Artefakte und erstellt die finale 'Project_Dashboard.html'.
    
    `python src/main.py generate_dashboard`

6.  **Echtzeit-Vorhersage treffen:**
    Nutzt das trainierte RandomForest-Modell, um eine einzelne Rohdatei zu klassifizieren.
    
    `python src/main.py predict --file ./data/Data_Challenge_PHM2023_training_data/pitting_level_1/V1000_0N_1.txt`

7.  **Finale Evaluierung:**
    Evaluiert das trainierte RandomForest-Modell auf dem separaten Test-Datensatz.
    
    `python src/main.py evaluate`
-----------------------------------------------------------------------------
"""
import logging
import argparse
import sys
from pathlib import Path
import time
import pandas as pd # NEU: Import für Benchmark
import numpy as np  # NEU: Import für Benchmark
import torch        # NEU: Import für Benchmark

sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from src import utils, config
    from src.data_processing import run_data_processing_pipeline
    from src.feature_orchestrator import run_feature_engineering_pipeline
    from src.model_training import run_model_training_pipeline, predict_single_instance, evaluate_on_test_data
    from src.deep_learning.trainer import run_cnn_training_pipeline
    from src.deep_learning.model import Simple1DCNN  # NEU: Import des CNN-Modells für Benchmark
    from src.transformer.trainer import run_transformer_training_pipeline
    from src.transfer_learning.trainer import run_transfer_learning_pipeline
    from src.generate_dashboard import run_dashboard_generation
except ImportError as e:
    print(f"Fehler beim Importieren der Module: {e}")
    print("Stellen Sie sicher, dass Sie das Skript vom Root-Verzeichnis des Projekts ausführen und alle Abhängigkeiten installiert sind.")
    sys.exit(1)


def handle_train_command(force_rerun: bool):
    """Orchestriert die klassische ML-Trainings-Pipeline (RandomForest)."""
    if force_rerun:
        logging.info("Option --force erkannt. Lösche gesamten Cache, um eine vollständige Neuausführung zu erzwingen.")
        utils.clear_cache()
    if not utils.check_cache('data_processing'):
        if not run_data_processing_pipeline(): logging.error("Pipeline-Abbruch: Datenverarbeitung."); return
        utils.update_cache('data_processing')
    else: logging.info("Schritt 1 (Datenverarbeitung) wird aufgrund des Caches übersprungen.")
    if not utils.check_cache('feature_engineering'):
        if not run_feature_engineering_pipeline(is_test_data=False): logging.error("Pipeline-Abbruch: Feature Engineering."); return
        utils.update_cache('feature_engineering')
    else: logging.info("Schritt 2 (Feature Engineering) wird aufgrund des Caches übersprungen.")
    if not utils.check_cache('model_training'):
        if not run_model_training_pipeline(): logging.error("Pipeline-Abbruch: Modelltraining."); return
        utils.update_cache('model_training')
    else: logging.info("Schritt 3 (Modelltraining) wird aufgrund des Caches übersprungen.")

def handle_run_all_command(force_rerun: bool):
    """Führt alle wichtigen Trainings-Pipelines nacheinander aus."""
    logging.info("=== Starte 'run-all': Führe alle Trainings-Pipelines aus ===")
    logging.info("\n--- Teil 1 von 4: Klassisches ML (RandomForest) ---")
    handle_train_command(force_rerun=force_rerun)
    logging.info("\n--- Teil 2 von 4: Deep Learning (1D-CNN) ---")
    run_cnn_training_pipeline()
    logging.info("\n--- Teil 3 von 4: Deep Learning (Transformer) ---")
    run_transformer_training_pipeline()
    logging.info("\n--- Teil 4 von 4: Demonstration (Transfer Learning) ---")
    run_transfer_learning_pipeline()
    logging.info("\n=== 'run-all' erfolgreich abgeschlossen ===")
    logging.info("Alle Modelle wurden trainiert. Sie können jetzt das Dashboard generieren: `python src/main.py generate_dashboard`")


def handle_benchmark_command():
    """Misst die Inferenzgeschwindigkeit der trainierten Modelle."""
    logging.info("="*50)
    logging.info("Starte Inferenz-Benchmark")
    logging.info("="*50)

    try:
        metadata = pd.read_csv(config.TRAIN_METADATA_PATH)
        sample_filepath = metadata.iloc[0]['filepath']
    except (FileNotFoundError, IndexError):
        logging.error("Metadaten-Datei nicht gefunden oder leer. Bitte `train` oder `run-all` zuerst ausführen.")
        return

    logging.info(f"Verwende Testdatei: {sample_filepath}")
    
    # 1. Benchmark RandomForest
    logging.info("--- Benchmark: RandomForest ---")
    try:
        start_time = time.perf_counter()
        for _ in range(100):
            predict_single_instance(sample_filepath)
        end_time = time.perf_counter()
        avg_time_rf = (end_time - start_time) * 10 
        logging.info(f"Durchschnittliche Inferenzzeit: {avg_time_rf:.4f} ms")
    except Exception as e:
        logging.error(f"Fehler beim RandomForest-Benchmark: {e}")

    # 2. Benchmark CNN
    logging.info("--- Benchmark: 1D-CNN ---")
    try:
        device = torch.device("cpu")
        # KORREKTUR: Lade die Anzahl der Klassen dynamisch
        num_classes = pd.read_csv(config.TRAIN_METADATA_PATH)['pitting_level'].nunique()
        cnn_model = Simple1DCNN(num_classes=num_classes).to(device)
        cnn_model.load_state_dict(torch.load(config.SAVED_CNN_MODEL_PATH, map_location=device))
        cnn_model.eval()

        data = pd.read_csv(sample_filepath, sep=r'\s+', header=None, names=['acc_x', 'acc_y', 'acc_z', 'tacho'], engine='python')
        signal = data['acc_x'].values[:config.SIGNAL_LENGTH].astype(np.float32)
        mean, std = np.mean(signal), np.std(signal)
        if std > 0: signal = (signal - mean) / std
        signal_tensor = torch.from_numpy(signal).unsqueeze(0).unsqueeze(0).to(device)

        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(100):
                cnn_model(signal_tensor)
        end_time = time.perf_counter()
        avg_time_cnn = (end_time - start_time) * 10
        logging.info(f"Durchschnittliche Inferenzzeit: {avg_time_cnn:.4f} ms")
    except FileNotFoundError:
        logging.warning("CNN-Modell nicht gefunden. Überspringe Benchmark.")
    except Exception as e:
        logging.error(f"Fehler beim CNN-Benchmark: {e}")


def main():
    """Hauptfunktion: Parst die Kommandozeilen-Argumente."""
    shared_argument_parser = argparse.ArgumentParser(add_help=False)
    shared_argument_parser.add_argument('--force', action='store_true', help="Erzwingt die Neuausführung durch Löschen des Caches.")
    shared_argument_parser.add_argument('--fraction', type=float, default=1.0, help="Bruchteil der Daten für schnelle Tests (z.B. 0.1).")

    parser = argparse.ArgumentParser(description="Zentraler Orchestrator für die Schadensfrüherkennungs-Pipeline.", formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', required=True, help="Verfügbare Befehle")

    subparsers.add_parser('run-all', help="Führt alle Trainings-Pipelines aus.", parents=[shared_argument_parser])
    subparsers.add_parser('train', help="Führt die klassische Trainings-Pipeline (RandomForest) aus.", parents=[shared_argument_parser])
    subparsers.add_parser('train_cnn', help="Führt die Deep Learning Trainings-Pipeline (1D-CNN) aus.")
    subparsers.add_parser('train_transformer', help="Führt die Transformer Trainings-Pipeline aus.")
    subparsers.add_parser('run_transfer_learning', help="Demonstriert den Transfer Learning Workflow.")
    subparsers.add_parser('generate_dashboard', help="Generiert das interaktive HTML-Dashboard.")
    subparsers.add_parser('benchmark', help="Misst die Inferenzgeschwindigkeit der Modelle.")
    
    predict_parser = subparsers.add_parser('predict', help="Macht eine Vorhersage für eine einzelne Datei.")
    predict_parser.add_argument('--file', type=str, required=True, help="Pfad zur .txt-Rohdatendatei.")
    subparsers.add_parser('evaluate', help="Evaluiert das klassische Modell auf dem separaten Test-Set.")
    
    args = parser.parse_args()
    
    if hasattr(args, 'fraction') and args.fraction < 1.0:
        config.DATA_FRACTION = args.fraction
        logging.info(f"DEV MODE: Verwende nur {config.DATA_FRACTION * 100:.0f}% der Trainingsdaten.")

    utils.setup_logging()
    logging.info("="*50)
    logging.info("STARTE SCHADENSFRÜHERKENNUNGS-ORCHESTRATOR")
    
    if args.command == 'run-all':
        handle_run_all_command(args.force)
    elif args.command == 'train':
        handle_train_command(args.force)
    elif args.command == 'train_cnn':
        run_cnn_training_pipeline()
    elif args.command == 'train_transformer':
        run_transformer_training_pipeline()
    elif args.command == 'run_transfer_learning':
        run_transfer_learning_pipeline()
    elif args.command == 'generate_dashboard':
        run_dashboard_generation()
    elif args.command == 'benchmark':
        handle_benchmark_command()
    elif args.command == 'predict':
        predict_single_instance(args.file)
    elif args.command == 'evaluate':
        evaluate_on_test_data()

    logging.info("ORCHESTRATOR-LAUF BEENDET")
    logging.info("="*50)

if __name__ == "__main__":
    main()
