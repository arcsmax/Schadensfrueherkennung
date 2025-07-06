# src/generate_dashboard.py
import json
import logging
import re
from pathlib import Path

from . import config

def parse_classification_report(report_path: Path) -> (float, str):
    """Liest eine .txt-Report-Datei und extrahiert den F1-Score und den Text."""
    if not report_path.exists():
        logging.warning(f"Report-Datei nicht gefunden: {report_path}")
        return 0.0, "Report nicht gefunden."
    
    with open(report_path, 'r') as f:
        report_text = f.read()
    
    # Extrahiere den gewichteten F1-Score
    match = re.search(r'weighted avg\s+[\d\.]+\s+[\d\.]+\s+([\d\.]+)', report_text)
    f1_score = float(match.group(1)) if match else 0.0
    return f1_score, report_text

def run_dashboard_generation():
    """Liest alle Ergebnisdateien und generiert eine dynamische HTML-Dashboard-Datei."""
    logging.info("="*50)
    logging.info("Starte Dashboard-Generierung")
    logging.info("="*50)

    # 1. Sammle alle Modelldaten
    model_data = {}
    
    # RandomForest
    rf_f1, rf_report = parse_classification_report(config.VALIDATION_CLASSIFICATION_REPORT_PATH)
    model_data['RandomForest'] = {'f1Score': rf_f1, 'report': rf_report}

    # 1D-CNN
    cnn_f1, cnn_report = parse_classification_report(config.CNN_CLASSIFICATION_REPORT_PATH)
    model_data['1D-CNN'] = {'f1Score': cnn_f1, 'report': cnn_report}

    # Transformer
    trans_f1, trans_report = parse_classification_report(config.TRANSFORMER_CLASSIFICATION_REPORT_PATH)
    model_data['Transformer'] = {'f1Score': trans_f1, 'report': trans_report}

    # 2. Lade Feature Importance Daten
    feature_data = {}
    if config.FEATURE_IMPORTANCE_PATH.exists():
        with open(config.FEATURE_IMPORTANCE_PATH, 'r') as f:
            feature_data = json.load(f)
    else:
        logging.warning(f"Feature Importance Datei nicht gefunden: {config.FEATURE_IMPORTANCE_PATH}")

    # 3. Lade die HTML-Vorlage
    # Annahme: Vorlage ist im selben Verzeichnis oder einem bekannten Pfad
    template_path = Path(__file__).parent.parent / "dashboard_template.html"
    if not template_path.exists():
        logging.error(f"Dashboard-Vorlage nicht gefunden unter: {template_path}. Bitte die Datei 'interactive_dashboard_html' als 'dashboard_template.html' im Hauptverzeichnis speichern.")
        return

    with open(template_path, 'r', encoding='utf-8') as f:
        template_html = f.read()

    # 4. Ersetze die Platzhalter mit echten Daten
    # Wandle die Python-Dicts in JSON-Strings um, die direkt in JS eingefügt werden können
    final_html = template_html.replace("'__MODEL_DATA_PLACEHOLDER__'", json.dumps(model_data))
    final_html = final_html.replace("'__FEATURE_IMPORTANCE_PLACEHOLDER__'", json.dumps(feature_data))

    # 5. Speichere das finale Dashboard
    output_path = config.BASE_DIR / "Project_Dashboard.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_html)
        
    logging.info(f"Dashboard erfolgreich generiert: {output_path}")

if __name__ == '__main__':
    # Ermöglicht das direkte Testen dieses Skripts
    from .utils import setup_logging
    setup_logging()
    run_dashboard_generation()
