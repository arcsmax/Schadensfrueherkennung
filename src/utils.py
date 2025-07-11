# src/utils.py

import logging
import sys
from pathlib import Path
from rich.logging import RichHandler

# Importiere Konfigurationsvariablen direkt aus dem config-Modul
from . import config

# Versucht, die Visualisierungs-Bibliotheken zu importieren. Gib eine Warnung aus, wenn sie fehlen.
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from rich.logging import RichHandler
    _VISUALS_AVAILABLE = True
except ImportError:
    # Wenn eine dieser Bibliotheken fehlt, setzen wir ein Flag,
    # damit das Programm nicht abstürzt, sondern nur eine Warnung ausgibt.
    _VISUALS_AVAILABLE = False

# NEU: Definition der Abhängigkeiten zwischen den Pipeline-Schritten.
# Dies teilt der 'clear_downstream_caches' Funktion mit, welche Schritte von welchem Vorgänger abhängen.
CACHE_DEPENDENCIES = {
    'data_processing': ['feature_engineering', 'model_training'],
    'feature_engineering': ['model_training'],
}

def setup_logging():
    """
    Konfiguriert ein zentrales Logging-System, das sowohl in die Konsole
    als auch in eine Log-Datei schreibt.
    """
    if logging.getLogger().hasHandlers():
        logging.getLogger().handlers.clear()

    handlers = [logging.FileHandler(config.LOG_FILE_PATH, mode='w')]
    if _VISUALS_AVAILABLE:
        handlers.append(RichHandler(rich_tracebacks=True, markup=True))
    else:
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers
    )
    if _VISUALS_AVAILABLE:
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("pandas").setLevel(logging.WARNING)

def check_cache(step_name: str) -> bool:
    """
    Überprüft, ob ein erfolgreicher Abschlussmarker für einen Pipeline-Schritt existiert.

    Args:
        step_name (str): Der Name des Schrittes (z.B. 'data_processing').

    Returns:
        bool: True, wenn der Marker existiert, ansonsten False.
    """
    marker_file = config.CACHE_STATUS_DIR / f".{step_name}.success"
    if marker_file.exists():
        logging.info(f"Cache-Marker für '{step_name}' gefunden.")
        return True
    logging.info(f"Kein Cache-Marker für '{step_name}' gefunden. Schritt wird ausgeführt.")
    return False

def update_cache(step_name: str):
    """
    Erstellt einen Erfolgs-Marker für einen abgeschlossenen Pipeline-Schritt.
    Löscht zuerst alle alten Marker, um einen sauberen Zustand zu gewährleisten.

    Args:
        step_name (str): Der Name des Schrittes (z.B. 'data_processing').
    """
    # Stelle sicher, dass das Cache-Verzeichnis existiert
    config.CACHE_STATUS_DIR.mkdir(exist_ok=True)
    
    # Erstelle die neue Marker-Datei
    marker_file = config.CACHE_STATUS_DIR / f".{step_name}.success"
    marker_file.touch()
    logging.info(f"Cache-Marker für '{step_name}' erfolgreich erstellt/aktualisiert.")

def clear_cache(step_name: str = None):
    """
    Löscht Cache-Marker. Wenn kein spezifischer Schritt angegeben wird,
    werden alle Marker im Verzeichnis gelöscht.

    Args:
        step_name (str, optional): Der spezifische Schritt, dessen Cache gelöscht werden soll.
                                     Defaults to None.
    """
    if step_name:
        marker_file = config.CACHE_STATUS_DIR / f".{step_name}.success"
        if marker_file.exists():
            marker_file.unlink()
            logging.info(f"Cache-Marker für '{step_name}' wurde entfernt.")
        else:
            logging.warning(f"Kein Cache-Marker für '{step_name}' zum Löschen gefunden.")
    else:
        # Lösche alle .success-Dateien im Cache-Verzeichnis
        for f in config.CACHE_STATUS_DIR.glob('.*.success'):
            f.unlink()
        logging.info("Alle Cache-Marker wurden entfernt.")

def clear_downstream_caches(changed_step: str):
    """
    Löscht die Cache-Marker aller Schritte, die vom 'changed_step' abhängen.
    
    Args:
        changed_step (str): Der Name des Schrittes, der ausgeführt wurde und
                            dessen nachfolgende Schritte zurückgesetzt werden müssen.
    """
    if changed_step in CACHE_DEPENDENCIES:
        downstream_steps = CACHE_DEPENDENCIES[changed_step]
        logging.warning(f"Schritt '{changed_step}' wurde ausgeführt. Lösche abhängige Caches: {downstream_steps}")
        for step_to_clear in downstream_steps:
            clear_cache(step_to_clear)

def plot_feature_distribution(feature_df: pd.DataFrame, output_path: Path):
    """
    Erstellt und speichert Plots für die Verteilung ausgewählter Merkmale.
    """
    if not _VISUALS_AVAILABLE:
        logging.warning("Visualisierungs-Bibliotheken nicht installiert. Überspringe Plot-Erstellung.")
        return

    logging.info(f"Erstelle Plot zur Merkmalsverteilung und speichere ihn in {output_path}...")
    
    try:
        # KORREKTUR: Robustere Auswahl der zu plottenden Spalten
        plot_cols = [
            'td_rms', 'td_kurtosis', 'td_crest_factor',
            'oa_gmf_harmonic_1_amp', 'oa_gmf_h1_upper_sb_energy'
        ]
        # Filtere auf Spalten, die tatsächlich im DataFrame existieren
        plot_cols_exist = [col for col in plot_cols if col in feature_df.columns]
        
        if not plot_cols_exist:
            logging.warning("Keine der typischen Merkmale zum Plotten gefunden. Überspringe Visualisierung.")
            return
            
        n_features = len(plot_cols_exist)
        fig, axes = plt.subplots(n_features, 1, figsize=(12, 5 * n_features), constrained_layout=True)
        if n_features == 1:
            axes = [axes]
            
        fig.suptitle('Vergleich der wichtigsten Merkmalsverteilungen', fontsize=18)

        for i, col in enumerate(plot_cols_exist):
            sns.histplot(data=feature_df, x=col, hue='label', kde=True, ax=axes[i], palette='viridis')
            axes[i].set_title(f'Verteilung für: {col}', fontsize=12)
            axes[i].set_xlabel('Wert')
            axes[i].set_ylabel('Anzahl')

        # Stelle sicher, dass das Zielverzeichnis existiert
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.close(fig)
        logging.info("Plot erfolgreich erstellt.")
    except Exception as e:
        logging.error(f"Fehler bei der Erstellung des Plots: {e}", exc_info=True)

