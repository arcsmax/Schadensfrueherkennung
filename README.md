# Industrielle Schadensfrüherkennung

Ein Prototyp zur Früherkennung von Schäden in industriellen Anlagen basierend auf Vibrationssensordaten. Dieses Repository enthält eine vollständige ML-Pipeline von der Datenvorverarbeitung über Feature-Engineering und Modelltraining bis hin zur Extraktion und CLI-Ausführung.

---

## Inhaltsverzeichnis

1. [Features](#features)
2. [Projektstruktur](#projektstruktur)
3. [Installation](#installation)
4. [Konfiguration](#konfiguration)
5. [Datenverarbeitung](#datenverarbeitung)
6. [Feature-Engineering](#feature-engineering)
7. [Modelltraining](#modelltraining)
8. [Pipeline-Ausführung](#pipeline-ausf%C3%BChrung)
9. [Roadmap & Verbesserungen](#roadmap--verbesserungen)
10. [Lizenz](#lizenz)

---

## Features

- Simulation und Einlesen von Rohdaten
- Standard-Skalierung und Split in Training/Test
- Extraktion umfangreicher Zeit- und Frequenzbereichsmerkmale
- Training und Evaluierung klassischer ML-Modelle
- CLI-Tools zur Automatisierung von Extraktion und End-to-End-Lauf
- Konfigurierbare Parameter über zentrale `config.py`

---

## Projektstruktur

```
Schadensfrueherkennung/             # Projekt-Root
├── .venv/                         # Virtuelle Umgebung
├── data/                          # Roh- und vorverarbeitete Daten
│   ├── Data_Challenge_PHM2023_training_data/  # Original-Trainingsdaten
├── notebooks/                     # Jupyter-Notebooks und Zwischenformate
│── processed_data/                # Verarbeitete Datten und Berichte
│   ├── phm2023_na_complete_data.parquet
│   └── phm2023_na_features.parquet
├── results/                       # Modellreports und Visualisierungen
│   ├── classification_report.txt
│   └── feature_comparison_plot.png
├── src/                           # Quellcode der ML-Pipeline
│   ├── feature_engineering/       # Modul für Feature-Extraktion
│   │   ├── __init__.py
│   │   ├── main_extractor.py      # CLI nur für Feature-Extraction
│   │   ├── order_analysis.py      # Analyse der Merkmalshierarchie
│   │   ├── time_domain.py         # Zeitbereichskennwerte
│   │   └── time_frequency.py      # Frequenzbereichskennwerte
│   ├── config.py                  # Zentrale Einstellungen und Pfade
│   ├── data_processing.py         # Datengenerierung, Skalierung, Split
│   ├── feature_engineering.py     # Orchestrierung der Feature-Pipeline
│   ├── model_training.py          # Training und Evaluierung der Modelle
│   ├── utils.py                   # Hilfsfunktionen für Logging & Visualisierung
│   └── main.py                    # CLI für End-to-End-Pipeline
├── feature_extractor_utils.py     # Zusätzliche Feature-Utilityfunktionen
├── main.py                        # Alternative Einstiegspunkt / Wrapper
├── README.md                      # Projektbeschreibung und Anleitung
├── requirements.txt               # Python-Abhängigkeiten
├── .gitignore                     # Ausschlussliste für Versionskontrolle
```
---

## Installation

Folge diesen Schritten, um das Projekt auf Linux, macOS oder Windows lokal einzurichten:

### 1. Repository klonen

```bash
git clone https://github.com/arcsmax/Schadensfrueherkennung.git
cd Schadensfrueherkennung
```

### 2. Systemvoraussetzungen prüfen

- **Python**: Version ≥ 3.8 installiert?  
  ```bash
  python --version
  ```
- **pip**: Aktuellstes `pip` verwenden:
  ```bash
  pip install --upgrade pip
  ```

### 3. Virtuelle Umgebung einrichten

Um Versionskonflikte zu vermeiden, solltest du eine isolierte Umgebung verwenden.

- Auf **Linux/macOS**:
  ```bash
  python -m venv venv
  source venv/bin/activate
  ```

- Auf **Windows** (PowerShell):
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```

- Auf **Windows** (CMD):
  ```cmd
  python -m venv venv
  venv\Scripts\activate.bat
  ```

Nach Aktivierung sollte die Shell-Eingabe eine `(venv)`-Kennzeichnung zeigen.

### 4. Abhängigkeiten installieren

Mit aktivierter virtueller Umgebung installiere alle benötigten Pakete:

```bash
pip install -r requirements.txt
```

> **Tipp:** Wenn du Probleme mit einzelnen Paketen hast, prüfe die passende Python-Version oder installiere systemweite Abhängigkeiten (z. B. `build-essential`, `libomp`).

---

## Konfiguration

Alle Pfade, Hyperparameter und globalen Settings befinden sich in `config.py`. Eine typische Struktur:

```python
# config.py
data_dir = "./data"
model_dir = "./models"
batch_size = 64
learning_rate = 1e-3
test_split = 0.2
# … weitere Einstellungen
```

*Hinweis:* Für größere Projekte empfiehlt sich der Einsatz von `pydantic` oder `dataclasses` zur Validierung sowie Umgebungsvariablen via `python-dotenv`.

---
## Datenverarbeitung

In `data_processing.py` erfolgt die Generierung oder das Einlesen der Vibrationsdaten, gefolgt von Standard-Skalierung und Split in Trainings- und Testdaten.

- **Extractor** lädt Rohdaten
- **Transformer** skaliert und normalisiert
- **Loader** teilt in Sets auf

*Ausführen:*

```bash
python data_processing.py
```

---

## Feature-Engineering

Die Extraktion findet in zwei Stufen statt:

1. **Zeitbereich** (`time_domain.py`): Mittelwert, Varianz, RMS, Kurtosis, Crest-Faktor, …
2. **Frequenzbereich** (`time_frequency.py`): FFT, Spektraldichte, Bandenergie, …

Alles orchestriert durch `feature_engineering.py`, das Funktionen zu einer modularen Pipeline kombiniert.

*Ausführen:*

```bash
python feature_engineering.py
```

---

## Modelltraining

In `model_training.py` werden Klassifikationsmodelle (z. B. Random Forest, SVM) trainiert und evaluiert (Accuracy, Precision, Recall, F1-Score).

*Ausführen:*

```bash
python model_training.py
```

---

## Pipeline-Ausführung

- End-to-End-Lauf:
  ```bash
  python main.py --config config.py
  ```
- Nur Feature-Extraktion:
  ```bash
  python main_extractor.py --input ./data/raw --output ./data/features
  ```

*CLI-Optionen* lassen sich über `Click` oder `Typer` leicht erweitern.

---

## Roadmap & Verbesserungen

1. **End-to-End Orchestrierung**\
   Integration von Airflow, Prefect oder Luigi zur Automatisierung, Scheduling und Monitoring.
2. **Echtzeit-Streaming**\
   Anbindung von Kafka/MQTT und Windowed-Feature-Extraction für Live-Daten.
3. **Deep Learning**\
   1D-CNNs, Transformer-Modelle und Autoencoder für Anomalieerkennung.
4. **Edge Deployment**\
   Docker/Kubernetes für Edge-Devices; TensorFlow Lite oder ONNX für schlanke Inferenz.
5. **Experiment-Tracking & Monitoring**\
   MLflow oder Weights & Biases, Alerts bei Modell-Drift.
6. **Tests & CI/CD**\
   Ausbau zu Unit-, Integrations- und E2E-Tests; GitHub Actions / GitLab CI.

---



## Wichtig

Nach jeder Änderung des Codes muss die alte Marker-Datei gelöscht werden.
```bash
rm .cache_status/.data_generation.success
```

Viel Spaß :)
