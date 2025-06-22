# Industrielle Schadensfrüherkennung (Predictive Maintenance)

Ein Prototyp zur Früherkennung von Schäden in industriellen Anlagen basierend auf Vibrationssensordaten. Dieses Repository ist darauf ausgelegt eine vollständige ML-Pipeline von der Datenvorverarbeitung über Feature-Engineering und Modelltraining bis hin zur Extraktion und CLI-Ausführung zu enthalten.

---

## Inhaltsverzeichnis
1. [Kernfeatures & Innovationen](#kernfeatures--innovationen)
2. [Projektstruktur](#projektstruktur)
3. [Installation & Setup](#installation--setup)
4. [Konfiguration](#konfiguration)
5. [Pipeline-Komponenten](#pipeline-komponenten)
   - [Datenverarbeitung](#datenverarbeitung)
   - [Feature-Engineering](#feature-engineering)
   - [Modelltraining](#modelltraining)
   - [Automatisierte Ausführung (CLI)](#automatisierte-ausführung-cli)
6. [Wichtige Hinweise](#wichtige-hinweise)
7. [Roadmap & Vision](#roadmap--vision)
8. [Mitwirken (Contributing)](#mitwirken-contributing)
9. [Lizenz](#lizenz)

---

## Kernfeatures & Innovationen

### Aktuelle Features

- **Daten-Simulation & Einlesen**: Verarbeitung von Rohdaten aus gängigen Formaten (CSV, Parquet, Datenbanken).
- **Modulare Vorverarbeitung**: Standard-Skalierung und strategischer Split in Trainings-, Validierungs- und Test-Sets.
- **Umfangreiches Feature-Engineering**: Extraktion von über 50 Merkmalen aus Zeit- und Frequenzbereich (statistische Momente, spektrale Kennwerte etc.).
- **Klassisches ML-Training**: Training und Evaluierung robuster Modelle (Random Forest, SVM, Gradient Boosting).
- **CLI-Tools**: Steuerung der Feature-Extraktion und der gesamten End-to-End-Pipeline über Kommandozeile.
- **Zentralisierte Konfiguration**: Alle Parameter und Pfade in `src/config.py` verwaltet.

### Projekt-Highlights & Performance

Dieses Projekt umfasst nicht nur ein funktionsfähiges ML-Modell, sondern auch eine nach professionellen MLOps-Prinzipien entworfene, robuste und performante Pipeline. Eine detaillierte Analyse der Architektur und der erzielten Performance-Optimierungen finden Sie in unserem gesonderten Projektbericht.

➡️ **[Zum vollständigen Projektbericht: Performance-Optimierung & Pipeline-Architektur](./performance-optimierung.md)**

### Geplante Features

- **Explainable AI (XAI)**: Integration von SHAP oder LIME für transparente Modellentscheidungen.
- **Automated Feature Engineering (AutoML)**: Bibliotheken wie FeatureTools, um komplexe Merkmale automatisch zu entdecken.
- **Deep Learning für Zeitreihen**: 1D-CNNs und Transformer-Modelle, um Merkmale direkt aus Rohsignalen zu lernen.
- **Anomalieerkennung mit Autoencodern**: Training auf gesunden Daten, um unbekannte Fehlertypen als Anomalien zu erkennen.
- **Digitale Zwillings-Integration**: Verbindung zu einem digitalen Zwilling für "Was-wäre-wenn"-Szenarien und Visualisierung.

---

## Projektstruktur

```plaintext
Schadensfrueherkennung/             # Projekt-Root
├── data/                          # Lokale Roh- und Vorverarbeitete Daten
│   ├── raw/                       # Original-Rohdaten (CSV, Parquet)
│   └── processed/                 # Verarbeitete Feature-Sets
├── notebooks/                     # Jupyter-Notebooks für Analyse & Prototyping
├── results/                       # Modellreports, Visualisierungen
│   ├── classification_report.txt
│   └── feature_importance.png
├── src/                           # Quellcode der ML-Pipeline
│   ├── feature_engineering/       # Modul für Feature-Extraktion
│   │   ├── main_extractor.py      # CLI für Feature-Extraction
│   │   ├── order_analysis.py      # Analyse der Merkmals-Hierarchie
│   │   ├── time_domain.py         # Zeitbereichskennwerte
│   │   └── time_frequency.py      # Frequenzbereichskennwerte
│   ├── config.py                  # Zentrale Einstellungen & Pfade
│   ├── data_processing.py         # Daten-Generator, Skalierung & Split
│   ├── feature_engineering.py     # Orchestrierung des Feature-Flows
│   ├── model_training.py          # Training & Evaluierung der Modelle
│   ├── utils.py                   # Logging, I/O, Visualisierung
│   └── main.py                    # CLI für End-to-End-Pipeline
├── tests/                         # Unit- und Integrations-Tests
├── .cache_status/                 # Cache-Dateien zur Statusverfolgung
├── requirements.txt               # Python-Abhängigkeiten
├── .gitignore                     # Versionskontrolle Ausnahmen
└── README.md                      # Projektbeschreibung
```

---

## Installation & Setup

Folge diesen Schritten, um das Projekt auf Linux, macOS oder Windows lokal einzurichten:

1. **Repository klonen**

   ```bash
   git clone https://github.com/arcsmax/Schadensfrueherkennung.git
   cd Schadensfrueherkennung
   ```

2. **Systemvoraussetzungen prüfen**

   - Python ≥ 3.8 installiert?  `python --version`
   - Aktualisiere `pip`:  `pip install --upgrade pip`

3. **Virtuelle Umgebung einrichten**

   - Linux/macOS:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   - Windows (PowerShell):

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   - Windows (CMD):

   ```cmd
   python -m venv .venv
   .venv\Scripts\activate.bat
   ```

4. **Abhängigkeiten installieren**

   ```bash
   pip install -r requirements.txt
   ```

> **Tipp:** Bei Problemen mit Paketen (z. B. SciPy), installiere systemweite Bibliotheken (`build-essential`, `libomp`).

---

## Konfiguration

Alle globalen Einstellungen, Pfade und Hyperparameter werden zentral in `src/config.py` verwaltet. Beispiel:

```python
# config.py
DATA_DIR = "./data"
MODEL_DIR = "./models"
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
TEST_SPLIT = 0.2
```

Für Produktionseinsatz empfiehlt sich `python-dotenv` oder Frameworks wie Hydra/Pydantic.

---

## Pipeline-Komponenten

### Datenverarbeitung

In `src/data_processing.py`:

- **Laden**: Rohdaten gemäß `DATA_DIR` einlesen
- **Bereinigen**: Fehlende Werte behandeln
- **Transformieren**: StandardScaler oder eigene Transformer anwenden
- **Aufteilen**: Split in Trainings- und Test-Sets

```bash
python src/data_processing.py
```

### Feature-Engineering

In `src/feature_engineering/`:

- **time\_domain.py**: Statistische Kennwerte (Mittelwert, Varianz, RMS, Kurtosis, Crest-Faktor)
- **time\_frequency.py**: FFT, Spektraldichte, Band-Energien, dominante Frequenzen
- **main\_extractor.py**: CLI-Workflow für reine Feature-Extraktion

```bash
python src/feature_engineering/main_extractor.py \
  --input-dir ./data/raw \
  --output-file ./data/processed/features.parquet
```

### Modelltraining

In `src/model_training.py`:

- Training klassischer Modelle (RandomForest, SVM, etc.)
- Evaluierung (Accuracy, Precision, Recall, F1-Score)
- Speichern von Modellen und Berichten in `results/`

```bash
python src/model_training.py
```

### Automatisierte Ausführung (CLI)

End-to-End-Pipeline:

```bash
python src/main.py --config src/config.py
```

Nur Feature-Extraction:

```bash
python src/feature_engineering/main_extractor.py --help
```

Für eine erweiterte CLI empfiehlt sich `Click` oder `Typer`.

---

## Wichtige Hinweise

- **Caching**: Das Projekt nutzt Marker in `.cache_status/` zur Vermeidung wiederholter Rechenoperationen. Bei Code-Änderungen an einem Schritt entferne das entsprechende Marker-File.
- **Dateipfade**: Passe `DATA_DIR` und `MODEL_DIR` in `config.py` bei Bedarf an.

---

## Roadmap & Vision

### Phase 1: Grundlagen & MLOps

- [ ] Testing Framework: Implementierung von pytest für Unit- und Integrationstests.

- [ ] CI/CD-Pipeline: Einrichtung von GitHub Actions zur automatischen Ausführung von Tests und Linting bei jedem Push.

- [ ] Experiment Tracking: Integration von MLflow oder Weights & Biases zur Protokollierung von Experimenten, Parametern und Metriken.

### Phase 2: Modellierungs-Innovation

- [ ] Explainable AI (XAI): Integration von SHAP zur Visualisierung von Modellentscheidungen.

- [ ] Deep Learning: Implementierung eines 1D-CNN-Modells als Alternative zu klassischem ML.

- [ ] Erweiterte Signalverarbeitung: Einsatz von Wavelet-Transformationen für eine kombinierte Zeit-Frequenz-Analyse.

### Phase 3: Produktivsetzung & Skalierung

- [ ] REST API: Entwicklung einer FastAPI-Schnittstelle, um das Modell für Echtzeitanfragen bereitzustellen.

- [ ] Echtzeit-Streaming: Anbindung an einen Kafka- oder MQTT-Broker zur Verarbeitung von Live-Sensordaten.

- [ ] Containerisierung & Deployment: Erstellung eines Docker-Images und Bereitstellung auf einer Edge- oder Cloud-Plattform (z.B. mit Kubernetes).

---

## Mitwirken (Contributing)

Beiträge sind willkommen! Bitte folge diesen Schritten:

1. Repository forken
2. Neuen Branch erstellen (`git checkout -b feature/mein-feature`)
3. Änderungen implementieren & dokumentieren
4. Code formatieren (`black .`, `isort .`)
5. Tests hinzufügen und lokal ausführen
6. Pull Request mit beschreibendem Titel eröffnen

---

## Lizenz

Dieses Projekt steht unter der MIT License. Siehe [LICENSE](./LICENSE) für Details.





## Wichtig

Nach jeder Änderung des Codes muss die alte Marker-Datei gelöscht werden.
```bash
rm .cache_status/.data_generation.success
```

Viel Spaß :nerd_face:
