# Industrielle Schadensfrüherkennung (Predictive Maintenance)

Ein Prototyp zur Früherkennung von Schäden in industriellen Anlagen basierend auf Vibrationssensordaten. Dieses Repository ist darauf ausgelegt eine vollständige ML-Pipeline von der Datenvorverarbeitung über Feature-Engineering und Modelltraining bis hin zur Extraktion und CLI-Ausführung zu enthalten.

---

## Inhaltsverzeichnis
1. [Kernfeatures & Innovationen](#kernfeatures--innovationen)
2. [Projektstruktur](#projektstruktur)
3. [Installation & Setup](#installation--setup)
4. [Konfiguration](#konfiguration)
5. [Workflow & Ausführung](#Workflow--Ausführung)  
   * [Schritt 1: Modelle trainieren](#schritt-1-modelle-trainieren) 
   * [Schritt 2: Interaktives Dashboard generieren](#schritt-2-interaktives-dashboard-generieren)  
   * [Weitere Befehle](#weitere-befehle)
6. [Pipeline-Komponenten](#pipeline-komponenten)
7. [Wichtige Hinweise](#wichtige-hinweise)
8. [Roadmap & Vision](#roadmap--vision)
9. [Mitwirken (Contributing)](#mitwirken-contributing)
10. [Lizenz](#lizenz)

---

## Kernfeatures & Innovationen

### Aktuelle Features

- **Daten-Simulation & Einlesen**: Verarbeitung von Rohdaten aus gängigen Formaten (CSV, Parquet).
- **Modulare Vorverarbeitung**: Standard-Skalierung und strategischer Split in Trainings-, Validierungs- und Test-Sets.
- **Umfangreiches Feature-Engineering**: Parallele Extraktion von über 50 Merkmalen aus Zeit-, Frequenz- und Ordnungs-Domänen (statistische Momente, spektrale Kennwerte etc.).
- **Multi-Modell-Training**: Training und Evaluierung robuster Modelle wie RandomForest, 1D-CNNs und Transformer.
- **CLI-Tools**: Steuerung der gesamten End-to-End-Pipeline, einzelner Trainingsläufe und der Inferenz über eine Kommandozeile.
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
Schadensfrueherkennung/            # Projekt-Root
├── data/                          # Lokale Roh- und Vorverarbeitete Daten
│   ├── phm_metadata_train.csv     # Original-Rohdaten (CSV, Parquet)
│   └── processed_features_train.csv # Verarbeitete Feature-Sets
├── notebooks/                     # Jupyter-Notebooks für Analyse & Prototyping
├── results/                       # Modellreports, Visualisierungen
│   ├── classification_report.txt
│   └── feature_importance.png
│   ├── classical_ml/              # Ergebnisse des RandomForest-Modells
│   ├── cnn_model/                 # Ergebnisse des 1D-CNN-Modells
│   └── transformer_model/         # Ergebnisse des Transformer-Modells
├── src/                           # Quellcode der ML-Pipeline
│   ├── feature_engineering/       # Module für Feature-Extraktion
│   │   ├── time_domain.py         # Zeitbereichskennwerte
│   │   ├── time_frequency.py      # Zeit-Frequenz-Analyse (DWT)
│   │   └── order_analysis.py      # Ordnungsanalyse
│   ├── deep_learning/             # CNN-spezifische Module
│   ├── transformer/               # Transformer-spezifische Module
│   ├── config.py                  # Zentrale Einstellungen & Pfade
│   ├── data_processing.py         # Metadaten-Erstellung & Sampling
│   ├── feature_orchestrator.py    # Orchestrierung des parallelen Feature-Flows
│   ├── model_training.py          # Training des klassischen Modells
│   ├── generate_dashboard.py      # Skript zur Dashboard-Generierung
│   ├── utils.py                   # Logging, Caching, I/O
│   └── main.py                    # CLI für End-to-End-Pipeline
├── .cache_status/                 # Cache-Dateien zur Statusverfolgung
├── requirements.txt               # Python-Abhängigkeiten
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
# src/config.py

# -- Pfade --
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
CLASSICAL_ML_DIR = RESULTS_DIR / "classical_ml"

# -- Klassisches Modelltraining Parameter --
VALIDATION_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 150

# -- Deep Learning Hyperparameter --
NUM_EPOCHS = 25
BATCH_SIZE = 32
LEARNING_RATE = 0.001
```

Für Produktionseinsatz empfiehlt sich `python-dotenv` oder Frameworks wie Hydra/Pydantic.

---
## Workflow & Ausführung

Dieses Projekt ist als interaktives Framework konzipiert. Der empfohlene Workflow besteht aus zwei einfachen Schritten: Zuerst werden die Modelle trainiert, deren Ergebnisse Sie interessieren, und danach wird das Dashboard generiert, das alle Resultate zusammenfasst.

### **Schritt 1: Modelle trainieren**

+ Führen Sie die folgenden Befehle in Ihrem Terminal aus dem Projekt-Hauptverzeichnis aus. 
+ Dank des Caching-Systems können Sie die Befehle beliebig oft ausführen – bereits abgeschlossene Schritte werden automatisch übersprungen.

  #### 1. Klassisches Modell (RandomForest) trainieren  
  (Erstellt handgemachte Features und trainiert darauf)  
  ```bash
  python src/main.py train
  ```

  #### 2. 1D-CNN Modell trainieren  
  (Lernt End-to-End aus den Rohdaten, nutzt GPU)
   ```bash
   python src/main.py train\_cnn
   ```
  #### 3. Transformer Modell trainieren  
  (State-of-the-Art-Ansatz, nutzt GPU)  
  ```bash
  python src/main.py train\_transformer
  ```

  #### 4. Transfer Learning Demonstration ausführen  
  (Simuliert die Anpassung an neue, seltene Fehler)  
  ```bash
  python src/main.py run\_transfer\_learning
  ```

Nachdem diese Befehle durchgelaufen sind, enthält Ihr results-Verzeichnis alle Ergebnisdateien (Reports, Modelle etc.), die für den nächsten Schritt benötigt werden.

## **Schritt 2: Interaktives Dashboard generieren**

Dies ist der finale und wichtigste Schritt, um alle Ergebnisse zu visualisieren. Führen Sie den folgenden Befehl aus:

- Liest alle Ergebnisdateien und erstellt eine einzelne HTML-Datei  
```bash
python src/main.py generate\_dashboard
```
  Das Skript liest automatisch alle generierten Klassifikationsberichte und Artefakte. Anschließend füllt es die Vorlage dashboard\_template.html mit den echten, aktuellen Daten Ihrer Trainingsläufe und speichert das Ergebnis als Project\_Dashboard.html im Hauptverzeichnis.  

**Öffnen Sie diese Project\_Dashboard.html-Datei in Ihrem Webbrowser, um eine vollständige, interaktive Übersicht über alle Projektergebnisse zu erhalten.**

### **Weitere Befehle**

Das Projekt bietet zusätzliche Befehle für spezifische Aufgaben:

* Training erzwingen: ```python src/main.py train \--force ``` 
  Löscht den Cache und führt die klassische Pipeline vollständig neu aus.  
* Echtzeit-Vorhersage (mit RandomForest): ```python src/main.py predict \--file PFAD\_ZUR\_DATEN.txt```  
  Macht eine Vorhersage für eine einzelne, neue Messung.

---

## Pipeline-Komponenten

- ### Datenverarbeitung

   In `src/data_processing.py`:

   - **Sammeln**: Metadaten aus Dateinamen und Ordnern extrahieren.
   - **Validieren**: Struktur der Rohdaten prüfen.
   - **Aufteilen**: SOptionales stratifiziertes Sampling für schnelle Tests.

- ### Feature-Engineering

   In `src/feature_orchestrator.py`und `src/feature_engineering/`:

   - **time_domain.py**:  Statistische Kennwerte (RMS, Kurtosis, Crest-Faktor)
   - **time_frequency.py**: Diskrete Wavelet-Transformation (DWT) zur Energieanalyse.
   - **order_analysis.py**: Ordnungsanalyse für drehzahlsynchrone Merkmale.
   - **feature_orchestrator.py**:Parallele Ausführung der Extraktion zur massiven Beschleunigung.

- ### Modelltraining

   In `src/model_training.py`, `src/deep_learning/trainer.py` etc.:

   - Training klassischer Modelle (RandomForest) und Deep-Learning-Modelle (CNN, Transformer).
   - Evaluierung (Accuracy, Precision, Recall, F1-Score)
   - Speichern von Modellen und Berichten in `results/`

---

## Wichtige Hinweise

- **Caching**: Das Projekt nutzt Marker in `.cache_status/` zur Vermeidung wiederholter Rechenoperationen. Wenn Sie Code in einem Schritt ändern (z. B. in der Datenverarbeitung), wird der Cache für alle abhängigen, nachfolgenden Schritte automatisch gelöscht.

- **Dateipfade**: Alle Pfade werden zentral in `src/config.py` verwaltet und sollten bei Bedarf dort angepasst werden.

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

Dieses Projekt steht unter der MIT License.

## Infos

Nach jeder Änderung des Codes muss die alte Marker-Datei gelöscht werden.
```bash
rm .cache_status/.data_generation.success
```
**Macbook zuklappen und trotzdem weiter rechnen lassen**
```bash
caffeinate -s python src/main.py run-all --force --fraction 0.1
```
   Was die Teile des Befehls bedeuten:

   - **caffeinate**: Der Name des Programms, das den Ruhezustand verhindert.
   - **-s**: Eine Option, die speziell das "System Sleep" verhindjrtzdjrz drj6dert, wenn das Gerät am Stromnetz hängt.
   - **python src/main.py ...**: Der ganz normale Befehl, der nun unter dem Schutz von caffeinate ausgeführt wird.

Viel Spaß :nerd_face:
