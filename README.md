# Industrielle Schadensfrüherkennung

Dieses Projekt ist ein Prototyp zur Früherkennung von Schäden in industriellen Anlagen mittels Vibrationssensordaten. Es implementiert eine vollständige Machine-Learning-Pipeline von der Datengenerierung über die Merkmalsextraktion bis hin zum Training und der Bewertung eines Klassifikationsmodells.

## Projektstruktur

- **/data/**: Speicherort für Roh- und verarbeitete Daten.
- **/notebooks/**: Enthält das ursprüngliche Jupyter Notebook für explorative Analysen.
- **/src/**: Der Hauptquellcode des Projekts.
  - `config.py`: Zentrale Konfiguration für Pfade, Hyperparameter etc.
  - `data_processing.py`: Skripte zur Generierung und Vorverarbeitung der Daten.
  - `feature_engineering.py`: Funktionen zur Extraktion von Merkmalen aus den Rohdaten.
  - `model_training.py`: Training und Evaluierung des Machine-Learning-Modells.
  - `utils.py`: Hilfsfunktionen, z. B. für die Visualisierung.
- `main.py`: Das Hauptskript zum Ausführen der gesamten Pipeline.
- `requirements.txt`: Liste der Python-Abhängigkeiten.

## Setup

1.  **Repository klonen:**
    ```bash
    git clone <repository-url>
    cd industrielle-schadenserkennung
    ```

2.  **Virtuelle Umgebung erstellen und aktivieren (empfohlen):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Auf Windows: venv\Scripts\activate
    ```

3.  **Abhängigkeiten installieren:**
    ```bash
    pip install -r requirements.txt
    ```

## Ausführung

Um die vollständige Pipeline auszuführen (Datengenerierung, Feature-Extraktion, Modelltraining und Auswertung), führe das Hauptskript aus:

```bash
python main.py
```

## Wichtig

Nach jeder Änderung des Codes muss die alte Marker-Datei gelöscht werden.
```bash
rm .cache_status/.data_generation.success
```

Viel Spaß :)
