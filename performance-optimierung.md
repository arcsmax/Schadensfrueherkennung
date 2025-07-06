## Projekt-Highlight: Performance-Optimierung & Pipeline-Architektur

Dieses Projekt demonstriert nicht nur die Entwicklung eines Machine-Learning-Modells, sondern auch den Aufbau einer robusten, automatisierten und performanten End-to-End-Pipeline, wie sie in einem produktiven Umfeld erforderlich ist.

### 1. Pipeline-Design

- **Modularität**  
  Jeder Schritt (Datenverarbeitung, Feature Engineering, Modelltraining) ist in einem eigenen, gekapselten Modul untergebracht.
- **Automatisierung**  
  Ein zentraler Orchestrator (`src/main.py`) steuert den gesamten Ablauf über eine einfache Kommandozeilenschnittstelle (CLI).
- **Effizienz durch Caching**  
  Ein intelligentes Caching-System (`.cache_status/`) verhindert die wiederholte Ausführung rechenintensiver Schritte und spart Entwicklungszeit.

### 2. Performance-Analyse und Optimierung

Der rechenintensivste Schritt der Pipeline ist die Merkmalsextraktion aus über 2.000 Rohdatendateien.

#### Baseline (sequenzielle Verarbeitung)

- **Laufzeit:** 9 Minuten 42 Sekunden  
- **CPU-Auslastung:** ~100 % (ein Kern)

#### Optimierung durch Parallelisierung

- **Technologie:** `joblib.Parallel(n_jobs=-1, verbose=10)`  
- **Laufzeit (MacBook M4 Pro, 12 Kerne):** 1 Minute 20 Sekunden  
- **CPU-Auslastung:** > 1000 % (alle Kerne werden genutzt)

> **Ergebnis:**  
> Die Laufzeit des Feature-Engineering-Schritts konnte um den Faktor **7,3** reduziert werden – von nahezu 10 Minuten auf unter 1,5 Minuten.

### 3. Modellergebnis

Die Pipeline trainiert erfolgreich ein `RandomForestClassifier`, das auf dem Validierungs-Datensatz eine **Genauigkeit von 97 %** erzielt und sieben Schadensgrade zuverlässig differenziert.

#### Validierungs-Report
```plaintext
          precision    recall  f1-score   support

        0       0.90      0.97      0.93        58
        1       0.95      0.95      0.95        59
        2       0.98      0.98      0.98        58
        3       0.98      1.00      0.99        54
        4       1.00      1.00      1.00        61
        6       0.98      0.93      0.95        55
        8       1.00      0.97      0.98        59

accuracy                            0.97       404
macro avg       0.97      0.97      0.97       404
weighted avg    0.97      0.97      0.97       404
```

---

## Ansatz B: Deep Learning (End-to-End)

**Modell:** 1D Convolutional Neural Network (1D-CNN)

**Vorgehen:**
Das Modell lernt relevante Merkmale direkt aus den rohen, segmentierten Zeitreihendaten und nutzt GPU-Beschleunigung (z. B. Apple MPS). Manuelles Feature-Engineering entfällt.

**Ergebnis (Validierung):**
*(Finalen Klassifikationsbericht hier einfügen)*

### Zusammenfassende Bewertung

| Kriterium           | Klassischer Ansatz (RandomForest)             | Deep Learning (1D-CNN)                        |
| ------------------- | --------------------------------------------- | --------------------------------------------- |
| **Performance**     | Sehr hoch (F1-Score: 0.97)                   | *(Ergebnis hier eintragen)*                   |
| **Erklärbarkeit**   | Hoch: Einfluss jedes handgemachten Merkmals   | Niedrig: Modell agiert als "Black Box"       |
| **Entwicklungsaufwand** | Hoch: Feature Engineering erfordert Domänenwissen | Mittel: Expertise im Deep Learning erforderlich |
| **Trainingszeit**   | Schnell (Sekunden bis Minuten)               | Lang (Minuten bis Stunden)                   |
| **Flexibilität**    | Muss bei Datenänderungen angepasst werden    | Potenziell anpassungsfähiger an neue Muster   |

---

Diese Kombination aus starker Modell-Performance und effizienter Engineering-Praxis bildet die Grundlage für ein produktionsreifes System zur Schadensfrüherkennung.
