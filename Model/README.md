# ATO-Stellsignal-Vorhersage — Modell

Deep-Learning-Pipeline zur Vorhersage des ATO-Stellsignals (Hebelstellung /
Fahr-Brems-Kommando) aus Fahrzeug-Telemetrie. Der Model-Teil umfasst die
Modellvergleiche, das finale Training, die Testset-Evaluation sowie die
beschreibenden Datenplots der Bachelorarbeit.

## Aufgabe

Regressionsproblem: Aus einer History von 100 Schritten (20 s @ 5 Hz) mit
13 z-standardisierten Features wird die Hebelstellung `label` in `[-1, 1]`
50 Schritte (10 s) in die Zukunft vorhergesagt (positiv = Beschleunigen,
negativ = Bremsen). Die Daten sind Zeitreihen pro Fahrt (`unit` / Trip).

## Projektstruktur

```
Model/
├── data/                       # Eingabedaten (nicht versioniert)
│   ├── train.parquet           # ~1,96 Mio. Zeilen
│   ├── val.parquet             # ~392k Zeilen
│   ├── test.parquet            # ~261k Zeilen
│   ├── feature_info.json
│   ├── scaler.json             # Mittelwerte/Streuungen (z-Standardisierung)
│   └── split_manifest.csv
├── src/
│   ├── model_comparison.py            # Basisvergleich MLP vs. LSTM vs. GRU
│   ├── lstm_architecture_comparison.py# LSTM-Architektursuche (Layer/Breiten)
│   ├── loss_optimizer_comparison.py   # Sweep Loss x Optimizer
│   ├── model.py                       # Finales Modelltraining
│   ├── test.py                        # Finale Testset-Evaluation (Kap. 7)
│   └── plot.py                        # Beschreibende Datenplots
├── reports/                    # Erzeugte HTML-Reports, Plots, Metriken, Checkpoints
├── requirements.txt
└── README.md
```

## Voraussetzungen

- Python 3.14 (getestet mit 3.14.5), Windows, CPU-only
- Abhängigkeiten: torch, numpy, pandas, matplotlib, pyarrow

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Daten

Die Parquet-Splits (`train`/`val`/`test`) und `scaler.json` müssen unter `data/`
liegen. Die Feature-Spalten sind bereits z-standardisiert (auf dem Train-Split
gefittet); `scaler.json` erlaubt die Rückrechnung auf Rohwerte (`raw = z*std+mean`).

13 Features: `V_EST, V_PERMITTED, A_EST, A_GRADIENT, D_STPDISTANCE, a_num, jerk,
v_headroom, v_ratio, a_est_roll_mean_2s, v_roll_std_2s, stop_proximity, grad_x_v`.
Zielspalte: `label` (geglättet, in `[-1, 1]`); zusätzlich `label_raw`.

## Nutzung

Alle Skripte werden aus dem Projektwurzelverzeichnis mit dem venv-Interpreter
ausgeführt und schreiben ihre Ergebnisse nach `reports/`.

### 1. Modellvergleich (MLP vs. LSTM vs. GRU)

```powershell
.\.venv\Scripts\python.exe src\model_comparison.py
```

Baseline-Vergleich der drei Architekturfamilien (parameter-budget-abgeglichen),
Bewertung ausschließlich auf dem Validierungs-Split. Ausgaben:
`model_comparison_report.html`, Metriken als CSV/JSON.

### 2. LSTM-Architektursuche

```powershell
.\.venv\Scripts\python.exe src\lstm_architecture_comparison.py
```

Brute-Force-Sweep über Layertiefen und Neuronenbreiten pro Layer. Der Lauf ist
fortsetzbar (Checkpoints unter `reports/lstm_architecture_comparison_checkpoints/`).
Ausgaben: `lstm_architecture_comparison_report.html`, Metriken CSV/JSON.

### 3. Loss- / Optimizer-Sweep

```powershell
.\.venv\Scripts\python.exe src\loss_optimizer_comparison.py
```

Sweep über Loss-Funktionen x Optimizer, inkrementelle Ausgabe nach jedem Lauf.

### 4. Finales Training

```powershell
.\.venv\Scripts\python.exe src\model.py
```

Trainiert das finale Modell (4-Layer-LSTM mit Hidden-Sizes `128-64-256-128`,
SmoothL1-Loss, Adam, Dropout, Weight-Decay, Grad-Clipping, additives
Gauß-Rauschen auf den Trainingsfeatures). Alle Hyperparameter sind als globale
Variablen am Kopf der Datei einstellbar. Das Training ist **fortsetzbar**: bei
Neustart mit identischer Konfiguration wird der Resume-Checkpoint geladen und
ab der nächsten Epoche weitertrainiert; bei geänderter Konfiguration beginnt ein
neuer Lauf.

Ausgaben:
- `reports/model_final_report.html`, `model_final_curves.png`
- `reports/model_final_metrics.csv` / `.json`
- `reports/model_checkpoints/training_checkpoint.pt` (Resume-Checkpoint)
- `reports/model_checkpoints/best_model.pt` (bestes Modell nach Val-Loss)

### 5. Testset-Evaluation

```powershell
.\.venv\Scripts\python.exe src\test.py
```

Lädt das beste Modell (`reports/model_checkpoints/best_model.pt`, Fallback:
`best_state` im Resume-Checkpoint) und erzeugt die Auswertungen aus Kapitel 7
(Ergebnistabelle vs. Persistenz-Baseline, Scatter, Residuen, Richtungs-
Konfusionsmatrix, klassenweise Precision/Recall/F1, Zeitreihen-Overlays,
Fehler nach Trip-Charakteristik). Erfordert einen mit den aktuellen 13 Features
trainierten Checkpoint.

Ausgaben: `reports/test_evaluation_report.html`, `test_eval_*.png`,
`test_evaluation_metrics.csv` / `.json`, `test_evaluation_trip_metrics.csv`.

### 6. Beschreibende Datenplots

```powershell
.\.venv\Scripts\python.exe src\plot.py
```

Erzeugt beschreibende Plots der Datengrundlage (A_EST-Histogramm,
Beispieltrips, Label-Übersicht, Vorhersage-Overlay eines Beispieltrips). Kein
Training. Ausgaben: `reports/plot_*.png` (300 dpi).

## Reihenfolge

Die Vergleichs- und Sweep-Skripte (1–3) dienen der Modellauswahl. Für die
finalen Ergebnisse: erst `model.py` (Training) ausführen, dann `test.py`
(Evaluation). `plot.py` benötigt für den Vorhersage-Plot ebenfalls einen
trainierten Checkpoint.

## Hinweise

- Alle Läufe sind CPU-basiert; die Geräteauswahl erfolgt dynamisch (CUDA/TPU,
  falls verfügbar, sonst CPU).
- Reports sind self-contained (Plots als base64 eingebettet).
- `data/` und `reports/` sind nicht versioniert.
