# Data_prep

Datenaufbereitungs-Pipeline einer Bachelorarbeit zur Analyse von ETCS/ATO-Fahrdaten
(Automatic Train Operation). Aus rohen Netzwerkmitschnitten (PCAPNG) werden pro
Fahrt zeitlich ausgerichtete Feature-Tabellen extrahiert, explorativ analysiert und
in einen train/val/test-Split fuer ein neuronales Netz ueberfuehrt.

## Ueberblick der Pipeline

```
input/<recording>/wireshark/*.pcapng
        │  (1) extract_features.py  ── tshark -T fields, Filter aoecl / era-subset-126
        ▼
output/<recording>/raw/features_*.parquet          (pro NID-Gruppe + Topologie)
output/trips/tripN/features_*.parquet              (globale, stillstandsbereinigte Fahrten)
        │  (2) eda.py  ── Qualitaet, Plausibilitaet, Feature-Wichtigkeit
        ▼
output/eda/eda_report.html + figures/ + _eda_trips_master.parquet (Cache/Zeitraster)
        │  (3) prepare_splits.py  ── Pruning, Feature-Engineering, Zeit-Split, Skalierung
        ▼
../Model/data/{train,val,test}.parquet + scaler.json + feature_info.json
```

## Voraussetzungen

- **Python** 3.11+ (getestet mit dem `.venv` des Projekts)
- **Wireshark / tshark >= 4.6** — der `aoecl`-Dissector ist ab Version 4.6 fest
  einkompiliert (kein Plugin noetig). `tshark` muss im `PATH` liegen oder ueber
  `--tshark` / die Umgebungsvariable `TSHARK_PATH` angegeben werden.
- Python-Pakete aus [requirements.txt](requirements.txt).

### Installation

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Verzeichnisstruktur

| Pfad | Inhalt |
|------|--------|
| `input/<recording>/wireshark/` | Rohmitschnitte (`*.pcapng`, `*.pcapng.gz`) je Aufzeichnung |
| `output/<recording>/raw/` | extrahierte Features pro NID-Gruppe (Parquet, zstd) |
| `output/trips/tripN/` | globale, stillstandsbereinigte Fahrten (+ `trip_manifest.csv`) |
| `output/eda/` | EDA-Report (`eda_report.html`), Figuren, CSV-Kennzahlen, Zeitraster-Cache |
| `src/` | Pipeline-Skripte |

## Skripte

### 1. `src/extract_features.py`

Performanter, tshark-basierter Extractor. Durchlaeuft `input/<recording>/wireshark`,
streamt alle Chunks in **einem** `tshark -T fields`-Pass, filtert die
`aoecl`-Anwendungsschicht und schreibt je `NID_PACKET`-Gruppe eine komprimierte
Parquet-Datei (LSTM-taugliches Schema). ERA-Subset-126-Paket 7 wird zu
`features_topology_radius.parquet` geparst. Optionaler Trip-Split trennt Fahrten
an Stillstand- und Datenluecken.

```powershell
# Vollstaendige Extraktion (Standard: --jobs = min(4, CPU))
.venv\Scripts\python.exe src\extract_features.py --input input --output output

# Globale Trip-Liste aus vorhandenen raw/features_*.parquet neu bauen (ohne tshark)
.venv\Scripts\python.exe src\extract_features.py --rebuild-global-trips
```

Wichtige Optionen: `--jobs`, `--recording-jobs`, `--tshark`, `--csv-export`,
`--trip-split` / `--no-trip-split`, `--trip-min-duration-minutes`,
`--trip-max-gap-seconds`, `--compression`. Resume: unveraenderte Recordings werden
anhand `output/<recording>/.extract_features_state.json` uebersprungen.

### 2. `src/eda.py`

Trip-fokussierte explorative Datenanalyse (nur `output/trips/tripN`) mit Fokus auf
Datenqualitaet, Plausibilitaet und Feature-Wichtigkeit. Baut ein gemeinsames
Zeitraster (`_eda_trips_master.parquet`) und erzeugt `output/eda/eda_report.html`
(inline SVG+PNG, interaktive Plotly-Abschnitte) sowie CSV-Kennzahlen.

```powershell
.venv\Scripts\python.exe src\eda.py --rebuild --resample-ms 200
```

Optionen: `--resample-ms` (Zeitraster, Default 200 ms), `--rebuild` (Cache neu bauen).

### 3. `src/prepare_splits.py`

Erzeugt aus dem EDA-Zeitraster-Cache den finalen Modell-Datensatz: Feature-Pruning
(Entfernen von Data-Leak- und redundanten Signalen), Feature-Engineering,
per-Trip-Imputation, zeitbasierten und nach Bremsanteil stratifizierten
Trip-Split (85/10/5) sowie eine ausschliesslich auf dem Trainingsanteil geschaetzte
Standardisierung.

```powershell
.venv\Scripts\python.exe src\prepare_splits.py
```

Optionen: `--cache` (Eingangs-Master), `--out` (Zielverzeichnis, Default
`..\Model\data`), `--period-ms`. Ausgaben: `train/val/test.parquet`, `scaler.json`,
`feature_info.json`, `split_manifest.csv`.

## Typischer End-to-End-Ablauf

```powershell
.venv\Scripts\python.exe src\extract_features.py --input input --output output
.venv\Scripts\python.exe src\extract_features.py --rebuild-global-trips
.venv\Scripts\python.exe src\eda.py --rebuild
.venv\Scripts\python.exe src\prepare_splits.py
```

## Hinweise

- Zeitbasis aller Features ist `frame.time_epoch` (ms); dadurch kein 32-bit-Rollover.
- Fehler-Sentinel- und Out-of-Range-Werte werden vor der Analyse als `NaN` maskiert
  (z.B. `V_EST` 0..4000 / Sentinel 65535, `A_EST` +/-32768).
- Label `M_ATO_RTBRq`: positiv = Beschleunigen, negativ = Bremsen; `/16384 -> [-1, 1]`.
