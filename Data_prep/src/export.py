import pandas as pd
import numpy as np
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = BASE_DIR / 'input'
OUTPUT_DIR = BASE_DIR / 'output'
INPUT_PCAP = INPUT_DIR / '20260508_merged.pcapng'
RAW_EXPORT = OUTPUT_DIR / 'export.csv'
PROCESSED_EXPORT = OUTPUT_DIR / 'export_processed.csv'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Schritt 1: tshark ausführen
print("🔄 Starte tshark-Export...")
tshark_cmd = [
    'tshark',
    '-r', str(INPUT_PCAP),
    '-Y', 'aoecl.header.NID_PACKET == 6',
    '-T', 'fields',
    '-E', 'header=y',
    '-E', 'separator=,',
    '-e', 'frame.time_epoch',
    '-e', 'aoecl.userdata.nid6.A_GRADIENT'
]

with open(RAW_EXPORT, 'w') as f:
    result = subprocess.run(tshark_cmd, stdout=f, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"❌ tshark Fehler: {result.stderr}")
        exit(1)

print("✓ tshark Export abgeschlossen")

# Schritt 2: CSV mit manuelem Parsing lesen (wegen der komplexen Kommas)
print("🔄 Verarbeite CSV...")
timestamps = []
gradients = []

with open(RAW_EXPORT) as f:
    lines = f.readlines()
    
    for i, line in enumerate(lines):
        if i == 0:  # Header überspringen
            continue
        
        # Erste Spalte (bis erstes Komma außerhalb von Escape)
        parts = line.strip().split(',', 1)  # Nur beim ersten Komma splitten
        
        if len(parts) == 2:
            # frame.time_epoch ist in Sekunden; fuer den restlichen Workflow in us umrechnen
            timestamps.append(int(float(parts[0]) * 1_000_000))
            gradients.append(parts[1])
        elif len(parts) == 1:
            timestamps.append(int(float(parts[0]) * 1_000_000))
            gradients.append('')

# Die gradient Spalte in einzelne grad-Spalten aufteilen
grad_values = []

for gradient_str in gradients:
    # Werte parsen
    if gradient_str == '':
        values = []
    else:
        # String wird geparst - Escape-Kommas entfernen (\, → ,)
        gradient_cleaned = gradient_str.replace('\\,', ',')
        try:
            values = [float(x.strip()) for x in gradient_cleaned.split(',') if x.strip()]
        except:
            values = []
    
    # Auf 10 Elemente begrenzen
    values = values[:10]
    
    # Mit NaN auffüllen bis 10 Elemente
    while len(values) < 10:
        values.append(np.nan)
    
    grad_values.append(values)

# DataFrame erstellen
grad_df = pd.DataFrame(grad_values, columns=[f'grad[{i}]' for i in range(10)])
grad_df.insert(0, 'timestamp', timestamps)

# Wertbereich -2500 bis 2500 anwenden (außerhalb → NaN)
for i in range(10):
    col = f'grad[{i}]'
    grad_df[col] = grad_df[col].apply(lambda x: x if pd.notna(x) and -2500 <= x <= 2500 else np.nan)

# Durch 1000 teilen
for i in range(10):
    col = f'grad[{i}]'
    grad_df[col] = grad_df[col] / 1000

# Speichern
grad_df.to_csv(PROCESSED_EXPORT, index=False)
print("✓ CSV-Verarbeitung abgeschlossen")
print(f"\n📊 Ergebnisse:")
print(f"  Zeilen verarbeitet: {len(grad_df)}")
print(f"  Output: {PROCESSED_EXPORT}")
print(f"\nErste 5 Zeilen:")
print(grad_df.head())
