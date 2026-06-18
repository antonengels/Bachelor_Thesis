import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / 'output'
GRADIENT_EXPORT = OUTPUT_DIR / 'A_GRADIENT.csv'
PLOT_PATH = OUTPUT_DIR / 'plot.png'

# CSV einlesen
df = pd.read_csv(GRADIENT_EXPORT)

# Timestamp und grad[0] numerisch machen
df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
df['grad[0]'] = pd.to_numeric(df['grad[0]'], errors='coerce')

# Nur ungueltige Timestamps entfernen; NaN in grad[0] bleiben erhalten
df_plot = df[df['timestamp'].notna()].copy()

# Absolute Unix-Zeitachse (timestamp in us seit Unix-Epoch)
df_plot['datetime'] = pd.to_datetime(df_plot['timestamp'], unit='us', utc=True, errors='coerce')
df_plot = df_plot[df_plot['datetime'].notna()]

# Plot erstellen
plt.figure(figsize=(14, 6))
plt.plot(df_plot['datetime'], df_plot['grad[0]'], linewidth=0.5, alpha=0.7)

# Gesamten Zeitbereich der Daten abdecken
plt.xlim(df_plot['datetime'].min(), df_plot['datetime'].max())

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S', tz=df_plot['datetime'].dt.tz))

plt.xlabel('Zeit (UTC)')
plt.ylabel('Gradient [0] in m/s^2')
plt.title('grad[0]')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

# Speichern
plt.savefig(PLOT_PATH, dpi=150)
print(f"✓ Plot gespeichert: {PLOT_PATH}")
print(f"  Datenpunkte (inkl. NaN in grad[0]): {len(df_plot)}")
print(f"  Zeitbereich absolut (UTC): {df_plot['datetime'].min()} bis {df_plot['datetime'].max()}")

# Optional: auch anzeigen
plt.show()
