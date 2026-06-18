import pandas as pd
import numpy as np
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = BASE_DIR / 'input'
OUTPUT_DIR = BASE_DIR / 'output'
INPUT_PCAP = INPUT_DIR / '20260508_merged.pcapng'
NID6_EXPORT = OUTPUT_DIR / 'NID_6.csv'
NID31_EXPORT = OUTPUT_DIR / 'NID_31.csv'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_tshark_export(display_filter: str, field_name: str) -> list[str]:
    tshark_cmd = [
        'tshark',
        '-r', str(INPUT_PCAP),
        '-Y', display_filter,
        '-T', 'fields',
        '-e', 'frame.time_epoch',
        '-e', field_name,
    ]

    result = subprocess.run(tshark_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"tshark Fehler fuer {field_name}: {result.stderr}")
        raise SystemExit(1)

    return [line for line in result.stdout.splitlines() if line.strip()]


def parse_timestamp(raw_value: str) -> int | None:
    try:
        # frame.time_epoch ist die Arrival Time in Sekunden; fuer den Workflow weiter in us.
        return int(float(raw_value) * 1_000_000)
    except (TypeError, ValueError):
        return None


def validate_and_normalize_velocity(raw_value_str: str) -> float:
    """
    Validiert und normiert Geschwindigkeiten (V_EST, V_MRSP, V_PERMITTED).
    
    Gültiger Roh-Bereich: 0 bis 16666
    Fehlerwert: 65535 → np.nan
    Konvertierung: / 100.0 → m/s
    """
    try:
        raw_value = float(raw_value_str.strip())
    except (TypeError, ValueError):
        return np.nan
    
    if raw_value == 65535 or raw_value < 0 or raw_value > 16666:
        return np.nan
    
    return raw_value / 100.0


def validate_and_normalize_acceleration(raw_value_str: str) -> float:
    """
    Validiert und normiert die Ist-Beschleunigung (A_EST).
    
    Gültiger Roh-Bereich: -5000 bis +5000
    Fehlerwert: 32767 → np.nan
    Konvertierung: / 1000.0 → m/s²
    """
    try:
        raw_value = float(raw_value_str.strip())
    except (TypeError, ValueError):
        return np.nan
    
    if raw_value == 32767 or raw_value < -5000 or raw_value > 5000:
        return np.nan
    
    return raw_value / 1000.0


def validate_and_normalize_gradient(raw_value_str: str) -> float:
    """
    Validiert und normiert ein einzelnes Gradient-Element.
    
    Gültiger Roh-Bereich: -2500 bis +2500
    Fehlerwert: 32767 → np.nan
    Konvertierung: / 1000.0 → m/s²
    """
    try:
        raw_value = float(raw_value_str.strip())
    except (TypeError, ValueError):
        return np.nan
    
    if raw_value == 32767 or raw_value < -2500 or raw_value > 2500:
        return np.nan
    
    return raw_value / 1000.0


def parse_packet_6(raw_values: dict) -> dict:
    """
    Parst und validiert ein NID_PACKET == 6.
    
    Erwartet ein Dictionary mit den tshark-Roh-Werten (als Strings).
    Gibt ein Dictionary mit validierten, normierten Werten zurück.
    """
    result = {
        'v_est': validate_and_normalize_velocity(raw_values.get('v_est', '')),
        'a_est': validate_and_normalize_acceleration(raw_values.get('a_est', '')),
        'v_mrsp': validate_and_normalize_velocity(raw_values.get('v_mrsp', '')),
        'v_permitted': validate_and_normalize_velocity(raw_values.get('v_permitted', '')),
    }
    
    # Gradienten-Array parsen (max 10 Elemente)
    gradient_str = raw_values.get('a_gradient', '')
    gradient_values = []
    
    if gradient_str.strip():
        # Escape-Kommas entfernen und splitten
        gradient_cleaned = gradient_str.replace('\\,', ',')
        parts = [p.strip() for p in gradient_cleaned.split(',')]
        gradient_values = [validate_and_normalize_gradient(p) for p in parts[:10]]
    
    # Auf exakt 10 Elemente auffüllen mit np.nan
    while len(gradient_values) < 10:
        gradient_values.append(np.nan)
    
    for i, val in enumerate(gradient_values):
        result[f'grad[{i}]'] = val
    
    return result


def build_nid6_export() -> pd.DataFrame:
    print("Verarbeite NID_6 (Fahrzeugdynamik)...")
    
    # tshark mit allen benötigten Feldern aufrufen
    tshark_cmd = [
        'tshark',
        '-r', str(INPUT_PCAP),
        '-Y', 'aoecl.header.NID_PACKET == 6',
        '-T', 'fields',
        '-e', 'frame.time_epoch',
        '-e', 'aoecl.userdata.nid6.V_EST',
        '-e', 'aoecl.userdata.nid6.A_EST',
        '-e', 'aoecl.userdata.nid6.V_MRSP',
        '-e', 'aoecl.userdata.nid6.V_PERMITTED',
        '-e', 'aoecl.userdata.nid6.A_GRADIENT',
    ]

    result = subprocess.run(tshark_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"tshark Fehler fuer NID_6: {result.stderr}")
        raise SystemExit(1)

    lines = [line for line in result.stdout.splitlines() if line.strip()]
    
    rows = []
    for line in lines:
        parts = line.split('\t')
        
        timestamp = parse_timestamp(parts[0]) if len(parts) > 0 else None
        if timestamp is None:
            continue
        
        # Raw-Werte aus tshark
        raw_values = {
            'v_est': parts[1] if len(parts) > 1 else '',
            'a_est': parts[2] if len(parts) > 2 else '',
            'v_mrsp': parts[3] if len(parts) > 3 else '',
            'v_permitted': parts[4] if len(parts) > 4 else '',
            'a_gradient': parts[5] if len(parts) > 5 else '',
        }
        
        # Parsing und Validierung
        parsed = parse_packet_6(raw_values)
        parsed['timestamp'] = timestamp
        
        rows.append(parsed)
    
    # DataFrame aus allen Zeilen erstellen
    if not rows:
        print("⚠ Keine NID_6-Pakete gefunden!")
        return pd.DataFrame()
    
    nid6_df = pd.DataFrame(rows)
    
    # Spalten-Reihenfolge: timestamp, dann Velocities, dann A_EST, dann Gradienten
    col_order = ['timestamp', 'v_est', 'a_est', 'v_mrsp', 'v_permitted'] + [f'grad[{i}]' for i in range(10)]
    nid6_df = nid6_df[col_order]
    
    nid6_df.to_csv(NID6_EXPORT, index=False)
    print(f"✓ NID_6 gespeichert: {NID6_EXPORT}")
    print(f"  Zeilen verarbeitet: {len(nid6_df)}")
    
    return nid6_df


def normalize_rtbrq(raw_value: float) -> float:
    if raw_value < -16384 or raw_value > 16384:
        return np.nan
    return raw_value / 16384


def build_nid31_export() -> pd.DataFrame:
    print("Verarbeite NID_31...")
    lines = run_tshark_export('aoecl.header.NID_PACKET == 31', 'aoecl.userdata.nid31.M_ATO_RTBRq')

    rows = []
    for line in lines:
        parts = line.split('\t', 1)
        timestamp = parse_timestamp(parts[0])
        if timestamp is None:
            continue

        value_str = parts[1].strip() if len(parts) == 2 else ''
        try:
            raw_value = float(value_str)
        except ValueError:
            normalized_value = np.nan
        else:
            normalized_value = normalize_rtbrq(raw_value)

        rows.append({'timestamp': timestamp, 'value': normalized_value})

    nid31_df = pd.DataFrame(rows)
    nid31_df.to_csv(NID31_EXPORT, index=False)
    print(f"✓ NID_31 gespeichert: {NID31_EXPORT}")
    print(f"  Zeilen verarbeitet: {len(nid31_df)}")
    return nid31_df


print("Starte tshark-Export...")
nid6_df = build_nid6_export()
nid31_df = build_nid31_export()

print("\nErgebnisse:")
print(f"  Output: {NID6_EXPORT}")
print(f"  Output: {NID31_EXPORT}")
print("\nErste 5 Zeilen NID_6:")
print(nid6_df.head())
print("\nErste 5 Zeilen NID_31:")
print(nid31_df.head())
