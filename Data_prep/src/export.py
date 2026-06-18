import pandas as pd
import numpy as np
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = BASE_DIR / 'input'
OUTPUT_DIR = BASE_DIR / 'output'
INPUT_PCAP = INPUT_DIR / '20260508_merged.pcapng'
GRADIENT_EXPORT = OUTPUT_DIR / 'A_GRADIENT.csv'
RTBRQ_EXPORT = OUTPUT_DIR / 'M_ATO_RTBRq.csv'

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


def build_gradient_export() -> pd.DataFrame:
    print("Verarbeite A_GRADIENT...")
    lines = run_tshark_export('aoecl.header.NID_PACKET == 6', 'aoecl.userdata.nid6.A_GRADIENT')

    timestamps = []
    grad_values = []

    for line in lines:
        parts = line.split('\t', 1)
        timestamp = parse_timestamp(parts[0])
        if timestamp is None:
            continue

        gradient_str = parts[1] if len(parts) == 2 else ''
        gradient_cleaned = gradient_str.replace('\\,', ',')

        try:
            values = [float(value.strip()) for value in gradient_cleaned.split(',') if value.strip()]
        except ValueError:
            values = []

        values = values[:10]
        while len(values) < 10:
            values.append(np.nan)

        timestamps.append(timestamp)
        grad_values.append(values)

    grad_df = pd.DataFrame(grad_values, columns=[f'grad[{i}]' for i in range(10)])
    grad_df.insert(0, 'timestamp', timestamps)

    for i in range(10):
        col = f'grad[{i}]'
        grad_df[col] = grad_df[col].apply(lambda x: x if pd.notna(x) and -2500 <= x <= 2500 else np.nan)
        grad_df[col] = grad_df[col] / 1000

    grad_df.to_csv(GRADIENT_EXPORT, index=False)
    print(f"✓ A_GRADIENT gespeichert: {GRADIENT_EXPORT}")
    print(f"  Zeilen verarbeitet: {len(grad_df)}")
    return grad_df


def normalize_rtbrq(raw_value: float) -> float:
    if raw_value < -16384 or raw_value > 16384:
        return np.nan
    return raw_value / 16384


def build_rtbrq_export() -> pd.DataFrame:
    print("Verarbeite M_ATO_RTBRq...")
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

        rows.append({'timestamp': timestamp, 'M_ATO_RTBRq': normalized_value})

    rtbrq_df = pd.DataFrame(rows)
    rtbrq_df.to_csv(RTBRQ_EXPORT, index=False)
    print(f"✓ M_ATO_RTBRq gespeichert: {RTBRQ_EXPORT}")
    print(f"  Zeilen verarbeitet: {len(rtbrq_df)}")
    return rtbrq_df


print("Starte tshark-Export...")
grad_df = build_gradient_export()
rtbrq_df = build_rtbrq_export()

print("\nErgebnisse:")
print(f"  Output: {GRADIENT_EXPORT}")
print(f"  Output: {RTBRQ_EXPORT}")
print("\nErste 5 Zeilen A_GRADIENT:")
print(grad_df.head())
print("\nErste 5 Zeilen M_ATO_RTBRq:")
print(rtbrq_df.head())
