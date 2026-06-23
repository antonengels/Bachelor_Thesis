import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "output"

CSV_INPUTS = {
    "nid1": OUTPUT_DIR / "NID_1.csv",
    "nid6": OUTPUT_DIR / "NID_6.csv",
    "nid31": OUTPUT_DIR / "NID_31.csv",
    "nid32": OUTPUT_DIR / "NID_32.csv",
}
PARQUET_INPUTS = {
    "nid1": OUTPUT_DIR / "NID_1.parquet",
    "nid6": OUTPUT_DIR / "NID_6.parquet",
    "nid31": OUTPUT_DIR / "NID_31.parquet",
    "nid32": OUTPUT_DIR / "NID_32.parquet",
}

DEFAULT_CSV_OUTPUT = OUTPUT_DIR / "merged.csv"
DEFAULT_PARQUET_OUTPUT = OUTPUT_DIR / "merged.parquet"

NID6_COLUMNS = ["v_est", "a_est", "v_mrsp", "v_permitted"] + [f"grad[{i}]" for i in range(10)]
CONTINUOUS_FEATURE_COLUMNS = ["D_STPDISTANCE"] + NID6_COLUMNS + ["M_RST_TBsetVal"]
DISCRETE_FEATURE_COLUMNS = ["M_RST_SlipSlide"]


def load_packet_frame(path: Path, expected_columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {path}")

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if "timestamp" not in df.columns:
        raise ValueError(f"Spalte 'timestamp' fehlt in: {path}")

    columns = ["timestamp"] + [col for col in expected_columns if col in df.columns]
    df = df[columns].copy()

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df[df["timestamp"].notna()].copy()
    df["timestamp"] = df["timestamp"].astype("int64")

    for col in columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if columns[1:]:
        aggregations = {col: "mean" for col in columns[1:]}
        df = df.groupby("timestamp", as_index=False).agg(aggregations)
    else:
        df = df[["timestamp"]].drop_duplicates()

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="us", utc=True)
    return df.sort_values("datetime").reset_index(drop=True)


def build_common_grid(frames: list[pd.DataFrame], freq: str) -> pd.DatetimeIndex:
    starts = [int(frame["timestamp"].min()) for frame in frames if not frame.empty]
    ends = [int(frame["timestamp"].max()) for frame in frames if not frame.empty]
    if not starts or not ends:
        raise ValueError("Keine gueltigen Eingabedaten vorhanden.")

    start_us = max(starts)
    end_us = min(ends)
    if start_us >= end_us:
        raise ValueError("Kein gemeinsames Zeitfenster zwischen den NID-Dateien gefunden.")

    start_dt = pd.to_datetime(start_us, unit="us", utc=True).ceil(freq)
    end_dt = pd.to_datetime(end_us, unit="us", utc=True).floor(freq)

    if start_dt > end_dt:
        raise ValueError("Zeitfenster ist nach Raster-Ausrichtung leer. Bitte andere Frequenz waehlen.")

    grid = pd.date_range(start=start_dt, end=end_dt, freq=freq, tz="UTC")
    if grid.empty:
        raise ValueError("Leeres Raster erzeugt. Bitte Frequenz oder Eingabedaten pruefen.")

    return grid


def align_last_known(
    grid_index: pd.DatetimeIndex,
    source: pd.DataFrame,
    value_columns: list[str],
    tolerance: pd.Timedelta,
) -> pd.DataFrame:
    if source.empty or not value_columns:
        return pd.DataFrame(index=grid_index)

    source_cols = ["datetime"] + [col for col in value_columns if col in source.columns]
    if len(source_cols) == 1:
        return pd.DataFrame(index=grid_index)

    grid_df = pd.DataFrame({"datetime": grid_index})
    source_df = source[source_cols].drop_duplicates(subset=["datetime"], keep="last")
    source_df = source_df.sort_values("datetime")

    merged = pd.merge_asof(
        grid_df,
        source_df,
        on="datetime",
        direction="backward",
        tolerance=tolerance,
    )
    return merged.set_index("datetime")


def fill_continuous_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def fill_discrete_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            filled = pd.to_numeric(df[col], errors="coerce")
            if col == "M_RST_SlipSlide":
                filled = filled.round().clip(lower=0, upper=1)
            df[col] = filled
    return df


def remove_short_zero_dropouts(series: pd.Series, max_zero_run: int, zero_eps: float = 1e-9) -> pd.Series:
    values = series.to_numpy(dtype=float, copy=True)
    n = len(values)
    i = 0

    while i < n:
        is_zero = not np.isnan(values[i]) and abs(values[i]) <= zero_eps
        if not is_zero:
            i += 1
            continue

        run_start = i
        while i < n and (not np.isnan(values[i])) and abs(values[i]) <= zero_eps:
            i += 1
        run_end = i

        run_len = run_end - run_start
        left_idx = run_start - 1
        right_idx = run_end

        if run_len <= max_zero_run and left_idx >= 0 and right_idx < n:
            left_val = values[left_idx]
            right_val = values[right_idx]
            left_non_zero = (not np.isnan(left_val)) and abs(left_val) > zero_eps
            right_non_zero = (not np.isnan(right_val)) and abs(right_val) > zero_eps
            if left_non_zero and right_non_zero:
                interp = np.linspace(left_val, right_val, run_len + 2, dtype=float)[1:-1]
                values[run_start:run_end] = interp

    return pd.Series(values, index=series.index, name=series.name)


def create_merged_dataset(
    input_paths: dict[str, Path],
    freq: str,
    moving_average_window: int,
    max_zero_run: int,
    max_age_ms: int,
) -> pd.DataFrame:
    nid1 = load_packet_frame(input_paths["nid1"], ["D_STPDISTANCE"])
    nid6 = load_packet_frame(input_paths["nid6"], NID6_COLUMNS)
    nid31 = load_packet_frame(input_paths["nid31"], ["value"]).rename(columns={"value": "M_ATO_RTBRq_raw"})
    nid32 = load_packet_frame(input_paths["nid32"], ["M_RST_TBsetVal", "M_RST_SlipSlide"])

    grid = build_common_grid([nid1, nid6, nid31, nid32], freq=freq)
    tolerance = pd.Timedelta(milliseconds=max_age_ms)

    merged = pd.DataFrame(index=grid)
    merged.index.name = "datetime"

    merged = merged.join(align_last_known(grid, nid1, ["D_STPDISTANCE"], tolerance), how="left")
    merged = merged.join(align_last_known(grid, nid6, NID6_COLUMNS, tolerance), how="left")
    merged = merged.join(align_last_known(grid, nid32, ["M_RST_TBsetVal", "M_RST_SlipSlide"], tolerance), how="left")
    merged = merged.join(align_last_known(grid, nid31, ["M_ATO_RTBRq_raw"], tolerance), how="left")

    merged = fill_continuous_columns(merged, CONTINUOUS_FEATURE_COLUMNS + ["M_ATO_RTBRq_raw"])
    merged = fill_discrete_columns(merged, DISCRETE_FEATURE_COLUMNS)

    merged["M_ATO_RTBRq_raw"] = merged["M_ATO_RTBRq_raw"].clip(lower=-1.0, upper=1.0)
    merged["M_ATO_RTBRq_deglitched"] = remove_short_zero_dropouts(
        merged["M_ATO_RTBRq_raw"],
        max_zero_run=max_zero_run,
    ).clip(lower=-1.0, upper=1.0)
    merged["M_ATO_RTBRq_smooth"] = (
        merged["M_ATO_RTBRq_deglitched"]
        .rolling(window=moving_average_window, min_periods=1, center=True)
        .mean()
        .clip(lower=-1.0, upper=1.0)
    )
    merged.loc[merged["M_ATO_RTBRq_deglitched"].isna(), "M_ATO_RTBRq_smooth"] = np.nan

    merged["timestamp"] = (merged.index.astype("int64") // 1000).astype("int64")
    merged["datetime_utc"] = merged.index.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    front_columns = ["timestamp", "datetime_utc"]
    remaining = [col for col in merged.columns if col not in front_columns]
    merged = merged[front_columns + remaining]

    if "M_RST_SlipSlide" in merged.columns:
        merged["M_RST_SlipSlide"] = merged["M_RST_SlipSlide"].round().astype("Int64")

    return merged.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fasst NID_1, NID_6, NID_31 und NID_32 auf ein gemeinsames Zeitraster zusammen, "
            "korrigiert kurze NID_31-Null-Dropouts und erstellt ein geglaettetes Label."
        )
    )
    parser.add_argument("--freq", default="100ms", help="Ziel-Raster, z.B. 100ms oder 50ms.")
    parser.add_argument(
        "--moving-average-window",
        type=int,
        default=5,
        help="Fenstergroesse fuer den Moving Average auf M_ATO_RTBRq (in Samples).",
    )
    parser.add_argument(
        "--max-zero-run",
        type=int,
        default=2,
        help="Maximale Laenge einer Null-Sequenz, die als Glitch ersetzt wird.",
    )
    parser.add_argument(
        "--max-age-ms",
        type=int,
        default=1000,
        help="Maximales Alter beim asof-Merge (stale sample cutoff).",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=DEFAULT_CSV_OUTPUT,
        help=f"CSV-Ausgabedatei (Default: {DEFAULT_CSV_OUTPUT}).",
    )
    parser.add_argument(
        "--parquet-output",
        type=Path,
        default=DEFAULT_PARQUET_OUTPUT,
        help=f"Parquet-Ausgabedatei (Default: {DEFAULT_PARQUET_OUTPUT}).",
    )
    return parser.parse_args()


def missing_inputs(input_paths: dict[str, Path]) -> list[Path]:
    return [path for path in input_paths.values() if not path.exists()]


def main() -> None:
    args = parse_args()

    if args.moving_average_window < 1:
        raise ValueError("--moving-average-window muss >= 1 sein.")
    if args.max_zero_run < 1:
        raise ValueError("--max-zero-run muss >= 1 sein.")
    if args.max_age_ms < 1:
        raise ValueError("--max-age-ms muss >= 1 sein.")

    wrote_any_output = False

    csv_missing = missing_inputs(CSV_INPUTS)
    if not csv_missing:
        merged_csv = create_merged_dataset(
            input_paths=CSV_INPUTS,
            freq=args.freq,
            moving_average_window=args.moving_average_window,
            max_zero_run=args.max_zero_run,
            max_age_ms=args.max_age_ms,
        )
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        merged_csv.to_csv(args.csv_output, index=False)
        wrote_any_output = True
        print(f"OK: Merged CSV gespeichert: {args.csv_output}")
        print(f"  Zeilen: {len(merged_csv)}")
        print(f"  Spalten: {len(merged_csv.columns)}")
    else:
        print("Info: CSV-Merge uebersprungen (fehlende Eingaben).")
        for path in csv_missing:
            print(f"  - fehlt: {path}")

    parquet_missing = missing_inputs(PARQUET_INPUTS)
    if not parquet_missing:
        merged_parquet = create_merged_dataset(
            input_paths=PARQUET_INPUTS,
            freq=args.freq,
            moving_average_window=args.moving_average_window,
            max_zero_run=args.max_zero_run,
            max_age_ms=args.max_age_ms,
        )
        args.parquet_output.parent.mkdir(parents=True, exist_ok=True)
        merged_parquet.to_parquet(args.parquet_output, index=False)
        wrote_any_output = True
        print(f"OK: Merged Parquet gespeichert: {args.parquet_output}")
        print(f"  Zeilen: {len(merged_parquet)}")
        print(f"  Spalten: {len(merged_parquet.columns)}")
    else:
        print("Info: Parquet-Merge uebersprungen (fehlende Eingaben).")
        for path in parquet_missing:
            print(f"  - fehlt: {path}")

    if not wrote_any_output:
        raise FileNotFoundError(
            "Kein Merge erzeugt, weil weder ein vollstaendiger CSV- noch Parquet-Eingabesatz vorhanden ist."
        )

    print("  Label-Spalten: M_ATO_RTBRq_raw, M_ATO_RTBRq_deglitched, M_ATO_RTBRq_smooth")


if __name__ == "__main__":
    main()
