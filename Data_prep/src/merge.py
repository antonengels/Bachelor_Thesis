import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "output"
DEFAULT_INPUT_DIR = OUTPUT_DIR

DEFAULT_CSV_OUTPUT = OUTPUT_DIR / "merged.csv"
DEFAULT_PARQUET_OUTPUT = OUTPUT_DIR / "merged.parquet"

NID6_COLUMNS = ["v_est", "a_est", "v_mrsp", "v_permitted"] + [f"grad[{i}]" for i in range(10)]
CONTINUOUS_FEATURE_COLUMNS = ["D_STPDISTANCE"] + NID6_COLUMNS + ["M_RST_TBsetVal"]
DISCRETE_FEATURE_COLUMNS = ["M_RST_SlipSlide"]


def build_input_paths(input_dir: Path, input_prefix: str, suffix: str) -> dict[str, Path]:
    prefix_str = f"{input_prefix}_" if input_prefix else ""
    return {
        "nid1": input_dir / f"{prefix_str}NID_1{suffix}",
        "nid6": input_dir / f"{prefix_str}NID_6{suffix}",
        "nid31": input_dir / f"{prefix_str}NID_31{suffix}",
        "nid32": input_dir / f"{prefix_str}NID_32{suffix}",
    }


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

    timestamp_ns = merged.index.to_numpy(dtype="datetime64[ns]").astype("int64")
    merged["timestamp"] = (timestamp_ns // 1000).astype("int64")
    merged["datetime_utc"] = merged.index.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    front_columns = ["timestamp", "datetime_utc"]
    remaining = [col for col in merged.columns if col not in front_columns]
    merged = merged[front_columns + remaining]

    if "M_RST_SlipSlide" in merged.columns:
        merged["M_RST_SlipSlide"] = merged["M_RST_SlipSlide"].round().astype("Int64")

    return merged.reset_index(drop=True)


def find_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None

    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i - 1))
            start = None

    if start is not None:
        runs.append((start, len(mask) - 1))

    return runs


def estimate_sample_interval_us(df: pd.DataFrame) -> int:
    if "timestamp" not in df.columns or len(df) < 2:
        return 100_000

    ts = pd.to_numeric(df["timestamp"], errors="coerce").dropna().astype("int64")
    if len(ts) < 2:
        return 100_000

    diffs = np.diff(ts.to_numpy())
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 100_000

    return int(np.median(diffs))


def choose_brake_column(df: pd.DataFrame) -> str | None:
    col = "M_ATO_RTBRq_raw"
    if col in df.columns and is_numeric_dtype(df[col]):
        return col
    return None


def filter_standstill_and_split(
    merged: pd.DataFrame,
    standstill_min_minutes: int,
    standstill_buffer_minutes: int,
) -> tuple[pd.DataFrame, int, str | None]:
    if merged.empty:
        return merged.copy(), 0, None

    if "v_est" not in merged.columns:
        return merged.copy(), 0, None

    brake_col = choose_brake_column(merged)
    if brake_col is None:
        return merged.copy(), 0, None

    df = merged.copy().reset_index(drop=True)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
    df["v_est"] = pd.to_numeric(df["v_est"], errors="coerce")
    df[brake_col] = pd.to_numeric(df[brake_col], errors="coerce")

    brake_threshold = 0.05
    v_est_threshold = 0.1
    gap_removed_rows = 0

    v_est_missing_runs = find_true_runs(df["v_est"].isna().to_numpy())
    is_standstill = (
        (df["v_est"].isna() | (df["v_est"].abs() < v_est_threshold))
        & df[brake_col].notna()
        & ((df[brake_col].abs() < brake_threshold) | (df[brake_col] < -brake_threshold))
    ).to_numpy()

    sample_interval_us = estimate_sample_interval_us(df)
    min_duration_us = standstill_min_minutes * 60 * 1_000_000
    buffer_duration_us = standstill_buffer_minutes * 60 * 1_000_000
    min_samples = max(1, int(np.ceil(min_duration_us / sample_interval_us)))
    buffer_samples = max(0, int(np.ceil(buffer_duration_us / sample_interval_us)))

    remove_mask = np.zeros(len(df), dtype=bool)
    for run_start, run_end in v_est_missing_runs:
        run_len = run_end - run_start + 1
        if run_len >= min_samples:
            remove_mask[run_start : run_end + 1] = True
            gap_removed_rows += run_len

    for run_start, run_end in find_true_runs(is_standstill):
        run_len = run_end - run_start + 1
        if run_len < min_samples:
            continue

        is_leading_run = run_start == 0
        is_trailing_run = run_end == len(df) - 1

        if is_leading_run or is_trailing_run:
            remove_mask[run_start : run_end + 1] = True
            continue

        remove_start = run_start + buffer_samples
        remove_end = run_end - buffer_samples
        if remove_start <= remove_end:
            remove_mask[remove_start : remove_end + 1] = True

    kept_df = df[~remove_mask].copy()
    if kept_df.empty:
        return kept_df, int(remove_mask.sum()), brake_col

    kept_positions = np.flatnonzero(~remove_mask)
    segment_breaks = np.diff(kept_positions) > 1
    trip_ids = np.cumsum(np.r_[True, segment_breaks]).astype(int)
    kept_df["trip_id"] = trip_ids

    return kept_df.reset_index(drop=True), int(remove_mask.sum()), brake_col


def write_trip_files(
    df: pd.DataFrame,
    output_path: Path,
    trip_subdir_prefix: str,
    file_label_prefix: str,
) -> int:
    if df.empty or "trip_id" not in df.columns:
        return 0

    created = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extension = output_path.suffix.lower()

    for idx, trip_id in enumerate(sorted(df["trip_id"].dropna().unique()), start=1):
        trip_id_int = int(trip_id)
        trip_df = df[df["trip_id"] == trip_id_int].copy()
        trip_dir = output_path.parent / f"{trip_subdir_prefix}{idx}"
        trip_dir.mkdir(parents=True, exist_ok=True)
        trip_file = trip_dir / f"{file_label_prefix}{trip_subdir_prefix}{idx}{extension}"

        if extension == ".parquet":
            trip_df.to_parquet(trip_file, index=False)
        else:
            trip_df.to_csv(trip_file, index=False)
        created += 1

    return created


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fasst NID_1, NID_6, NID_31 und NID_32 auf ein gemeinsames Zeitraster zusammen, "
            "korrigiert kurze NID_31-Null-Dropouts und erstellt ein geglaettetes Label."
        )
    )
    parser.add_argument("--freq", default="100ms", help="Ziel-Raster, z.B. 100ms oder 50ms.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Eingabeverzeichnis fuer NID-Dateien (Default: {DEFAULT_INPUT_DIR}).",
    )
    parser.add_argument(
        "--input-prefix",
        type=str,
        default="",
        help="Praefix fuer Eingabedateien (z.B. '20251015' fuer '20251015_NID_1.parquet').",
    )
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
    parser.add_argument(
        "--standstill-min-minutes",
        type=int,
        default=10,
        help="Stillstand wird ab dieser Dauer erkannt (in Minuten, Default: 10).",
    )
    parser.add_argument(
        "--standstill-buffer-minutes",
        type=int,
        default=10,
        help="Puffer vor/nach erkanntem Stillstand, der erhalten bleibt (in Minuten, Default: 10).",
    )
    parser.add_argument(
        "--no-trip-split",
        action="store_true",
        help="Keine separaten Trip-Dateien erzeugen.",
    )
    parser.add_argument(
        "--trips-only",
        action="store_true",
        help="Nur Trip-Dateien schreiben, keine merged.csv/merged.parquet speichern.",
    )
    parser.add_argument(
        "--trip-subdir-prefix",
        type=str,
        default="trip",
        help="Praefix fuer Unterordner der Trip-Dateien (Default: trip -> trip1, trip2, ...).",
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
    if args.standstill_min_minutes < 1:
        raise ValueError("--standstill-min-minutes muss >= 1 sein.")
    if args.standstill_buffer_minutes < 0:
        raise ValueError("--standstill-buffer-minutes muss >= 0 sein.")
    if not args.trip_subdir_prefix.strip():
        raise ValueError("--trip-subdir-prefix darf nicht leer sein.")
    if args.trips_only and args.no_trip_split:
        raise ValueError("--trips-only erfordert aktiven Trip-Split (kein --no-trip-split).")

    if args.trips_only:
        if args.csv_output.exists():
            args.csv_output.unlink()
            print(f"Info: Vorhandene merged.csv entfernt: {args.csv_output}")
        if args.parquet_output.exists():
            args.parquet_output.unlink()
            print(f"Info: Vorhandene merged.parquet entfernt: {args.parquet_output}")

    csv_inputs = build_input_paths(args.input_dir, args.input_prefix, ".csv")
    parquet_inputs = build_input_paths(args.input_dir, args.input_prefix, ".parquet")
    file_label_prefix = f"{args.input_prefix}_" if args.input_prefix else ""

    wrote_any_output = False
    merge_attempted = False
    remaining_rows_after_filter = 0

    csv_missing = missing_inputs(csv_inputs)
    if not csv_missing:
        merge_attempted = True
        merged_csv = create_merged_dataset(
            input_paths=csv_inputs,
            freq=args.freq,
            moving_average_window=args.moving_average_window,
            max_zero_run=args.max_zero_run,
            max_age_ms=args.max_age_ms,
        )
        merged_csv, removed_rows_csv, brake_col_csv = filter_standstill_and_split(
            merged_csv,
            standstill_min_minutes=args.standstill_min_minutes,
            standstill_buffer_minutes=args.standstill_buffer_minutes,
        )
        remaining_rows_after_filter += len(merged_csv)
        if not args.trips_only:
            args.csv_output.parent.mkdir(parents=True, exist_ok=True)
            merged_csv.to_csv(args.csv_output, index=False)
            wrote_any_output = True
            print(f"OK: Merged CSV gespeichert: {args.csv_output}")
            print(f"  Zeilen: {len(merged_csv)}")
            print(f"  Spalten: {len(merged_csv.columns)}")
        if brake_col_csv is not None:
            print(
                f"  Stillstandfilter: {removed_rows_csv} Zeilen entfernt "
                f"(Kriterium: abs(v_est)<0.1 oder NaN und "
                f"(abs({brake_col_csv})<0.05 oder {brake_col_csv}<-0.05), "
                f"ab {args.standstill_min_minutes} min, Puffer {args.standstill_buffer_minutes} min)."
            )
        if not args.no_trip_split:
            trip_files = write_trip_files(
                merged_csv,
                args.csv_output,
                trip_subdir_prefix=args.trip_subdir_prefix,
                file_label_prefix=file_label_prefix,
            )
            print(f"  Trip-Split CSV: {trip_files} Datei(en) erzeugt.")
            wrote_any_output = wrote_any_output or (trip_files > 0)
    else:
        print("Info: CSV-Merge uebersprungen (fehlende Eingaben).")
        for path in csv_missing:
            print(f"  - fehlt: {path}")

    parquet_missing = missing_inputs(parquet_inputs)
    if not parquet_missing:
        merge_attempted = True
        merged_parquet = create_merged_dataset(
            input_paths=parquet_inputs,
            freq=args.freq,
            moving_average_window=args.moving_average_window,
            max_zero_run=args.max_zero_run,
            max_age_ms=args.max_age_ms,
        )
        merged_parquet, removed_rows_parquet, brake_col_parquet = filter_standstill_and_split(
            merged_parquet,
            standstill_min_minutes=args.standstill_min_minutes,
            standstill_buffer_minutes=args.standstill_buffer_minutes,
        )
        remaining_rows_after_filter += len(merged_parquet)
        if not args.trips_only:
            args.parquet_output.parent.mkdir(parents=True, exist_ok=True)
            merged_parquet.to_parquet(args.parquet_output, index=False)
            wrote_any_output = True
            print(f"OK: Merged Parquet gespeichert: {args.parquet_output}")
            print(f"  Zeilen: {len(merged_parquet)}")
            print(f"  Spalten: {len(merged_parquet.columns)}")
        if brake_col_parquet is not None:
            print(
                f"  Stillstandfilter: {removed_rows_parquet} Zeilen entfernt "
                f"(Kriterium: abs(v_est)<0.1 oder NaN und "
                f"(abs({brake_col_parquet})<0.05 oder {brake_col_parquet}<-0.05), "
                f"ab {args.standstill_min_minutes} min, Puffer {args.standstill_buffer_minutes} min)."
            )
        if not args.no_trip_split:
            trip_files = write_trip_files(
                merged_parquet,
                args.parquet_output,
                trip_subdir_prefix=args.trip_subdir_prefix,
                file_label_prefix=file_label_prefix,
            )
            print(f"  Trip-Split Parquet: {trip_files} Datei(en) erzeugt.")
            wrote_any_output = wrote_any_output or (trip_files > 0)
    else:
        print("Info: Parquet-Merge uebersprungen (fehlende Eingaben).")
        for path in parquet_missing:
            print(f"  - fehlt: {path}")

    if not wrote_any_output:
        if merge_attempted and remaining_rows_after_filter == 0:
            print(
                "Info: Nach Stillstand-/Gap-Filter sind keine gueltigen Fahrdaten uebrig. "
                "Keine Trip-Dateien erzeugt (standstill-only)."
            )
            return
        raise FileNotFoundError(
            "Kein Merge erzeugt, weil weder ein vollstaendiger CSV- noch Parquet-Eingabesatz vorhanden ist."
        )

    print("  Label-Spalten: M_ATO_RTBRq_raw, M_ATO_RTBRq_deglitched, M_ATO_RTBRq_smooth")


if __name__ == "__main__":
    main()
