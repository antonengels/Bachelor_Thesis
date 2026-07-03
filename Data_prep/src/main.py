import argparse
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = BASE_DIR / "input"
DEFAULT_OUTPUT_DIR = BASE_DIR / "output"
DEFAULT_INPUT_PCAP = DEFAULT_INPUT_DIR / "20260508_merged.pcapng"
DEFAULT_MERGED_CSV_OUTPUT = DEFAULT_OUTPUT_DIR / "merged.csv"
DEFAULT_MERGED_PARQUET_OUTPUT = DEFAULT_OUTPUT_DIR / "merged.parquet"

CSV_EXPORT = False
PARQUET_EXPORT = True
PLOT_ALL = True
PLOT_MERGED = False

MERGE_FREQ = "100ms"
MOVING_AVERAGE_WINDOW = 5
MAX_ZERO_RUN = 2
MAX_AGE_MS = 1000
STANDSTILL_MIN_MINUTES = 10
STANDSTILL_BUFFER_MINUTES = 5
TRIP_SPLIT_ENABLED = True
TRIPS_ONLY_OUTPUT = True
TRIP_SUBDIR_PREFIX = "trip"


def run_step(step_name: str, command: list[str]) -> None:
    print(f"\n=== {step_name} ===")
    print("Befehl:", " ".join(command))
    result = subprocess.run(command, cwd=BASE_DIR)
    if result.returncode != 0:
        raise SystemExit(f"Fehler in Schritt '{step_name}' (Exit-Code {result.returncode}).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuehrt den kompletten Datenfluss aus: Export -> Merge -> Plot."
    )
    parser.add_argument(
        "--input-pcapng",
        type=Path,
        default=DEFAULT_INPUT_PCAP,
        help=f"Pfad zur PCAPNG-Datei (Default: {DEFAULT_INPUT_PCAP})."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Ausgabeverzeichnis (Default: {DEFAULT_OUTPUT_DIR})."
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="",
        help="Präfix für die Ausgabedateien (z.B. '20251015' für '20251015_NID_1.parquet')."
    )
    parser.add_argument("--skip-export", action="store_true", help="Export-Schritt ueberspringen.")
    parser.add_argument("--skip-merge", action="store_true", help="Merge-Schritt ueberspringen.")
    parser.add_argument("--skip-plot", action="store_true", help="Plot-Schritt ueberspringen.")

    parser.add_argument(
        "--merge-csv-output",
        type=Path,
        default=None,
        help=(
            "Optionale Zieldatei fuer Merge-CSV. "
            f"Wenn nicht gesetzt, nutzt merge.py den Default ({DEFAULT_MERGED_CSV_OUTPUT})"
        ),
    )
    parser.add_argument(
        "--merge-parquet-output",
        type=Path,
        default=None,
        help=(
            "Optionale Zieldatei fuer Merge-Parquet. "
            f"Wenn nicht gesetzt, nutzt merge.py den Default ({DEFAULT_MERGED_PARQUET_OUTPUT})"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python_exec = sys.executable

    if not args.skip_export and (CSV_EXPORT or PARQUET_EXPORT):
        export_cmd = [
            python_exec,
            str(SRC_DIR / "export.py"),
            "--input-pcapng",
            str(args.input_pcapng),
            "--output-dir",
            str(args.output_dir),
            "--output-prefix",
            args.output_prefix,
        ]
        if CSV_EXPORT:
            export_cmd.append("--csv")
        if PARQUET_EXPORT:
            export_cmd.append("--parquet")
        run_step("Export", export_cmd)
    else:
        print("\n=== Export ===")
        if args.skip_export:
            print("Uebersprungen (--skip-export).")
        else:
            print("Uebersprungen (keine Export-Flags aktiv).")

    if not args.skip_merge:
        prefix_str = f"{args.output_prefix}_" if args.output_prefix else ""
        merge_csv_output = args.merge_csv_output if args.merge_csv_output is not None else args.output_dir / f"{prefix_str}merged.csv"
        merge_parquet_output = args.merge_parquet_output if args.merge_parquet_output is not None else args.output_dir / f"{prefix_str}merged.parquet"
        
        merge_cmd = [
            python_exec,
            str(SRC_DIR / "merge.py"),
            "--freq",
            MERGE_FREQ,
            "--input-dir",
            str(args.output_dir),
            "--input-prefix",
            args.output_prefix,
            "--moving-average-window",
            str(MOVING_AVERAGE_WINDOW),
            "--max-zero-run",
            str(MAX_ZERO_RUN),
            "--max-age-ms",
            str(MAX_AGE_MS),
            "--standstill-min-minutes",
            str(STANDSTILL_MIN_MINUTES),
            "--standstill-buffer-minutes",
            str(STANDSTILL_BUFFER_MINUTES),
            "--csv-output",
            str(merge_csv_output),
            "--parquet-output",
            str(merge_parquet_output),
            "--trip-subdir-prefix",
            TRIP_SUBDIR_PREFIX,
        ]
        if not TRIP_SPLIT_ENABLED:
            merge_cmd.append("--no-trip-split")
        if TRIPS_ONLY_OUTPUT:
            merge_cmd.append("--trips-only")
        run_step("Merge", merge_cmd)
    else:
        print("\n=== Merge ===")
        print("Uebersprungen (--skip-merge).")

    if not args.skip_plot and (PLOT_ALL or PLOT_MERGED):
        plot_cmd = [
            python_exec,
            str(SRC_DIR / "plot.py"),
            "--input-dir",
            str(args.output_dir),
            "--output-dir",
            str(args.output_dir),
            "--file-prefix",
            args.output_prefix,
            "--trip-subdir-prefix",
            TRIP_SUBDIR_PREFIX,
        ]
        if not PLOT_MERGED:
            plot_cmd.append("--skip-merged-plot")
        if PLOT_MERGED and not PLOT_ALL:
            plot_cmd.append("--merged-only")
        run_step("Plot", plot_cmd)
    else:
        print("\n=== Plot ===")
        if args.skip_plot:
            print("Uebersprungen (--skip-plot).")
        else:
            print("Uebersprungen (keine Plot-Flags aktiv).")

    print("\nWorkflow erfolgreich abgeschlossen.")


if __name__ == "__main__":
    main()