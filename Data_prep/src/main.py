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


# Flags: hier zentral steuern, was die Pipeline ausfuehrt.
CSV_EXPORT = False
PARQUET_EXPORT = True
PLOT_ALL = False
PLOT_MERGED = False


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

    parser.add_argument("--freq", default="100ms", help="Merge-Zeitraster, z.B. 100ms oder 50ms.")
    parser.add_argument(
        "--moving-average-window",
        type=int,
        default=5,
        help="Fenstergroesse fuer den Moving Average im Merge.",
    )
    parser.add_argument(
        "--max-zero-run",
        type=int,
        default=2,
        help="Maximale Laenge kurzer Null-Sequenzen fuer die Label-Korrektur.",
    )
    parser.add_argument(
        "--max-age-ms",
        type=int,
        default=1000,
        help="Maximales Alter beim asof-Merge in Millisekunden.",
    )
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
            args.freq,
            "--input-dir",
            str(args.output_dir),
            "--input-prefix",
            args.output_prefix,
            "--moving-average-window",
            str(args.moving_average_window),
            "--max-zero-run",
            str(args.max_zero_run),
            "--max-age-ms",
            str(args.max_age_ms),
            "--csv-output",
            str(merge_csv_output),
            "--parquet-output",
            str(merge_parquet_output),
        ]
        run_step("Merge", merge_cmd)
    else:
        print("\n=== Merge ===")
        print("Uebersprungen (--skip-merge).")

    if not args.skip_plot and (PLOT_ALL or PLOT_MERGED):
        plot_cmd = [python_exec, str(SRC_DIR / "plot.py")]
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