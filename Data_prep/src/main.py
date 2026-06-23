import argparse
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = Path(__file__).resolve().parent
DEFAULT_MERGED_OUTPUT = BASE_DIR / "output" / "merged.csv"


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
        "--merge-output",
        type=Path,
        default=DEFAULT_MERGED_OUTPUT,
        help=f"Zieldatei fuer Merge-CSV (Default: {DEFAULT_MERGED_OUTPUT}).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python_exec = sys.executable

    if not args.skip_export:
        run_step("Export", [python_exec, str(SRC_DIR / "export.py")])
    else:
        print("\n=== Export ===")
        print("Uebersprungen (--skip-export).")

    if not args.skip_merge:
        merge_cmd = [
            python_exec,
            str(SRC_DIR / "merge.py"),
            "--freq",
            args.freq,
            "--moving-average-window",
            str(args.moving_average_window),
            "--max-zero-run",
            str(args.max_zero_run),
            "--max-age-ms",
            str(args.max_age_ms),
            "--output",
            str(args.merge_output),
        ]
        run_step("Merge", merge_cmd)
    else:
        print("\n=== Merge ===")
        print("Uebersprungen (--skip-merge).")

    if not args.skip_plot:
        run_step("Plot", [python_exec, str(SRC_DIR / "plot.py")])
    else:
        print("\n=== Plot ===")
        print("Uebersprungen (--skip-plot).")

    print("\nWorkflow erfolgreich abgeschlossen.")


if __name__ == "__main__":
    main()