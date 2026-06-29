import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "output"


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Erzeugt interaktive Plots aus CSV- oder Parquet-Dateien.")
	parser.add_argument("--merged-only", action="store_true", help="Nur den Plot fuer merged erzeugen.")
	parser.add_argument(
		"--input-dir",
		type=Path,
		default=OUTPUT_DIR,
		help=f"Eingabeverzeichnis fuer NID- und merged-Dateien (Default: {OUTPUT_DIR}).",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=OUTPUT_DIR,
		help=f"Ausgabeverzeichnis fuer Plot-HTML-Dateien (Default: {OUTPUT_DIR}).",
	)
	parser.add_argument(
		"--file-prefix",
		type=str,
		default="",
		help="Praefix fuer Eingabe- und Plot-Dateien (z.B. '20251015').",
	)
	return parser.parse_args()


def prefixed_name(prefix: str, stem: str, extension: str) -> str:
	prefix_str = f"{prefix}_" if prefix else ""
	return f"{prefix_str}{stem}{extension}"


def resolve_input_path(input_dir: Path, prefix: str, stem: str) -> Path | None:
	parquet_path = input_dir / prefixed_name(prefix, stem, ".parquet")
	csv_path = input_dir / prefixed_name(prefix, stem, ".csv")

	if parquet_path.exists():
		return parquet_path
	if csv_path.exists():
		return csv_path
	return None


def read_table(path: Path) -> pd.DataFrame:
	if path.suffix.lower() == ".parquet":
		return pd.read_parquet(path)
	return pd.read_csv(path)


def prepare_time_column(df: pd.DataFrame) -> pd.DataFrame:
	if "timestamp" not in df.columns:
		return pd.DataFrame()
	df = df.copy()
	df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
	df = df[df["timestamp"].notna()]
	df["datetime"] = pd.to_datetime(df["timestamp"], unit="us", utc=True, errors="coerce")
	return df[df["datetime"].notna()]


def load_nid6(path: Path) -> pd.DataFrame:
	df = read_table(path)
	numeric_cols = ["v_est", "a_est", "v_mrsp", "v_permitted"] + [f"grad[{i}]" for i in range(10)]

	for col in numeric_cols:
		if col in df.columns:
			df[col] = pd.to_numeric(df[col], errors="coerce")

	return prepare_time_column(df)


def load_nid31(path: Path) -> pd.DataFrame:
	df = read_table(path)
	if "value" in df.columns:
		df["value"] = pd.to_numeric(df["value"], errors="coerce")
	return prepare_time_column(df)


def load_nid1(path: Path) -> pd.DataFrame:
	df = read_table(path)
	if "D_STPDISTANCE" in df.columns:
		df["D_STPDISTANCE"] = pd.to_numeric(df["D_STPDISTANCE"], errors="coerce")
	return prepare_time_column(df)


def load_nid32(path: Path) -> pd.DataFrame:
	df = read_table(path)
	if "M_RST_TBsetVal" in df.columns:
		df["M_RST_TBsetVal"] = pd.to_numeric(df["M_RST_TBsetVal"], errors="coerce")
	if "M_RST_SlipSlide" in df.columns:
		df["M_RST_SlipSlide"] = pd.to_numeric(df["M_RST_SlipSlide"], errors="coerce")
	return prepare_time_column(df)


def load_merged(path: Path) -> pd.DataFrame:
	df = read_table(path)

	if "datetime_utc" in df.columns:
		df["datetime"] = pd.to_datetime(df["datetime_utc"], utc=True, errors="coerce")
	elif "timestamp" in df.columns:
		df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
		df = df[df["timestamp"].notna()].copy()
		timestamp_unit = "us" if df["timestamp"].median() > 10**14 else "ms"
		df["datetime"] = pd.to_datetime(df["timestamp"], unit=timestamp_unit, utc=True, errors="coerce")
	else:
		df["datetime"] = pd.NaT

	numeric_cols = [
		"D_STPDISTANCE",
		"v_est",
		"a_est",
		"v_mrsp",
		"v_permitted",
		"M_RST_TBsetVal",
		"M_RST_SlipSlide",
		"M_ATO_RTBRq_raw",
		"M_ATO_RTBRq_deglitched",
		"M_ATO_RTBRq_smooth",
	] + [f"grad[{i}]" for i in range(10)]

	for col in numeric_cols:
		if col in df.columns:
			df[col] = pd.to_numeric(df[col], errors="coerce")

	return df[df["datetime"].notna()].copy()


def build_nid6_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    primary_cols = ["v_est", "v_mrsp", "v_permitted", "a_est"]
    for col in primary_cols:
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["datetime"],
                    y=df[col],
                    mode="lines",
                    name=col,
                )
            )

    for i in range(10):
        col = f"grad[{i}]"
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["datetime"],
                    y=df[col],
                    mode="lines",
                    name=col,
                    visible="legendonly",
                )
            )

    fig.update_layout(
        title="NID 6: Fahrzeugdynamik (interaktiv)",
        xaxis_title="Zeit (UTC)",
        yaxis_title="Wert (m/s bzw. m/s^2)",
        hovermode="x unified",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )

    fig.update_xaxes(rangeslider_visible=True)
    return fig


def build_nid31_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["value"],
            mode="lines",
            name="NID31 value",
            line={"width": 1.5, "color": "#d62728"},
        )
    )

    fig.update_layout(
        title="NID 31: Hebelstellung (interaktiv)",
        xaxis_title="Zeit (UTC)",
        yaxis_title="Normierter Wert [-1, 1]",
        hovermode="x unified",
        template="plotly_white",
    )

    fig.update_xaxes(rangeslider_visible=True)
    return fig


def build_nid1_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["D_STPDISTANCE"],
            mode="lines",
            name="Distanz bis Haltepunkt",
            line={"width": 1.5, "color": "#1f77b4"},
        )
    )

    fig.update_layout(
        title="NID 1: Distanz bis zum naechsten Haltepunkt",
        xaxis_title="Zeit (UTC)",
        yaxis_title="Distanz (Meter)",
        hovermode="x unified",
        template="plotly_white",
    )

    fig.update_xaxes(rangeslider_visible=True)
    return fig


def build_nid32_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["M_RST_TBsetVal"],
            mode="lines",
            name="Zugkraft-Feedback",
            line={"width": 1.5, "color": "#ff7f0e"},
            yaxis="y1",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["datetime"],
            y=df["M_RST_SlipSlide"],
            mode="lines",
            name="Radschlupf-Indikator",
            line={"width": 1.5, "color": "#2ca02c"},
            yaxis="y2",
        )
    )

    fig.update_layout(
        title="NID 32: Zugkraft-Feedback und Radschlupf",
        xaxis_title="Zeit (UTC)",
        hovermode="x unified",
        template="plotly_white",
        yaxis=dict(
            title="Zugkraft-Feedback [-1.0 bis +1.0]",
            side="left",
        ),
        yaxis2=dict(
            title="Radschlupf [Boolean]",
            overlaying="y",
            side="right",
        ),
    )

    fig.update_xaxes(rangeslider_visible=True)
    return fig


def build_merged_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    primary_cols = [
        "D_STPDISTANCE",
        "v_est",
        "v_mrsp",
        "v_permitted",
        "a_est",
        "M_RST_TBsetVal",
        "M_RST_SlipSlide",
        "M_ATO_RTBRq_raw",
        "M_ATO_RTBRq_deglitched",
        "M_ATO_RTBRq_smooth",
    ]

    for col in primary_cols:
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["datetime"],
                    y=df[col],
                    mode="lines",
                    name=col,
                    visible=True if col in {"D_STPDISTANCE", "v_est", "M_ATO_RTBRq_smooth"} else "legendonly",
                )
            )

    for i in range(10):
        col = f"grad[{i}]"
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["datetime"],
                    y=df[col],
                    mode="lines",
                    name=col,
                    visible="legendonly",
                )
            )

    fig.update_layout(
        title="Merged: Zeitreihen-Uebersicht (interaktiv)",
        xaxis_title="Zeit (UTC)",
        yaxis_title="Wert",
        hovermode="x unified",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )

    fig.update_xaxes(rangeslider_visible=True)
    return fig


def write_plot(fig: go.Figure, output_path: Path, label: str, count: int) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.write_html(output_path, include_plotlyjs="cdn")
	print(f"OK: Interaktiver Plot gespeichert: {output_path}")
	print(f"  {label} Punkte: {count}")


def main() -> None:
	args = parse_args()
	args.output_dir.mkdir(parents=True, exist_ok=True)

	input_paths = {
		"nid1": resolve_input_path(args.input_dir, args.file_prefix, "NID_1"),
		"nid6": resolve_input_path(args.input_dir, args.file_prefix, "NID_6"),
		"nid31": resolve_input_path(args.input_dir, args.file_prefix, "NID_31"),
		"nid32": resolve_input_path(args.input_dir, args.file_prefix, "NID_32"),
		"merged": resolve_input_path(args.input_dir, args.file_prefix, "merged"),
	}

	prefix = args.file_prefix
	plot_paths = {
		"nid1": args.output_dir / prefixed_name(prefix, "NID_1_interactive", ".html"),
		"nid6": args.output_dir / prefixed_name(prefix, "NID_6_interactive", ".html"),
		"nid31": args.output_dir / prefixed_name(prefix, "NID_31_interactive", ".html"),
		"nid32": args.output_dir / prefixed_name(prefix, "NID_32_interactive", ".html"),
		"merged": args.output_dir / prefixed_name(prefix, "merged_interactive", ".html"),
	}

	if args.merged_only:
		merged_input = input_paths["merged"]
		if merged_input is None:
			print("Datei merged.csv bzw. merged.parquet nicht gefunden, Merge-Plot wird uebersprungen.")
			return

		merged_df = load_merged(merged_input)
		if merged_df.empty:
			print(f"Keine Daten in {merged_input.name} gefunden.")
			return

		write_plot(build_merged_figure(merged_df), plot_paths["merged"], "merged", len(merged_df))
		return

	merged_input = input_paths["merged"]
	if merged_input is None:
		print("Datei merged.csv bzw. merged.parquet nicht gefunden, Merge-Plot wird uebersprungen.")
	else:
		merged_df = load_merged(merged_input)
		if merged_df.empty:
			print(f"Keine Daten in {merged_input.name} gefunden.")
		else:
			write_plot(build_merged_figure(merged_df), plot_paths["merged"], "merged", len(merged_df))

	nid1_input = input_paths["nid1"]
	if nid1_input is None:
		print("Datei NID_1.csv bzw. NID_1.parquet nicht gefunden.")
	else:
		nid1_df = load_nid1(nid1_input)
		if nid1_df.empty:
			print(f"Keine Daten in {nid1_input.name} gefunden.")
		else:
			write_plot(build_nid1_figure(nid1_df), plot_paths["nid1"], "NID_1", len(nid1_df))

	nid6_input = input_paths["nid6"]
	if nid6_input is None:
		print("Datei NID_6.csv bzw. NID_6.parquet nicht gefunden.")
	else:
		nid6_df = load_nid6(nid6_input)
		if nid6_df.empty:
			print(f"Keine Daten in {nid6_input.name} gefunden.")
		else:
			write_plot(build_nid6_figure(nid6_df), plot_paths["nid6"], "NID_6", len(nid6_df))

	nid31_input = input_paths["nid31"]
	if nid31_input is None:
		print("Datei NID_31.csv bzw. NID_31.parquet nicht gefunden.")
	else:
		nid31_df = load_nid31(nid31_input)
		if nid31_df.empty:
			print(f"Keine Daten in {nid31_input.name} gefunden.")
		else:
			write_plot(build_nid31_figure(nid31_df), plot_paths["nid31"], "NID_31", len(nid31_df))

	nid32_input = input_paths["nid32"]
	if nid32_input is None:
		print("Datei NID_32.csv bzw. NID_32.parquet nicht gefunden.")
	else:
		nid32_df = load_nid32(nid32_input)
		if nid32_df.empty:
			print(f"Keine Daten in {nid32_input.name} gefunden.")
		else:
			write_plot(build_nid32_figure(nid32_df), plot_paths["nid32"], "NID_32", len(nid32_df))


if __name__ == "__main__":
	main()
