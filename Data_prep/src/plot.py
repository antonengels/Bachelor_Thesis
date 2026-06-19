import pandas as pd
from pathlib import Path
import plotly.graph_objects as go

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / 'output'
NID1_EXPORT = OUTPUT_DIR / 'NID_1.csv'
NID6_EXPORT = OUTPUT_DIR / 'NID_6.csv'
NID31_EXPORT = OUTPUT_DIR / 'NID_31.csv'
NID32_EXPORT = OUTPUT_DIR / 'NID_32.csv'

PLOT_NID1_PATH = OUTPUT_DIR / 'NID_1_interactive.html'
PLOT_NID6_PATH = OUTPUT_DIR / 'NID_6_interactive.html'
PLOT_NID31_PATH = OUTPUT_DIR / 'NID_31_interactive.html'
PLOT_NID32_PATH = OUTPUT_DIR / 'NID_32_interactive.html'


def prepare_time_column(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()
	df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
	df = df[df['timestamp'].notna()]
	df['datetime'] = pd.to_datetime(df['timestamp'], unit='us', utc=True, errors='coerce')
	return df[df['datetime'].notna()]


def load_nid6() -> pd.DataFrame:
	df = pd.read_csv(NID6_EXPORT)
	numeric_cols = ['v_est', 'a_est', 'v_mrsp', 'v_permitted'] + [f'grad[{i}]' for i in range(10)]

	for col in numeric_cols:
		if col in df.columns:
			df[col] = pd.to_numeric(df[col], errors='coerce')

	return prepare_time_column(df)


def load_nid31() -> pd.DataFrame:
	df = pd.read_csv(NID31_EXPORT)
	if 'value' in df.columns:
		df['value'] = pd.to_numeric(df['value'], errors='coerce')
	return prepare_time_column(df)


def load_nid1() -> pd.DataFrame:
	df = pd.read_csv(NID1_EXPORT)
	if 'D_STPDISTANCE' in df.columns:
		df['D_STPDISTANCE'] = pd.to_numeric(df['D_STPDISTANCE'], errors='coerce')
	return prepare_time_column(df)


def load_nid32() -> pd.DataFrame:
	df = pd.read_csv(NID32_EXPORT)
	if 'M_RST_TBsetVal' in df.columns:
		df['M_RST_TBsetVal'] = pd.to_numeric(df['M_RST_TBsetVal'], errors='coerce')
	if 'M_RST_SlipSlide' in df.columns:
		df['M_RST_SlipSlide'] = pd.to_numeric(df['M_RST_SlipSlide'], errors='coerce')
	return prepare_time_column(df)


def build_nid6_figure(df: pd.DataFrame) -> go.Figure:
	fig = go.Figure()

	primary_cols = ['v_est', 'v_mrsp', 'v_permitted', 'a_est']
	for col in primary_cols:
		if col in df.columns:
			fig.add_trace(
				go.Scatter(
					x=df['datetime'],
					y=df[col],
					mode='lines',
					name=col,
				)
			)

	for i in range(10):
		col = f'grad[{i}]'
		if col in df.columns:
			fig.add_trace(
				go.Scatter(
					x=df['datetime'],
					y=df[col],
					mode='lines',
					name=col,
					visible='legendonly',
				)
			)

	fig.update_layout(
		title='NID 6: Fahrzeugdynamik (interaktiv)',
		xaxis_title='Zeit (UTC)',
		yaxis_title='Wert (m/s bzw. m/s^2)',
		hovermode='x unified',
		template='plotly_white',
		legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02, 'xanchor': 'left', 'x': 0},
	)

	fig.update_xaxes(rangeslider_visible=True)
	return fig


def build_nid31_figure(df: pd.DataFrame) -> go.Figure:
	fig = go.Figure()

	fig.add_trace(
		go.Scatter(
			x=df['datetime'],
			y=df['value'],
			mode='lines',
			name='NID31 value',
			line={'width': 1.5, 'color': '#d62728'},
		)
	)

	fig.update_layout(
		title='NID 31: Hebelstellung (interaktiv)',
		xaxis_title='Zeit (UTC)',
		yaxis_title='Normierter Wert [-1, 1]',
		hovermode='x unified',
		template='plotly_white',
	)

	fig.update_xaxes(rangeslider_visible=True)
	return fig


def build_nid1_figure(df: pd.DataFrame) -> go.Figure:
	fig = go.Figure()

	fig.add_trace(
		go.Scatter(
			x=df['datetime'],
			y=df['D_STPDISTANCE'],
			mode='lines',
			name='Distanz bis Haltepunkt',
			line={'width': 1.5, 'color': '#1f77b4'},
		)
	)

	fig.update_layout(
		title='NID 1: Distanz bis zum nächsten Haltepunkt',
		xaxis_title='Zeit (UTC)',
		yaxis_title='Distanz (Meter)',
		hovermode='x unified',
		template='plotly_white',
	)

	fig.update_xaxes(rangeslider_visible=True)
	return fig


def build_nid32_figure(df: pd.DataFrame) -> go.Figure:
	fig = go.Figure()

	fig.add_trace(
		go.Scatter(
			x=df['datetime'],
			y=df['M_RST_TBsetVal'],
			mode='lines',
			name='Zugkraft-Feedback',
			line={'width': 1.5, 'color': '#ff7f0e'},
			yaxis='y1',
		)
	)

	fig.add_trace(
		go.Scatter(
			x=df['datetime'],
			y=df['M_RST_SlipSlide'],
			mode='lines',
			name='Radschlupf-Indikator',
			line={'width': 1.5, 'color': '#2ca02c'},
			yaxis='y2',
		)
	)

	fig.update_layout(
		title='NID 32: Zugkraft-Feedback und Radschlupf',
		xaxis_title='Zeit (UTC)',
		hovermode='x unified',
		template='plotly_white',
		yaxis=dict(
			title='Zugkraft-Feedback [-1.0 bis +1.0]',
			side='left'
		),
		yaxis2=dict(
			title='Radschlupf [Boolean]',
			overlaying='y',
			side='right'
		),
	)

	fig.update_xaxes(rangeslider_visible=True)
	return fig


def main() -> None:
	nid1_df = load_nid1()
	nid6_df = load_nid6()
	nid31_df = load_nid31()
	nid32_df = load_nid32()

	if nid1_df.empty:
		print('Keine Daten in NID_1.csv gefunden.')
	else:
		nid1_fig = build_nid1_figure(nid1_df)
		nid1_fig.write_html(PLOT_NID1_PATH, include_plotlyjs='cdn')
		print(f'✓ Interaktiver Plot gespeichert: {PLOT_NID1_PATH}')
		print(f'  NID_1 Punkte: {len(nid1_df)}')

	if nid6_df.empty:
		print('Keine Daten in NID_6.csv gefunden.')
		return

	if nid31_df.empty:
		print('Keine Daten in NID_31.csv gefunden.')
		return

	if nid32_df.empty:
		print('Keine Daten in NID_32.csv gefunden.')
	else:
		nid32_fig = build_nid32_figure(nid32_df)
		nid32_fig.write_html(PLOT_NID32_PATH, include_plotlyjs='cdn')
		print(f'✓ Interaktiver Plot gespeichert: {PLOT_NID32_PATH}')
		print(f'  NID_32 Punkte: {len(nid32_df)}')

	nid6_fig = build_nid6_figure(nid6_df)
	nid31_fig = build_nid31_figure(nid31_df)

	nid6_fig.write_html(PLOT_NID6_PATH, include_plotlyjs='cdn')
	nid31_fig.write_html(PLOT_NID31_PATH, include_plotlyjs='cdn')

	print(f'✓ Interaktiver Plot gespeichert: {PLOT_NID6_PATH}')
	print(f'✓ Interaktiver Plot gespeichert: {PLOT_NID31_PATH}')
	print(f'  NID_6 Punkte: {len(nid6_df)}')
	print(f'  NID_31 Punkte: {len(nid31_df)}')


if __name__ == '__main__':
	main()
