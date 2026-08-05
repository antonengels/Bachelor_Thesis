#!/usr/bin/env python3
"""High-performance ETCS/ATO feature extraction from PCAPNG.

The input directory is expected to contain one sub-folder per recording with
the structure ``<recording>/wireshark``. The capture folder may contain
``*.pcapng`` and ``*.pcapng.gz`` chunks (``*.pcapng.txt.gz`` is ignored).
For every recording all chunks are streamed through a single
``tshark -T fields`` pass, filtered on the ``aoecl`` application layer and
written as one compressed Parquet file per ``NID_PACKET`` group
into ``output/<recording>/features_<key>.parquet`` in an LSTM-ready schema.
Optional CSV export writes matching ``features_<key>.csv`` files. If available, packet
``era-subset-126.nid_packet_ato == 7`` is parsed into
``output/<recording>/features_topology_radius.parquet`` (and optional CSV).

After the first export pass, output completeness is checked per recording.
If files are missing, only missing outputs are re-exported. Recordings with
fully unreadable/corrupt wireshark chunks are skipped.

Using ``subprocess`` + ``tshark -T fields`` (instead of pyshark) keeps the hot
path in native code and is dramatically faster.  Rows are routed to their
feature group by ``aoecl.header.NID_PACKET`` in a single pass; the ``tshark``
field list declared in :data:`FEATURE_SPEC` is validated against the running
``tshark`` build so missing signals degrade gracefully to empty (NaN) columns.

Example:
    python src/extract_features.py --input input --output output --jobs 4

Disable CSV export:
    python src/extract_features.py --input input --output output --no-csv-export
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# User-editable defaults.
DEFAULT_INPUT_ROOT = Path("input")
DEFAULT_OUTPUT_ROOT = Path("output")

# --------------------------------------------------------------------------- #
# aoecl/frame fields (always extracted, used for routing + time key)
# --------------------------------------------------------------------------- #
HDR_NID = "aoecl.header.NID_PACKET"
HDR_FRAME_TIME_EPOCH = "frame.time_epoch"

# ERA Subset-126 packet 7 (segment profile) topology fields
TOPO_FILTER = "era-subset-126.nid_packet_ato == 7"
TOPO_FIELDS: tuple[str, ...] = (
    "frame.time_epoch",
    "era-subset-126.nid_packet_ato",
    "era-subset-126.pkt7.n_iter1",
    "era-subset-126.pkt7.l_sp",
    "era-subset-126.pkt7.q_radius_category1",
    "era-subset-126.pkt7.n_iter6",
    "era-subset-126.pkt7.d_location3",
    "era-subset-126.pkt7.q_radius_category2",
)
TOPO_REQUIRED_FIELDS: frozenset[str] = frozenset(TOPO_FIELDS)
TOPOLOGY_COLUMNS: tuple[str, ...] = (
    "t_timestamp",
    "segment_index",
    "segment_count",
    "segment_start_abs",
    "segment_end_abs",
    "segment_length",
    "segment_curve_change_count",
    "radius_interval_index",
    "radius_interval_start",
    "radius_interval_end",
    "radius_interval_start_abs",
    "radius_interval_end_abs",
    "q_radius_category",
)

DAY_RE = re.compile(r"^(\d{8})")
RECORDING_DIR_RE = re.compile(r"^(\d{8})_([A-Za-z][A-Za-z0-9-]*)_Loco(\d+)$", re.IGNORECASE)
PCAP_FILE_RE = re.compile(
    r"^(\d{8})_\d{6}_([A-Za-z][A-Za-z0-9-]*)\.loco(\d+)_\d+\.pcapng(?:\.gz)?$",
    re.IGNORECASE,
)
FEATURE_PREFIX = "features_"
RESUME_STATE_FILENAME = ".extract_features_state.json"

TRIP_SPEED_KEY = "nid6"
TRIP_SPEED_COLUMN = "V_EST"
TRIP_BRAKE_KEY = "nid31"
TRIP_BRAKE_COLUMN = "M_ATO_RTBRq"
TRIP_REQUIRED_KEYS: frozenset[str] = frozenset({TRIP_SPEED_KEY, TRIP_BRAKE_KEY})

DEFAULT_TRIP_SPLIT = True
DEFAULT_TRIP_STANDSTILL_MINUTES = 5.0
DEFAULT_TRIP_MIN_DURATION_MINUTES = 5.0
DEFAULT_TRIP_SPEED_THRESHOLD = 5.0
DEFAULT_TRIP_BRAKE_THRESHOLD = 82.0
# Split a trip whenever the recorded data has a temporal gap longer than this
# (no packets at all). 0 disables gap-based splitting. This prevents the long
# intra-trip data gaps observed in the EDA (e.g. multi-hour holes).
DEFAULT_TRIP_MAX_GAP_SECONDS = 60.0
DEFAULT_TRIP_OUTPUT_DIRNAME = "trips"
DEFAULT_DAILY_MERGE = False
DEFAULT_JOBS = 3
DEFAULT_RECORDING_JOBS = 2


# --------------------------------------------------------------------------- #
# Declarative feature specification
#   Edit here to add / remap signals. Field names are validated at runtime
#   against `tshark -G fields`; unknown fields become empty (NaN) columns.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Scalar:
    """A single scalar signal extracted from one tshark field."""

    name: str            # output column name
    field: str           # tshark field abbreviation
    as_int: bool = False  # keep as nullable integer instead of float


@dataclass(frozen=True)
class Gradient:
    """A comma-separated array field expanded to a fixed number of columns."""

    name: str            # column prefix -> name[0] .. name[width-1]
    field: str
    width: int


@dataclass(frozen=True)
class Group:
    """All signals that share the same output file (selected by NID_PACKET)."""

    key: str                       # -> features_<key>.parquet
    nids: tuple[int, ...]
    scalars: tuple[Scalar, ...] = ()
    gradients: tuple[Gradient, ...] = ()


FEATURE_SPEC: tuple[Group, ...] = (
    # NID 6 — vehicle dynamics & limits
    Group(
        key="nid6",
        nids=(6,),
        scalars=(
            Scalar("V_EST", "aoecl.userdata.nid6.V_EST"),
            Scalar("V_PERMITTED", "aoecl.userdata.nid6.V_PERMITTED"),
            Scalar("A_EST", "aoecl.userdata.nid6.A_EST"),
        ),
        gradients=(
            Gradient("grad", "aoecl.userdata.nid6.A_GRADIENT", 10),
        ),
    ),
    # NID 31 / 100 — label (brake lever request)
    Group(
        key="nid31",
        nids=(31, 100),
        scalars=(
            Scalar("M_ATO_RTBRq", "aoecl.userdata.nid31.M_ATO_RTBRq"),
        ),
    ),
    # NID 1 — distance
    Group(
        key="nid1",
        nids=(1,),
        scalars=(
            Scalar("D_STPDISTANCE", "aoecl.userdata.nid1.D_STPDISTANCE"),
        ),
    ),
    # NID 32 — vehicle feedback
    Group(
        key="nid32",
        nids=(32,),
        scalars=(
            Scalar("M_RST_TBsetVal", "aoecl.userdata.nid32.M_RST_TBsetVal"),
            Scalar("M_RST_SlipSlide", "aoecl.userdata.nid32.M_RST_SlipSlide", as_int=True),
        ),
    ),
)


# --------------------------------------------------------------------------- #
# Runtime extraction plan
# --------------------------------------------------------------------------- #
@dataclass
class GroupPlan:
    """A :class:`Group` augmented with the fields actually available."""

    spec: Group
    valid_fields: tuple[str, ...]
    field_index: dict[str, int]   # tshark field -> column index in output line


@dataclass
class ExtractionPlan:
    """Everything needed to run and parse a single tshark pass."""

    master_fields: list[str]          # ordered -e fields (NID, TS, then values)
    groups: list[GroupPlan]
    nid_to_group: dict[int, GroupPlan]
    valid_fields: set[str]
    missing_fields: set[str]
    display_filter: str
    # Topology (ERA Subset-126 packet 7) is extracted in the SAME tshark pass
    # when all required fields are available. ``topo_index`` maps each topology
    # field to its column position in ``master_fields``.
    topology_enabled: bool = False
    topo_index: dict[str, int] = field(default_factory=dict)
    missing_topology: tuple[str, ...] = ()

    def build_command(self, tshark: str, path: Path) -> list[str]:
        cmd = [
            tshark, "-r", str(path),
            "-Y", self.display_filter,
            "-T", "fields",
            "-E", "separator=\t",
            "-E", "occurrence=a",
            "-E", "aggregator=,",
        ]
        for fld in self.master_fields:
            cmd.extend(("-e", fld))
        return cmd


@dataclass(frozen=True)
class TripSplitConfig:
    """Configuration for standstill-based trip segmentation."""

    enabled: bool = DEFAULT_TRIP_SPLIT
    standstill_min_minutes: float = DEFAULT_TRIP_STANDSTILL_MINUTES
    min_trip_minutes: float = DEFAULT_TRIP_MIN_DURATION_MINUTES
    speed_threshold: float = DEFAULT_TRIP_SPEED_THRESHOLD
    brake_threshold: float = DEFAULT_TRIP_BRAKE_THRESHOLD
    max_gap_seconds: float = DEFAULT_TRIP_MAX_GAP_SECONDS
    output_dirname: str = DEFAULT_TRIP_OUTPUT_DIRNAME


@dataclass
class RecordingResult:
    """Structured result of :func:`process_recording` (replaces an 11-tuple)."""

    written_parquet: dict[str, tuple[Path, int]]
    written_csv: dict[str, tuple[Path, int]]
    aoecl_packets: int
    topology_packets: int
    aoecl_failed_files: int
    topology_failed_files: int
    aoecl_failed_paths: set[Path]
    available_keys: set[str]
    trip_written_parquet: list[tuple[int, str, Path, int]]
    trip_written_csv: list[tuple[int, str, Path, int]]
    trip_count: int


def build_plan(available: set[str]) -> ExtractionPlan:
    """Validate the declared feature fields against a tshark build."""
    master = [HDR_NID, HDR_FRAME_TIME_EPOCH]
    seen = set(master)
    valid: set[str] = set()
    missing: set[str] = set()
    staged: list[tuple[Group, tuple[str, ...]]] = []

    for group in FEATURE_SPEC:
        fields = [s.field for s in group.scalars] + [g.field for g in group.gradients]
        group_valid: list[str] = []
        for fld in fields:
            if fld in available:
                valid.add(fld)
                group_valid.append(fld)
                if fld not in seen:
                    seen.add(fld)
                    master.append(fld)
            else:
                missing.add(fld)
        staged.append((group, tuple(group_valid)))

    position = {fld: i for i, fld in enumerate(master)}
    groups: list[GroupPlan] = []
    nid_to_group: dict[int, GroupPlan] = {}
    for group, group_valid in staged:
        plan = GroupPlan(
            spec=group,
            valid_fields=group_valid,
            field_index={fld: position[fld] for fld in group_valid},
        )
        groups.append(plan)
        for nid in group.nids:
            nid_to_group[nid] = plan

    nids = sorted({nid for group in FEATURE_SPEC for nid in group.nids})
    display_filter = "aoecl.header.NID_PACKET in {" + ",".join(map(str, nids)) + "}"

    # Topology packet 7 is extracted in the same pass when all fields exist.
    missing_topology = tuple(sorted(TOPO_REQUIRED_FIELDS.difference(available)))
    topology_enabled = not missing_topology
    topo_index: dict[str, int] = {}
    if topology_enabled:
        for fld in TOPO_FIELDS:
            if fld not in seen:
                seen.add(fld)
                master.append(fld)
        combined_position = {fld: i for i, fld in enumerate(master)}
        topo_index = {fld: combined_position[fld] for fld in TOPO_FIELDS}
        display_filter = f"({display_filter}) || ({TOPO_FILTER})"

    return ExtractionPlan(
        master,
        groups,
        nid_to_group,
        valid,
        missing,
        display_filter,
        topology_enabled,
        topo_index,
        missing_topology,
    )


# --------------------------------------------------------------------------- #
# Value transforms
# --------------------------------------------------------------------------- #
def _scalar_series(raw: list[str], spec: Scalar) -> pd.Series:
    """Convert raw tshark strings into a numeric pandas Series without scaling."""
    series = pd.to_numeric(pd.Series(raw, dtype="object"), errors="coerce")
    if spec.as_int:
        return series.round().astype("Int64")
    return series.astype("float64")


def _gradient_frame(raw: list[str], spec: Gradient) -> pd.DataFrame:
    """Expand a comma-separated array field to exactly ``spec.width`` columns."""
    columns = [f"{spec.name}[{i}]" for i in range(spec.width)]
    split = pd.Series(raw, dtype="object").str.split(",", expand=True)
    # Force exactly `width` columns: pad missing with NaN, truncate the rest.
    split = split.reindex(columns=range(spec.width))
    num = split.apply(pd.to_numeric, errors="coerce")
    num.columns = columns
    return num.astype("float64")


def _empty_scalar(spec: Scalar, n: int) -> pd.Series:
    if spec.as_int:
        return pd.Series(pd.array([pd.NA] * n, dtype="Int64"))
    return pd.Series(np.full(n, np.nan), dtype="float64")


def _to_int(raw: str) -> int | None:
    token = raw.strip()
    if not token:
        return None
    try:
        return int(token, 0)
    except ValueError:
        try:
            return int(token)
        except ValueError:
            try:
                return int(float(token))
            except ValueError:
                return None


def _epoch_to_ms(raw: str) -> int | None:
    token = raw.strip()
    if not token:
        return None
    try:
        return int(round(float(token) * 1000.0))
    except ValueError:
        return None


def _parse_int_csv(raw: str) -> list[int]:
    if not raw:
        return []
    out: list[int] = []
    for part in raw.split(","):
        value = _to_int(part)
        if value is not None:
            out.append(value)
    return out


def _at_or_none(values: list[int], idx: int) -> int | None:
    return values[idx] if idx < len(values) else None


def _build_group_df(plan: GroupPlan, ts: list[str], raw: dict[str, list[str]]) -> pd.DataFrame:
    """Assemble one group's DataFrame for a single file."""
    n = len(ts)
    ts_ms = [_epoch_to_ms(value) for value in ts]
    data: dict[str, pd.Series] = {
        "t_timestamp": pd.Series(ts_ms, dtype="Int64"),
    }
    for scalar in plan.spec.scalars:
        if scalar.field in plan.field_index:
            data[scalar.name] = _scalar_series(raw[scalar.field], scalar)
        else:
            data[scalar.name] = _empty_scalar(scalar, n)

    frames = [pd.DataFrame(data)]
    for gradient in plan.spec.gradients:
        if gradient.field in plan.field_index:
            frames.append(_gradient_frame(raw[gradient.field], gradient))
        else:
            columns = [f"{gradient.name}[{i}]" for i in range(gradient.width)]
            frames.append(pd.DataFrame(np.nan, index=range(n), columns=columns))
    return pd.concat(frames, axis=1) if len(frames) > 1 else frames[0]


def _sort_by_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame sorted by timestamp (stable for equal timestamps)."""
    if df.empty or "t_timestamp" not in df.columns:
        return df.copy()
    return df.copy().sort_values("t_timestamp", kind="stable").reset_index(drop=True)


def _to_lstm_ready(key: str, df: pd.DataFrame) -> pd.DataFrame:
    """Convert one feature table to an LSTM-friendly sequential schema."""
    if df.empty:
        return _sort_by_timestamp(df)

    # NID6 contains gradient arrays as wide columns; convert to long events.
    if key == "nid6":
        grad_cols = [c for c in df.columns if c.startswith("grad[") and c.endswith("]")]
        base_cols = [c for c in df.columns if c not in grad_cols]
        if grad_cols:
            long = df.melt(
                id_vars=base_cols,
                value_vars=grad_cols,
                var_name="grad_bin",
                value_name="grad_value",
            )
            long["grad_index"] = (
                long["grad_bin"].str.extract(r"\[(\d+)\]", expand=False)
                .astype("Int64")
            )
            long = long.drop(columns=["grad_bin"])
            # Drop empty bins to reduce sequence noise for training.
            long = long[long["grad_value"].notna()].reset_index(drop=True)
            return _sort_by_timestamp(long)
    return _sort_by_timestamp(df)


# --------------------------------------------------------------------------- #
# Single-file streaming extraction
# --------------------------------------------------------------------------- #
def _stream_tshark(cmd: list[str]) -> tuple[subprocess.Popen, "deque[str]", threading.Thread]:
    """Start a tshark subprocess and stream its stdout as text.

    The tail of ``stderr`` (last lines) is captured in a background thread so a
    non-zero exit code can be reported with the actual tshark error instead of a
    bare exit code.
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1 << 20,
    )
    stderr_tail: deque[str] = deque(maxlen=20)

    def _drain_stderr() -> None:
        if proc.stderr is None:
            return
        for err_line in proc.stderr:
            stripped = err_line.rstrip("\r\n")
            if stripped:
                stderr_tail.append(stripped)

    drain_thread = threading.Thread(target=_drain_stderr, daemon=True)
    drain_thread.start()
    return proc, stderr_tail, drain_thread


def _tshark_failure_message(rc: int, stderr_tail: "deque[str]") -> str:
    detail = " | ".join(stderr_tail)
    if detail:
        return f"tshark failed with exit code {rc}: {detail}"
    return f"tshark failed with exit code {rc}"


def process_combined_file(tshark: str, path: Path, plan: ExtractionPlan):
    """Stream one pcapng once and extract BOTH aoecl features and topology.

    A single ``tshark`` pass filters on ``aoecl.header.NID_PACKET`` and
    ``era-subset-126.nid_packet_ato == 7`` at the same time; each row is routed
    by which fields are populated. This halves the tshark work compared with
    two independent passes over the same file.

    Returns ``(aoecl_groups, aoecl_packets, topology_df, topology_packets)``.
    """
    cmd = plan.build_command(tshark, path)
    ts: dict[str, list[str]] = {gp.spec.key: [] for gp in plan.groups}
    raw: dict[str, dict[str, list[str]]] = {
        gp.spec.key: {fld: [] for fld in gp.valid_fields} for gp in plan.groups
    }
    nid_to_group = plan.nid_to_group
    aoecl_packets = 0

    topo_on = plan.topology_enabled
    topo_rows: list[dict[str, int | None]] = []
    topo_packets = 0
    if topo_on:
        ti = plan.topo_index
        idx_time = ti["frame.time_epoch"]
        idx_nid = ti["era-subset-126.nid_packet_ato"]
        idx_n1 = ti["era-subset-126.pkt7.n_iter1"]
        idx_lsp = ti["era-subset-126.pkt7.l_sp"]
        idx_cat1 = ti["era-subset-126.pkt7.q_radius_category1"]
        idx_n6 = ti["era-subset-126.pkt7.n_iter6"]
        idx_loc = ti["era-subset-126.pkt7.d_location3"]
        idx_cat2 = ti["era-subset-126.pkt7.q_radius_category2"]

    proc, stderr_tail, drain_thread = _stream_tshark(cmd)
    assert proc.stdout is not None
    for line in proc.stdout:
        parts = line.rstrip("\r\n").split("\t")
        token = parts[0] if parts else ""
        if token:
            try:
                nid = int(token)
            except ValueError:
                nid = None
            if nid is not None:
                group = nid_to_group.get(nid)
                if group is not None:
                    aoecl_packets += 1
                    key = group.spec.key
                    ts[key].append(parts[1] if len(parts) > 1 else "")
                    buffers = raw[key]
                    for fld, idx in group.field_index.items():
                        buffers[fld].append(parts[idx] if idx < len(parts) else "")
        if topo_on:
            topo_nid_raw = parts[idx_nid] if len(parts) > idx_nid else ""
            if topo_nid_raw and 7 in _parse_int_csv(topo_nid_raw):
                topo_packets += 1
                topo_rows.extend(parse_segment_profile_radius_rows(
                    frame_time_epoch_raw=parts[idx_time] if len(parts) > idx_time else "",
                    n_segments_raw=parts[idx_n1] if len(parts) > idx_n1 else "",
                    segment_lengths_raw=parts[idx_lsp] if len(parts) > idx_lsp else "",
                    segment_start_radius_raw=parts[idx_cat1] if len(parts) > idx_cat1 else "",
                    curve_change_counts_raw=parts[idx_n6] if len(parts) > idx_n6 else "",
                    curve_change_locations_raw=parts[idx_loc] if len(parts) > idx_loc else "",
                    curve_change_radius_raw=parts[idx_cat2] if len(parts) > idx_cat2 else "",
                ))
    rc = proc.wait()
    drain_thread.join(timeout=5.0)
    if rc != 0:
        raise RuntimeError(_tshark_failure_message(rc, stderr_tail))

    out: dict[str, pd.DataFrame] = {}
    for group in plan.groups:
        key = group.spec.key
        if ts[key]:
            out[key] = _build_group_df(group, ts[key], raw[key])
    return out, aoecl_packets, _build_topology_df(topo_rows), topo_packets


# --------------------------------------------------------------------------- #
# Topology extraction (ERA Subset-126 packet 7)
# --------------------------------------------------------------------------- #
def parse_segment_profile_radius_rows(
    frame_time_epoch_raw: str,
    n_segments_raw: str,
    segment_lengths_raw: str,
    segment_start_radius_raw: str,
    curve_change_counts_raw: str,
    curve_change_locations_raw: str,
    curve_change_radius_raw: str,
) -> list[dict[str, int | None]]:
    """Parse packet-7 segment profile data into flat radius intervals.

    Packet-7 structure:
    - outer iteration (segments): ``L_SP`` + initial ``Q_Radius_Category``
    - inner iteration (curve changes): ``D_Location`` + new
      ``Q_Radius_Category``

    Output structure:
    - one row per radius interval (piecewise-constant category)
    - absolute segment offsets to keep multi-segment packets easy to consume
      for ML pipelines
    """
    t_timestamp = _epoch_to_ms(frame_time_epoch_raw)

    segment_lengths = _parse_int_csv(segment_lengths_raw)
    segment_start_categories = _parse_int_csv(segment_start_radius_raw)
    curve_change_counts = _parse_int_csv(curve_change_counts_raw)
    curve_change_locations = _parse_int_csv(curve_change_locations_raw)
    curve_change_categories = _parse_int_csv(curve_change_radius_raw)
    n_segments_hint_values = _parse_int_csv(n_segments_raw)
    n_segments_hint = n_segments_hint_values[0] if n_segments_hint_values else None

    n_segments = max(
        len(segment_lengths),
        len(segment_start_categories),
        len(curve_change_counts),
        n_segments_hint or 0,
    )
    if n_segments == 0:
        return []

    rows: list[dict[str, int | None]] = []
    curve_offset = 0
    segment_start_abs = 0
    for segment_idx in range(n_segments):
        l_sp = _at_or_none(segment_lengths, segment_idx)
        if l_sp is not None and l_sp < 0:
            l_sp = 0
        start_category = _at_or_none(segment_start_categories, segment_idx)
        raw_curve_changes = _at_or_none(curve_change_counts, segment_idx)
        curve_change_count = raw_curve_changes if raw_curve_changes is not None else 0
        if curve_change_count < 0:
            curve_change_count = 0

        raw_pairs: list[tuple[int | None, int | None]] = []
        for _ in range(curve_change_count):
            d_location = _at_or_none(curve_change_locations, curve_offset)
            q_radius_category = _at_or_none(curve_change_categories, curve_offset)
            raw_pairs.append((d_location, q_radius_category))
            curve_offset += 1

        points: list[tuple[int, int | None]] = [(0, start_category)]
        last_location = 0
        for d_location, q_radius_category in raw_pairs:
            if d_location is None or q_radius_category is None:
                continue
            if d_location <= last_location:
                continue
            if l_sp is not None and d_location >= l_sp:
                continue
            points.append((d_location, q_radius_category))
            last_location = d_location

        segment_end_abs = segment_start_abs + l_sp if l_sp is not None else None
        for interval_idx, (interval_start, q_radius_category) in enumerate(points):
            if interval_idx + 1 < len(points):
                interval_end = points[interval_idx + 1][0]
            else:
                interval_end = l_sp

            interval_start_abs = segment_start_abs + interval_start
            interval_end_abs = (
                segment_start_abs + interval_end if interval_end is not None else segment_end_abs
            )
            rows.append({
                "t_timestamp": t_timestamp,
                "segment_index": segment_idx,
                "segment_count": n_segments,
                "segment_start_abs": segment_start_abs,
                "segment_end_abs": segment_end_abs,
                "segment_length": l_sp,
                "segment_curve_change_count": curve_change_count,
                "radius_interval_index": interval_idx,
                "radius_interval_start": interval_start,
                "radius_interval_end": interval_end,
                "radius_interval_start_abs": interval_start_abs,
                "radius_interval_end_abs": interval_end_abs,
                "q_radius_category": q_radius_category,
            })

        if l_sp is not None:
            segment_start_abs += l_sp
    return rows


def _build_topology_df(rows: list[dict[str, int | None]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=list(TOPOLOGY_COLUMNS))
    df = pd.DataFrame(rows, columns=list(TOPOLOGY_COLUMNS))
    for col in TOPOLOGY_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


# --------------------------------------------------------------------------- #
# Recording driver
# --------------------------------------------------------------------------- #
def _set_progress_postfix(bar, label: str, path_name: str) -> None:
    """Show an ETA derived from processed bytes and elapsed wall time."""
    if bar is None:
        return

    elapsed = float(bar.format_dict.get("elapsed", 0.0) or 0.0)
    processed = float(bar.n)
    total = float(bar.total or 0.0)

    eta_text = "--:--"
    if elapsed > 0.0 and processed > 0.0 and total > processed:
        bytes_per_second = processed / elapsed
        if bytes_per_second > 0.0:
            remaining_seconds = int(round((total - processed) / bytes_per_second))
            eta_text = tqdm.format_interval(max(0, remaining_seconds))
    elif total > 0.0 and processed >= total:
        eta_text = "00:00"

    bar.set_postfix_str(f"{label} · {path_name} · ETA {eta_text}", refresh=False)


def _process_files(files, plan, tshark, jobs, bar, label):
    """Process all chunks of one recording in a single combined tshark pass.

    Returns ``(per_group, aoecl_packets, topology_df, topology_packets,
    failed_files, failed_paths)``.
    """
    results: list[tuple[dict, pd.DataFrame] | None] = [None] * len(files)
    aoecl_packets = 0
    topology_packets = 0
    failed_files = 0
    failed_paths: set[Path] = set()

    if jobs <= 1:
        for i, path in enumerate(files):
            _set_progress_postfix(bar, label, path.name)
            try:
                dfs, n_aoecl, topo_df, n_topo = process_combined_file(tshark, path, plan)
                results[i] = (dfs, topo_df)
                aoecl_packets += n_aoecl
                topology_packets += n_topo
            except Exception as exc:  # noqa: BLE001 - keep going on a bad chunk
                tqdm.write(
                    f"WARNING: unreadable/corrupt chunk skipped: {path.name} ({exc})"
                )
                failed_files += 1
                failed_paths.add(path)
            if bar is not None:
                bar.update(path.stat().st_size)
            _set_progress_postfix(bar, label, path.name)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {
                pool.submit(process_combined_file, tshark, path, plan): i
                for i, path in enumerate(files)
            }
            for future in as_completed(futures):
                i = futures[future]
                path = files[i]
                try:
                    dfs, n_aoecl, topo_df, n_topo = future.result()
                    results[i] = (dfs, topo_df)
                    aoecl_packets += n_aoecl
                    topology_packets += n_topo
                except Exception as exc:  # noqa: BLE001
                    tqdm.write(
                        f"WARNING: unreadable/corrupt chunk skipped: {path.name} ({exc})"
                    )
                    failed_files += 1
                    failed_paths.add(path)
                if bar is not None:
                    bar.update(path.stat().st_size)
                _set_progress_postfix(bar, label, path.name)

    per_group: dict[str, list[pd.DataFrame]] = defaultdict(list)
    topo_frames: list[pd.DataFrame] = []
    for res in results:
        if res is None:
            continue
        dfs, topo_df = res
        if dfs:
            for key, df in dfs.items():
                per_group[key].append(df)
        if topo_df is not None and not topo_df.empty:
            topo_frames.append(topo_df)
    if topo_frames:
        topology_df = pd.concat(topo_frames, ignore_index=True)
    else:
        topology_df = pd.DataFrame(columns=list(TOPOLOGY_COLUMNS))
    return (
        per_group,
        aoecl_packets,
        topology_df,
        topology_packets,
        failed_files,
        failed_paths,
    )


def process_recording(
    name,
    files,
    plan,
    tshark,
    out_root,
    jobs,
    bar,
    compression,
    topology_enabled,
    csv_export,
    trip_split_config: TripSplitConfig,
    missing_only: dict[str, set[str]] | None = None,
    exclude_files: set[Path] | None = None,
):
    """Extract one recording and write per-group output files."""
    excluded = exclude_files or set()
    process_files = [path for path in files if path not in excluded]

    (per_group,
     packets,
     topology_df,
     topology_packets,
     aoecl_failed_files,
     aoecl_failed_paths) = _process_files(
        process_files,
        plan,
        tshark,
        jobs,
        bar,
        name,
    )
    out_dir = out_root / name
    raw_dir = out_dir / "raw"
    written_parquet: dict[str, tuple[Path, int]] = {}
    written_csv: dict[str, tuple[Path, int]] = {}
    available_keys: set[str] = set()
    per_key_lstm_frames: dict[str, pd.DataFrame] = {}
    per_key_raw_frames: dict[str, pd.DataFrame] = {}

    for group in plan.groups:
        key = group.spec.key
        if missing_only is not None and key not in missing_only:
            continue

        formats_to_write = {"parquet"}
        if csv_export:
            formats_to_write.add("csv")
        if missing_only is not None:
            formats_to_write &= missing_only.get(key, set())
        if not formats_to_write:
            continue

        frames = per_group.get(key)
        if not frames:
            continue
        full = pd.concat(frames, ignore_index=True)
        if full.empty:
            continue

        lstm_df = _to_lstm_ready(key, full)
        if lstm_df.empty:
            continue
        per_key_raw_frames[key] = full
        per_key_lstm_frames[key] = lstm_df
        available_keys.add(key)
        raw_dir.mkdir(parents=True, exist_ok=True)
        if "parquet" in formats_to_write:
            parquet_path = raw_dir / f"features_{key}.parquet"
            lstm_df.to_parquet(parquet_path, index=False, compression=compression)
            written_parquet[key] = (parquet_path, len(lstm_df))
        if "csv" in formats_to_write:
            csv_path = raw_dir / f"features_{key}.csv"
            lstm_df.to_csv(csv_path, index=False)
            written_csv[key] = (csv_path, len(lstm_df))

    topology_failed_files = aoecl_failed_files if topology_enabled else 0
    if topology_enabled:
        topology_formats_to_write = {"parquet"}
        if csv_export:
            topology_formats_to_write.add("csv")
        if missing_only is not None:
            topology_formats_to_write &= missing_only.get("topology_radius", set())

        if not topology_df.empty and topology_formats_to_write:
            lstm_topology_df = _to_lstm_ready("topology_radius", topology_df)
            if not lstm_topology_df.empty:
                per_key_raw_frames["topology_radius"] = topology_df
                per_key_lstm_frames["topology_radius"] = lstm_topology_df
                available_keys.add("topology_radius")
                raw_dir.mkdir(parents=True, exist_ok=True)
                if "parquet" in topology_formats_to_write:
                    parquet_path = raw_dir / "features_topology_radius.parquet"
                    lstm_topology_df.to_parquet(parquet_path, index=False, compression=compression)
                    written_parquet["topology_radius"] = (parquet_path, len(lstm_topology_df))
                if "csv" in topology_formats_to_write:
                    csv_path = raw_dir / "features_topology_radius.csv"
                    lstm_topology_df.to_csv(csv_path, index=False)
                    written_csv["topology_radius"] = (csv_path, len(lstm_topology_df))

    trip_written_parquet: list[tuple[int, str, Path, int]] = []
    trip_written_csv: list[tuple[int, str, Path, int]] = []
    trip_count = 0
    if trip_split_config.enabled and missing_only is None and per_key_lstm_frames:
        (
            trip_written_parquet,
            trip_written_csv,
            trip_count,
        ) = _write_trip_outputs(
            out_dir=out_dir,
            per_key_lstm_frames=per_key_lstm_frames,
            per_key_raw_frames=per_key_raw_frames,
            compression=compression,
            csv_export=csv_export,
            cfg=trip_split_config,
        )

    return RecordingResult(
        written_parquet=written_parquet,
        written_csv=written_csv,
        aoecl_packets=packets,
        topology_packets=topology_packets,
        aoecl_failed_files=aoecl_failed_files,
        topology_failed_files=topology_failed_files,
        aoecl_failed_paths=aoecl_failed_paths,
        available_keys=available_keys,
        trip_written_parquet=trip_written_parquet,
        trip_written_csv=trip_written_csv,
        trip_count=trip_count,
    )


# --------------------------------------------------------------------------- #
# Discovery & environment helpers
# --------------------------------------------------------------------------- #
def find_recordings(input_path: Path):
    """Return ``[(recording_name, [pcapng_files])]`` for the given input path.

    Supported layouts:
    - ``input/<recording>/wireshark/*.pcapng[.gz]``
    - ``input/<recording>/*.pcapng[.gz]`` (directly in recording folder)
    """
    if input_path.is_file():
        if not _is_capture_file(input_path):
            sys.exit("Input file must be .pcapng or .pcapng.gz (excluding .pcapng.txt.gz).")
        return [(_canonical_recording_name(input_path), [input_path])]

    recordings = []
    for sub in sorted(p for p in input_path.iterdir() if p.is_dir()):
        wireshark_dir = sub / "wireshark"
        if wireshark_dir.is_dir():
            files = sorted(
                p for p in wireshark_dir.rglob("*")
                if p.is_file() and _is_capture_file(p)
            )
        else:
            # Fallback for recordings that keep capture chunks directly in the
            # day/recording folder (no nested "wireshark" directory).
            files = sorted(
                p for p in sub.rglob("*")
                if p.is_file() and _is_capture_file(p)
            )
        if files:
            recordings.append((_canonical_recording_name(sub), files))
    return recordings


def _is_capture_file(path: Path) -> bool:
    name = path.name.lower()
    if name.endswith(".pcapng.txt.gz"):
        return False
    return name.endswith(".pcapng") or name.endswith(".pcapng.gz")


def _canonical_recording_name(path: Path) -> str:
    """Infer stable output folder names like ``YYYYMMDD_Valburg_Loco1``."""

    def _build(day: str, place: str, loco: str) -> str:
        return f"{day}_{place[0].upper() + place[1:]}_Loco{int(loco)}"

    # Prefer explicit recording directories in the path hierarchy.
    for candidate in [path, *path.parents]:
        name = candidate.name
        match = RECORDING_DIR_RE.match(name)
        if match:
            return _build(match.group(1), match.group(2), match.group(3))

    # Fallback: parse known pcap naming convention.
    match = PCAP_FILE_RE.match(path.name)
    if match:
        return _build(match.group(1), match.group(2), match.group(3))

    # Final fallback keeps old behavior for unknown naming patterns.
    return path.stem if path.is_file() else path.name


def _recording_day(name: str) -> str | None:
    match = DAY_RE.match(name)
    return match.group(1) if match else None


def _recording_day_loco(name: str) -> tuple[str, str] | None:
    """Return (day, loco) for canonical recording names.

    This keeps same-day recordings from different locomotives separated during
    optional daily merge.
    """
    match = RECORDING_DIR_RE.match(name)
    if not match:
        return None
    return match.group(1), f"Loco{int(match.group(3))}"


def _collect_day_key_frames(output_root: Path):
    day_key_frames: dict[str, dict[str, list[pd.DataFrame]]] = {}

    for rec_dir in sorted(p for p in output_root.iterdir() if p.is_dir()):
        day_loco = _recording_day_loco(rec_dir.name)
        if day_loco is None:
            continue
        day_loco_key = f"{day_loco[0]}_{day_loco[1]}"

        raw_dir = rec_dir / "raw"
        files = sorted(raw_dir.glob("features_*.parquet")) if raw_dir.exists() else []
        if not files:
            continue

        by_key = day_key_frames.setdefault(day_loco_key, {})
        for path in files:
            key = path.stem.removeprefix(FEATURE_PREFIX)
            df = pd.read_parquet(path)
            if "t_timestamp" not in df.columns:
                continue
            frame = df.copy()
            frame["t_timestamp"] = pd.to_numeric(
                frame["t_timestamp"], errors="coerce"
            ).astype("Int64")
            by_key.setdefault(key, []).append(frame)
    return day_key_frames


def _collect_day_recording_dirs(output_root: Path):
    day_dirs: dict[str, list[Path]] = {}
    for rec_dir in sorted(p for p in output_root.iterdir() if p.is_dir()):
        day_loco = _recording_day_loco(rec_dir.name)
        if day_loco is None:
            continue
        day_loco_key = f"{day_loco[0]}_{day_loco[1]}"
        day_dirs.setdefault(day_loco_key, []).append(rec_dir)
    return day_dirs


def _prepare_trip_signal(df: pd.DataFrame, value_col: str) -> pd.Series:
    """Return one numeric value per timestamp for standstill detection."""
    if df.empty or "t_timestamp" not in df.columns or value_col not in df.columns:
        return pd.Series(dtype="float64")

    frame = df.loc[:, ["t_timestamp", value_col]].copy()
    frame["t_timestamp"] = pd.to_numeric(frame["t_timestamp"], errors="coerce")
    frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
    frame = frame.dropna(subset=["t_timestamp", value_col])
    if frame.empty:
        return pd.Series(dtype="float64")

    frame["t_timestamp"] = frame["t_timestamp"].round().astype("int64")
    frame = frame.sort_values("t_timestamp", kind="stable")
    dedup = frame.groupby("t_timestamp", as_index=False)[value_col].last()
    return dedup.set_index("t_timestamp")[value_col].astype("float64")


def _find_standstill_ranges_ms(
    speed_signal: pd.Series,
    brake_signal: pd.Series,
    cfg: TripSplitConfig,
) -> list[tuple[int, int]]:
    """Return standstill intervals [start_ms, end_ms) based on both signals."""
    if speed_signal.empty or brake_signal.empty:
        return []

    timeline = speed_signal.index.union(brake_signal.index)
    if timeline.empty:
        return []

    timeline = pd.Index(np.asarray(timeline, dtype="int64")).sort_values()
    signal = pd.DataFrame(index=timeline)
    signal["speed"] = speed_signal.reindex(timeline).ffill()
    signal["brake"] = brake_signal.reindex(timeline).ffill()
    signal = signal.dropna(subset=["speed", "brake"])
    if signal.empty:
        return []

    timestamps = signal.index.to_numpy(dtype=np.int64)
    speed_values = signal["speed"].to_numpy(dtype=np.float64, copy=False)
    brake_values = signal["brake"].to_numpy(dtype=np.float64, copy=False)
    standstill_mask = (
        (speed_values < cfg.speed_threshold)
        & (brake_values < cfg.brake_threshold)
    )
    if not standstill_mask.any():
        return []

    min_duration_ms = max(0, int(round(cfg.standstill_min_minutes * 60_000)))
    ranges: list[tuple[int, int]] = []
    idx = 0
    size = len(timestamps)
    while idx < size:
        if not standstill_mask[idx]:
            idx += 1
            continue

        start_idx = idx
        while idx + 1 < size and standstill_mask[idx + 1]:
            idx += 1
        end_idx = idx

        end_exclusive = (
            int(timestamps[end_idx + 1])
            if end_idx + 1 < size
            else int(timestamps[end_idx])
        )
        start_ts = int(timestamps[start_idx])
        if end_exclusive - start_ts >= min_duration_ms:
            ranges.append((start_ts, end_exclusive))
        idx += 1
    return ranges


def _global_time_bounds(per_key_frames: dict[str, pd.DataFrame]) -> tuple[int | None, int | None]:
    """Return min/max timestamps over all provided frames."""
    min_ts: int | None = None
    max_ts: int | None = None

    for df in per_key_frames.values():
        if df.empty or "t_timestamp" not in df.columns:
            continue
        ts = pd.to_numeric(df["t_timestamp"], errors="coerce").dropna()
        if ts.empty:
            continue

        local_min = int(ts.min())
        local_max = int(ts.max())
        min_ts = local_min if min_ts is None else min(min_ts, local_min)
        max_ts = local_max if max_ts is None else max(max_ts, local_max)

    return min_ts, max_ts


def _find_data_gap_ranges_ms(
    per_key_frames: dict[str, pd.DataFrame],
    cfg: TripSplitConfig,
) -> list[tuple[int, int]]:
    """Return intervals [gap_start, gap_end) where the recorded data has a hole.

    A gap is any stretch longer than ``cfg.max_gap_seconds`` without a single
    sample in one of the trip-required signals. Splitting on these prevents the
    long intra-trip data holes seen in the EDA (a stopped-and-not-transmitting
    train is not detected by standstill logic because there are no samples).
    """
    max_gap_ms = int(round(cfg.max_gap_seconds * 1000))
    if max_gap_ms <= 0:
        return []

    gaps: list[tuple[int, int]] = []
    for key in TRIP_REQUIRED_KEYS:
        df = per_key_frames.get(key)
        if df is None or df.empty or "t_timestamp" not in df.columns:
            continue
        ts = pd.to_numeric(df["t_timestamp"], errors="coerce").dropna()
        if ts.empty:
            continue
        arr = np.unique(ts.round().astype("int64").to_numpy())
        if arr.size < 2:
            continue
        diffs = np.diff(arr)
        for i in np.nonzero(diffs > max_gap_ms)[0]:
            gap_start = int(arr[i]) + 1        # first ms after the last sample
            gap_end = int(arr[i + 1])          # next sample -> new trip starts here
            if gap_end > gap_start:
                gaps.append((gap_start, gap_end))
    return gaps


def _build_trip_ranges(
    per_key_raw_frames: dict[str, pd.DataFrame],
    cfg: TripSplitConfig,
) -> list[tuple[int, int]]:
    """Build trip intervals [start_ms, end_ms) by removing long standstill blocks
    and long data gaps (holes without any samples)."""
    min_ts, max_ts = _global_time_bounds(per_key_raw_frames)
    if min_ts is None or max_ts is None:
        return []

    speed_signal = _prepare_trip_signal(
        per_key_raw_frames.get(TRIP_SPEED_KEY, pd.DataFrame()),
        TRIP_SPEED_COLUMN,
    )
    brake_signal = _prepare_trip_signal(
        per_key_raw_frames.get(TRIP_BRAKE_KEY, pd.DataFrame()),
        TRIP_BRAKE_COLUMN,
    )
    standstill_ranges = _find_standstill_ranges_ms(speed_signal, brake_signal, cfg)
    gap_ranges = _find_data_gap_ranges_ms(per_key_raw_frames, cfg)
    break_ranges = standstill_ranges + gap_ranges
    if not break_ranges:
        return [(min_ts, max_ts + 1)]

    clamped: list[tuple[int, int]] = []
    for start_ts, end_ts in break_ranges:
        start = max(start_ts, min_ts)
        end = min(end_ts, max_ts + 1)
        if end > start:
            clamped.append((start, end))
    if not clamped:
        return [(min_ts, max_ts + 1)]

    clamped.sort()
    merged_breaks: list[tuple[int, int]] = [clamped[0]]
    for start, end in clamped[1:]:
        prev_start, prev_end = merged_breaks[-1]
        if start <= prev_end:
            merged_breaks[-1] = (prev_start, max(prev_end, end))
        else:
            merged_breaks.append((start, end))

    trips: list[tuple[int, int]] = []
    cursor = min_ts
    for start, end in merged_breaks:
        if start > cursor:
            trips.append((cursor, start))
        cursor = max(cursor, end)
    if cursor <= max_ts:
        trips.append((cursor, max_ts + 1))

    trips = [(start, end) for start, end in trips if end > start]
    return trips or [(min_ts, max_ts + 1)]


def _filter_trip_ranges_by_min_duration(
    trip_ranges: list[tuple[int, int]],
    cfg: TripSplitConfig,
) -> list[tuple[int, int]]:
    """Drop trip intervals shorter than ``cfg.min_trip_minutes``."""
    min_trip_ms = max(0, int(round(cfg.min_trip_minutes * 60_000)))
    if min_trip_ms <= 0:
        return trip_ranges
    return [
        (start_ts, end_ts)
        for start_ts, end_ts in trip_ranges
        if (end_ts - start_ts) >= min_trip_ms
    ]


def _slice_trip_frame(df: pd.DataFrame, start_ts: int, end_ts: int) -> pd.DataFrame:
    """Slice one feature frame to a single trip interval [start_ts, end_ts)."""
    if df.empty or "t_timestamp" not in df.columns:
        return pd.DataFrame(columns=df.columns)

    ts = pd.to_numeric(df["t_timestamp"], errors="coerce")
    mask = ts.ge(start_ts) & ts.lt(end_ts)
    if not mask.any():
        return pd.DataFrame(columns=df.columns)
    return df.loc[mask].copy().reset_index(drop=True)


def _trip_has_required_keys(per_key_trip_frames: dict[str, pd.DataFrame]) -> bool:
    """Trip is valid only if all required keys are present and non-empty."""
    for key in TRIP_REQUIRED_KEYS:
        df = per_key_trip_frames.get(key)
        if df is None or df.empty:
            return False
    return True


def _write_trip_outputs(
    out_dir: Path,
    per_key_lstm_frames: dict[str, pd.DataFrame],
    per_key_raw_frames: dict[str, pd.DataFrame],
    compression: str | None,
    csv_export: bool,
    cfg: TripSplitConfig,
) -> tuple[list[tuple[int, str, Path, int]], list[tuple[int, str, Path, int]], int]:
    """Write trip folders like trip1, trip2 with per-feature outputs."""
    trip_ranges = _build_trip_ranges(per_key_raw_frames, cfg)
    trip_ranges = _filter_trip_ranges_by_min_duration(trip_ranges, cfg)
    if not trip_ranges:
        return [], [], 0

    trips_root = out_dir / cfg.output_dirname
    if trips_root.exists():
        shutil.rmtree(trips_root)

    written_parquet: list[tuple[int, str, Path, int]] = []
    written_csv: list[tuple[int, str, Path, int]] = []
    trip_idx = 0

    for start_ts, end_ts in trip_ranges:
        per_key_trip_frames: dict[str, pd.DataFrame] = {}
        for key, frame in per_key_lstm_frames.items():
            trip_df = _slice_trip_frame(frame, start_ts, end_ts)
            if not trip_df.empty:
                per_key_trip_frames[key] = trip_df
        if not per_key_trip_frames:
            continue
        if not _trip_has_required_keys(per_key_trip_frames):
            continue

        trip_idx += 1
        trip_dir = trips_root / f"trip{trip_idx}"
        trip_dir.mkdir(parents=True, exist_ok=True)

        for key, trip_df in sorted(per_key_trip_frames.items()):
            parquet_path = trip_dir / f"features_{key}.parquet"
            trip_df.to_parquet(parquet_path, index=False, compression=compression)
            written_parquet.append((trip_idx, key, parquet_path, len(trip_df)))

            if csv_export:
                csv_path = trip_dir / f"features_{key}.csv"
                trip_df.to_csv(csv_path, index=False)
                written_csv.append((trip_idx, key, csv_path, len(trip_df)))

    return written_parquet, written_csv, trip_idx


def _collect_recording_feature_frames_from_parquet(output_root: Path):
    """Load per-recording feature frames from ``output/<recording>/raw``."""
    recordings: list[tuple[str, dict[str, pd.DataFrame]]] = []
    for rec_dir in sorted(p for p in output_root.iterdir() if p.is_dir()):
        raw_dir = rec_dir / "raw"
        if not raw_dir.is_dir():
            continue

        per_key_frames: dict[str, pd.DataFrame] = {}
        for path in sorted(raw_dir.glob("features_*.parquet")):
            key = path.stem.removeprefix(FEATURE_PREFIX)
            try:
                df = pd.read_parquet(path)
            except Exception as exc:  # noqa: BLE001 - continue with remaining files
                tqdm.write(f"WARNING: could not read {path}: {exc}")
                continue
            if "t_timestamp" not in df.columns:
                continue
            per_key_frames[key] = _sort_by_timestamp(df)

        if per_key_frames:
            recordings.append((rec_dir.name, per_key_frames))
    return recordings


def rebuild_global_trips_from_parquet(
    output_root: Path,
    compression: str | None,
    csv_export: bool,
    cfg: TripSplitConfig,
):
    """Build a day-independent global trip list from existing Parquet features.

    Reads ``output/<recording>/raw/features_*.parquet`` and writes
    ``output/trips/trip1..tripN`` without re-running tshark/pcap extraction.
    """
    recordings = _collect_recording_feature_frames_from_parquet(output_root)
    if not recordings:
        return [], 0, 0

    candidate_trips: list[tuple[int, int, str, dict[str, pd.DataFrame]]] = []
    dropped_short = 0

    for recording_name, per_key_lstm_frames in recordings:
        trip_ranges = _build_trip_ranges(per_key_lstm_frames, cfg)
        filtered_ranges = _filter_trip_ranges_by_min_duration(trip_ranges, cfg)
        dropped_short += max(0, len(trip_ranges) - len(filtered_ranges))

        for start_ts, end_ts in filtered_ranges:
            per_key_trip_frames: dict[str, pd.DataFrame] = {}
            for key, frame in per_key_lstm_frames.items():
                trip_df = _slice_trip_frame(frame, start_ts, end_ts)
                if not trip_df.empty:
                    per_key_trip_frames[key] = trip_df
            if not per_key_trip_frames:
                continue
            if not _trip_has_required_keys(per_key_trip_frames):
                continue
            candidate_trips.append((start_ts, end_ts, recording_name, per_key_trip_frames))

    if not candidate_trips:
        return [], dropped_short, len(recordings)

    candidate_trips.sort(key=lambda item: (item[0], item[1], item[2]))

    trips_root = output_root / cfg.output_dirname
    if trips_root.exists():
        shutil.rmtree(trips_root)
    trips_root.mkdir(parents=True, exist_ok=True)

    written: list[tuple[int, str, Path, int]] = []
    manifest_rows: list[dict[str, int | str]] = []

    for trip_idx, (start_ts, end_ts, recording_name, per_key_trip_frames) in enumerate(
        candidate_trips,
        start=1,
    ):
        trip_dir = trips_root / f"trip{trip_idx}"
        trip_dir.mkdir(parents=True, exist_ok=True)
        duration_ms = max(0, end_ts - start_ts)
        manifest_rows.append(
            {
                "trip_id": trip_idx,
                "source_recording": recording_name,
                "start_t_timestamp": int(start_ts),
                "end_t_timestamp_exclusive": int(end_ts),
                "duration_ms": int(duration_ms),
            }
        )

        for key, trip_df in sorted(per_key_trip_frames.items()):
            parquet_path = trip_dir / f"features_{key}.parquet"
            trip_df.to_parquet(parquet_path, index=False, compression=compression)
            written.append((trip_idx, key, parquet_path, len(trip_df)))

            if csv_export:
                csv_path = trip_dir / f"features_{key}.csv"
                trip_df.to_csv(csv_path, index=False)

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(trips_root / "trip_manifest.csv", index=False)

    return written, dropped_short, len(recordings)


def _expected_recording_outputs(
    recording_name: str,
    output_root: Path,
    csv_export: bool,
    topology_enabled: bool,
    available_keys: set[str] | None = None,
) -> dict[str, dict[str, Path]]:
    raw_dir = output_root / recording_name / "raw"
    expected: dict[str, dict[str, Path]] = {}
    keys_filter = available_keys if available_keys is not None else set()

    for group in FEATURE_SPEC:
        key = group.key
        if available_keys is not None and key not in keys_filter:
            continue
        entry = {"parquet": raw_dir / f"features_{key}.parquet"}
        if csv_export:
            entry["csv"] = raw_dir / f"features_{key}.csv"
        expected[key] = entry

    if topology_enabled:
        if available_keys is not None and "topology_radius" not in keys_filter:
            return expected
        topo = {"parquet": raw_dir / "features_topology_radius.parquet"}
        if csv_export:
            topo["csv"] = raw_dir / "features_topology_radius.csv"
        expected["topology_radius"] = topo
    return expected


def _missing_recording_outputs(
    recording_name: str,
    output_root: Path,
    csv_export: bool,
    topology_enabled: bool,
    available_keys: set[str] | None = None,
) -> dict[str, set[str]]:
    expected = _expected_recording_outputs(
        recording_name,
        output_root,
        csv_export,
        topology_enabled,
        available_keys,
    )
    missing: dict[str, set[str]] = {}
    for key, paths in expected.items():
        miss = {fmt for fmt, path in paths.items() if not path.exists()}
        if miss:
            missing[key] = miss
    return missing


def _resume_state_path(recording_name: str, output_root: Path) -> Path:
    """Return the path of the per-recording resume state file."""
    return output_root / recording_name / RESUME_STATE_FILENAME


def _infer_available_keys_from_outputs(recording_name: str, output_root: Path) -> set[str]:
    """Infer available feature keys from existing output file names."""
    raw_dir = output_root / recording_name / "raw"
    if not raw_dir.is_dir():
        return set()

    keys: set[str] = set()
    for path in raw_dir.glob("features_*.parquet"):
        keys.add(path.stem.removeprefix(FEATURE_PREFIX))
    for path in raw_dir.glob("features_*.csv"):
        keys.add(path.stem.removeprefix(FEATURE_PREFIX))
    return keys


def _load_resume_state(recording_name: str, output_root: Path) -> dict | None:
    """Load per-recording resume state if present and readable."""
    state_path = _resume_state_path(recording_name, output_root)
    if not state_path.exists():
        return None
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - keep extraction resilient
        tqdm.write(f"WARNING: could not read resume state {state_path}: {exc}")
        return None


def _resume_state_matches_config(
    state: dict,
    *,
    csv_export: bool,
    trip_split_enabled: bool,
    topology_enabled: bool,
) -> bool:
    """Return True when saved state is compatible with current CLI settings."""
    return (
        bool(state.get("csv_export", False)) == bool(csv_export)
        and bool(state.get("trip_split", False)) == bool(trip_split_enabled)
        and bool(state.get("topology_enabled", False)) == bool(topology_enabled)
    )


def _recording_source_signature(files: list[Path]) -> str:
    """Return a stable signature for the current recording chunk set.

    The signature changes when files are added/removed or when size/mtime
    metadata changes, allowing incremental re-runs to process only updated
    recordings.
    """
    digest = hashlib.sha1()
    for path in sorted(files):
        stat = path.stat()
        digest.update(str(path).encode("utf-8", errors="replace"))
        digest.update(b"|")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"|")
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _resume_state_source_matches(state: dict, source_signature: str) -> bool:
    """Return True when saved state belongs to the same input chunk set."""
    return str(state.get("source_signature", "")) == source_signature


def _write_resume_state(
    recording_name: str,
    output_root: Path,
    available_keys: set[str],
    *,
    csv_export: bool,
    trip_split_enabled: bool,
    topology_enabled: bool,
    source_signature: str,
) -> None:
    """Persist a minimal marker so interrupted runs can resume efficiently."""
    state = {
        "version": 1,
        "available_keys": sorted(available_keys),
        "csv_export": bool(csv_export),
        "trip_split": bool(trip_split_enabled),
        "topology_enabled": bool(topology_enabled),
        "source_signature": source_signature,
    }
    state_path = _resume_state_path(recording_name, output_root)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _prepare_merge_key_df(key: str, frames: list[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("t_timestamp", kind="stable").reset_index(drop=True)
    df["_row_in_ts"] = df.groupby("t_timestamp", dropna=False).cumcount()

    renamed: dict[str, str] = {}
    for col in df.columns:
        if col in {"t_timestamp", "_row_in_ts"}:
            continue
        renamed[col] = f"{key}__{col}"
    return df.rename(columns=renamed)


def _merge_day_frames(by_key: dict[str, list[pd.DataFrame]]) -> pd.DataFrame:
    prepared = [
        _prepare_merge_key_df(key, frames)
        for key, frames in sorted(by_key.items())
        if frames
    ]
    if not prepared:
        return pd.DataFrame(columns=["t_timestamp"])

    merged = prepared[0]
    for nxt in prepared[1:]:
        merged = merged.merge(nxt, on=["t_timestamp", "_row_in_ts"], how="outer", sort=False)

    merged = merged.sort_values(["t_timestamp", "_row_in_ts"], kind="stable").reset_index(drop=True)
    return merged.rename(columns={"_row_in_ts": "row_in_timestamp"})


def merge_daily_outputs(output_root: Path, compression: str | None):
    """Build one merged CSV/Parquet per day under ``<output>``."""
    day_key_frames = _collect_day_key_frames(output_root)
    if not day_key_frames:
        return []
    day_dirs = _collect_day_recording_dirs(output_root)

    written: list[tuple[str, Path, Path, int, int]] = []
    for day_loco_key, by_key in sorted(day_key_frames.items()):
        merged = _merge_day_frames(by_key)
        if merged.empty:
            continue
        target_dirs = day_dirs.get(day_loco_key, [])
        out_dir = target_dirs[0] if target_dirs else output_root
        merged_dir = out_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = merged_dir / f"{day_loco_key}_merged.parquet"
        csv_path = merged_dir / f"{day_loco_key}_merged.csv"
        merged.to_parquet(parquet_path, index=False, compression=compression)
        merged.to_csv(csv_path, index=False)
        written.append((day_loco_key, parquet_path, csv_path, len(merged), len(merged.columns)))
    return written


def resolve_tshark(explicit: str | None) -> str:
    """Locate the tshark executable in a portable, machine-independent order.

    Priority: explicit ``--tshark`` argument, ``TSHARK_PATH`` env var, ``tshark``
    on ``PATH``, then common install locations (portable build listed last).
    """
    env_path = os.environ.get("TSHARK_PATH")
    candidates = (
        explicit,
        env_path,
        shutil.which("tshark"),
        r"C:\Program Files\Wireshark\tshark.exe",
        r"C:\Program Files (x86)\Wireshark\tshark.exe",
        r"C:\Users\Anton\Documents\Data_acq\WiresharkPortable64_v4-4\App\Wireshark\tshark.exe",
    )
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    sys.exit(
        "ERROR: tshark not found. Install Wireshark, add it to PATH, "
        "set TSHARK_PATH, or pass --tshark PATH."
    )


def available_fields(tshark: str) -> set[str]:
    """Collect all field abbreviations known to this tshark build."""
    res = subprocess.run([tshark, "-G", "fields"], capture_output=True, text=True)
    fields: set[str] = set()
    for line in res.stdout.splitlines():
        if line.startswith("F\t"):
            cols = line.split("\t")
            if len(cols) > 2:
                fields.add(cols[2])
    return fields


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract ETCS/ATO features from recording folders with the structure "
            "<recording>/wireshark into LSTM-ready Parquet/CSV."
        ),
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_ROOT,
                        help="Input root with <recording>/wireshark. "
                             "Accepts .pcapng and .pcapng.gz (ignores .pcapng.txt.gz). "
                             "Default: input")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT,
                        help="Output directory. Default: output")
    parser.add_argument("--tshark", default=None,
                        help="Path to the tshark executable (auto-detected).")
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS,
                        help="Parallel tshark workers per recording. Default: 3. "
                             "Use 1 to minimise memory.")
    parser.add_argument("--recording-jobs", type=int, default=DEFAULT_RECORDING_JOBS,
                        help=(
                            "Number of recordings to process in parallel. "
                            "Default: 2."
                        ))
    parser.add_argument("--compression", default="zstd",
                        choices=["zstd", "snappy", "gzip", "brotli", "none"],
                        help="Parquet compression codec. Default: zstd")
    parser.add_argument("--csv-export", dest="csv_export", action="store_true",
                        help="Write CSV files alongside Parquet output (default: disabled).")
    parser.add_argument("--no-csv-export", dest="csv_export", action="store_false",
                        help="Disable CSV output and write only Parquet files.")
    parser.add_argument("--trip-split", dest="trip_split", action="store_true",
                        help=(
                            "Write additional trip folders under each recording "
                            "(default: disabled)."
                        ))
    parser.add_argument("--no-trip-split", dest="trip_split", action="store_false",
                        help="Disable trip-wise export folders.")
    parser.add_argument("--daily-merge", dest="daily_merge", action="store_true",
                        help="Create daily merged files under output/<recording>/merged.")
    parser.add_argument("--no-daily-merge", dest="daily_merge", action="store_false",
                        help="Skip daily merged outputs (default).")
    parser.add_argument("--trip-standstill-minutes", type=float,
                        default=DEFAULT_TRIP_STANDSTILL_MINUTES,
                        help=(
                            "Split at standstill when speed and brake stay below "
                            "thresholds for at least this many minutes. Default: 5.0"
                        ))
    parser.add_argument("--trip-min-duration-minutes", type=float,
                        default=DEFAULT_TRIP_MIN_DURATION_MINUTES,
                        help=(
                            "Keep only trips with at least this duration in minutes. "
                            "Default: 5.0"
                        ))
    parser.add_argument("--trip-speed-threshold", type=float,
                        default=DEFAULT_TRIP_SPEED_THRESHOLD,
                        help=(
                            "Standstill speed threshold in raw units. "
                            "Default: 5.0 (approx. 0.05 m/s when raw/100)."
                        ))
    parser.add_argument("--trip-brake-threshold", type=float,
                        default=DEFAULT_TRIP_BRAKE_THRESHOLD,
                        help=(
                            "Standstill brake-lever threshold in raw units. "
                            "Default: 82 (approx. 0.005 when raw/16384)."
                        ))
    parser.add_argument("--trip-max-gap-seconds", type=float,
                        default=DEFAULT_TRIP_MAX_GAP_SECONDS,
                        help=(
                            "Also split a trip wherever the data has a hole longer "
                            "than this many seconds (no samples at all). Prevents "
                            "long intra-trip gaps. 0 disables. Default: 60."
                        ))
    parser.add_argument("--trip-output-dir", default=DEFAULT_TRIP_OUTPUT_DIRNAME,
                        help="Trip output folder name under each recording. Default: trips")
    parser.add_argument(
        "--rebuild-global-trips",
        action="store_true",
        help=(
            "Rebuild a single global trip list from existing "
            "output/<recording>/raw/features_*.parquet without re-reading pcap files. "
            "Writes output/<trip-output-dir>/trip1..N."
        ),
    )
    parser.set_defaults(
        csv_export=False,
        trip_split=False,
        daily_merge=DEFAULT_DAILY_MERGE,
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    compression = None if args.compression == "none" else args.compression

    if args.trip_standstill_minutes < 0:
        sys.exit("ERROR: --trip-standstill-minutes must be >= 0")
    if args.trip_min_duration_minutes < 0:
        sys.exit("ERROR: --trip-min-duration-minutes must be >= 0")
    if args.trip_speed_threshold < 0:
        sys.exit("ERROR: --trip-speed-threshold must be >= 0")
    if args.trip_brake_threshold < 0:
        sys.exit("ERROR: --trip-brake-threshold must be >= 0")
    if args.trip_max_gap_seconds < 0:
        sys.exit("ERROR: --trip-max-gap-seconds must be >= 0")
    if args.recording_jobs < 1:
        sys.exit("ERROR: --recording-jobs must be >= 1")

    trip_output_dir = args.trip_output_dir.strip()
    if not trip_output_dir:
        sys.exit("ERROR: --trip-output-dir must not be empty")

    trip_split_config = TripSplitConfig(
        enabled=args.trip_split,
        standstill_min_minutes=args.trip_standstill_minutes,
        min_trip_minutes=args.trip_min_duration_minutes,
        speed_threshold=args.trip_speed_threshold,
        brake_threshold=args.trip_brake_threshold,
        max_gap_seconds=args.trip_max_gap_seconds,
        output_dirname=trip_output_dir,
    )

    if args.rebuild_global_trips:
        written, dropped_short, recordings_used = rebuild_global_trips_from_parquet(
            output_root=args.output,
            compression=compression,
            csv_export=args.csv_export,
            cfg=trip_split_config,
        )
        print("=== Global Trip Rebuild (from Parquet) ===")
        print(f"source recordings: {recordings_used}")
        print(f"dropped short trips (< {trip_split_config.min_trip_minutes:.2f} min): {dropped_short}")
        if not written:
            print("No trips written.")
        else:
            by_trip: dict[int, int] = defaultdict(int)
            for trip_id, _, _, _ in written:
                by_trip[trip_id] += 1
            print(f"written trips: {len(by_trip)} -> {args.output / trip_split_config.output_dirname}")
            for trip_id, key, out_path, rows in written:
                print(f"  trip{trip_id} features_{key}.parquet: {rows:,} rows -> {out_path}")
        print("Done.")
        return

    tshark = resolve_tshark(args.tshark)

    print(f"tshark: {tshark}")
    fields = available_fields(tshark)
    if HDR_NID not in fields:
        sys.exit("ERROR: this tshark build does not know the 'aoecl' protocol.")

    plan = build_plan(fields)
    if plan.missing_fields:
        print("WARNING: fields not available in this tshark build "
              "(written as empty NaN columns):")
        for fld in sorted(plan.missing_fields):
            print(f"  - {fld}")

    topology_enabled = plan.topology_enabled
    if topology_enabled:
        print("Topology parser enabled (same pass): era-subset-126.nid_packet_ato == 7")
    else:
        print("WARNING: topology parser disabled (missing tshark fields):")
        for fld in plan.missing_topology:
            print(f"  - {fld}")

    recordings = find_recordings(args.input)
    if not recordings:
        sys.exit(f"No .pcapng files found under {args.input}")

    total_recordings = len(recordings)
    total_files = sum(len(files) for _, files in recordings)
    total_bytes_all = sum(p.stat().st_size for _, files in recordings for p in files)
    jobs = max(1, args.jobs)
    recording_jobs = max(1, args.recording_jobs)
    # Guard against CPU oversubscription: recordings run in a thread pool, each
    # spawning up to ``jobs`` tshark worker processes. Cap the product so the two
    # levels of parallelism together stay within the available cores.
    cpu_count = os.cpu_count() or 1
    if recording_jobs > 1 and jobs * recording_jobs > cpu_count:
        capped_jobs = max(1, cpu_count // recording_jobs)
        if capped_jobs < jobs:
            print(
                f"NOTE: capping jobs per recording {jobs} -> {capped_jobs} to avoid "
                f"oversubscription ({jobs}x{recording_jobs} > {cpu_count} cores)."
            )
            jobs = capped_jobs
    summary = []
    corrupted_recordings: set[str] = set()
    recording_available_keys: dict[str, set[str]] = {}
    recording_bad_files: dict[str, set[Path]] = {}

    pending_recordings: list[tuple[str, list[Path]]] = []
    resumed_recordings = 0
    changed_recordings = 0
    for name, files in recordings:
        source_signature = _recording_source_signature(files)
        state = _load_resume_state(name, args.output)
        if state is not None and _resume_state_matches_config(
            state,
            csv_export=args.csv_export,
            trip_split_enabled=trip_split_config.enabled,
            topology_enabled=topology_enabled,
        ):
            if _resume_state_source_matches(state, source_signature):
                state_keys = {str(key) for key in state.get("available_keys", [])}
                missing = _missing_recording_outputs(
                    name,
                    args.output,
                    args.csv_export,
                    topology_enabled,
                    state_keys,
                )
                if not missing:
                    recording_available_keys[name] = state_keys
                    recording_bad_files[name] = set()
                    resumed_recordings += 1
                    continue
            else:
                changed_recordings += 1

        inferred_keys = _infer_available_keys_from_outputs(name, args.output)
        if inferred_keys:
            missing = _missing_recording_outputs(
                name,
                args.output,
                args.csv_export,
                topology_enabled,
                inferred_keys,
            )
            if not missing:
                recording_available_keys[name] = inferred_keys
                recording_bad_files[name] = set()
                resumed_recordings += 1
                continue

        pending_recordings.append((name, files))

    total_bytes = sum(p.stat().st_size for _, files in pending_recordings for p in files)
    n_files = sum(len(files) for _, files in pending_recordings)
    print(
        f"Recordings: {total_recordings} (resume-skip: {resumed_recordings}, changed: {changed_recordings}, to-process: {len(pending_recordings)}) | "
        f"files: {n_files}/{total_files} | {total_bytes / 1e9:.2f}/{total_bytes_all / 1e9:.2f} GB | "
        f"jobs: {jobs} | recording-jobs: {recording_jobs}"
    )

    if recording_jobs == 1:
        with tqdm(total=total_bytes, unit="B", unit_scale=True, unit_divisor=1024,
                  desc="Extracting", smoothing=0.05) as bar:
            for name, files in pending_recordings:
                result = process_recording(
                    name,
                    files,
                    plan,
                    tshark,
                    args.output,
                    jobs,
                    bar,
                    compression,
                    topology_enabled,
                    args.csv_export,
                    trip_split_config,
                )
                recording_available_keys[name] = result.available_keys
                recording_bad_files[name] = set(result.aoecl_failed_paths)
                _write_resume_state(
                    name,
                    args.output,
                    result.available_keys,
                    csv_export=args.csv_export,
                    trip_split_enabled=trip_split_config.enabled,
                    topology_enabled=topology_enabled,
                    source_signature=_recording_source_signature(files),
                )
                if files and result.aoecl_failed_files >= len(files):
                    corrupted_recordings.add(name)
                    tqdm.write(
                        f"WARNING: skipping corrupted recording {name} "
                        f"({result.aoecl_failed_files}/{len(files)} chunks unreadable)"
                    )
                summary.append(
                    (
                        name,
                        result.aoecl_packets,
                        result.topology_packets,
                        result.written_parquet,
                        result.written_csv,
                        result.aoecl_failed_files,
                        result.topology_failed_files,
                        len(files),
                        result.trip_written_parquet,
                        result.trip_written_csv,
                        result.trip_count,
                    )
                )
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with tqdm(total=len(pending_recordings), unit="rec", desc="Extracting recordings") as rec_bar:
            with ThreadPoolExecutor(max_workers=recording_jobs) as pool:
                futures = {
                    pool.submit(
                        process_recording,
                        name,
                        files,
                        plan,
                        tshark,
                        args.output,
                        jobs,
                        None,
                        compression,
                        topology_enabled,
                        args.csv_export,
                        trip_split_config,
                    ): (name, files)
                    for name, files in pending_recordings
                }
                for future in as_completed(futures):
                    name, files = futures[future]
                    result = future.result()
                    recording_available_keys[name] = result.available_keys
                    recording_bad_files[name] = set(result.aoecl_failed_paths)
                    _write_resume_state(
                        name,
                        args.output,
                        result.available_keys,
                        csv_export=args.csv_export,
                        trip_split_enabled=trip_split_config.enabled,
                        topology_enabled=topology_enabled,
                        source_signature=_recording_source_signature(files),
                    )
                    if files and result.aoecl_failed_files >= len(files):
                        corrupted_recordings.add(name)
                        tqdm.write(
                            f"WARNING: skipping corrupted recording {name} "
                            f"({result.aoecl_failed_files}/{len(files)} chunks unreadable)"
                        )
                    summary.append(
                        (
                            name,
                            result.aoecl_packets,
                            result.topology_packets,
                            result.written_parquet,
                            result.written_csv,
                            result.aoecl_failed_files,
                            result.topology_failed_files,
                            len(files),
                            result.trip_written_parquet,
                            result.trip_written_csv,
                            result.trip_count,
                        )
                    )
                    rec_bar.update(1)

    print("\n=== Summary ===")
    for (
        name,
        aoecl_packets,
        topology_packets,
        written_parquet,
        written_csv,
        aoecl_failed_files,
        topology_failed_files,
        n_chunks,
        trip_written_parquet,
        trip_written_csv,
        trip_count,
    ) in summary:
        if topology_enabled:
            print(f"{name}: {aoecl_packets:,} aoecl packets | "
                  f"{topology_packets:,} topology packets")
        else:
            print(f"{name}: {aoecl_packets:,} aoecl packets")
        if aoecl_failed_files or topology_failed_files:
            print(
                f"  chunk failures: aoecl={aoecl_failed_files}/{n_chunks}, "
                f"topology={topology_failed_files}/{n_chunks}"
            )
        if not written_parquet and not written_csv:
            print("  (no matching packets)")
        for key, (out_path, rows) in written_parquet.items():
            print(f"  features_{key}.parquet: {rows:,} rows -> {out_path}")
        for key, (out_path, rows) in written_csv.items():
            print(f"  features_{key}.csv: {rows:,} rows -> {out_path}")
        if trip_split_config.enabled:
            trip_root = args.output / name / trip_split_config.output_dirname
            csv_info = f", {len(trip_written_csv)} csv" if args.csv_export else ""
            print(
                f"  trip split: {trip_count} trip(s), "
                f"{len(trip_written_parquet)} parquet{csv_info} -> {trip_root}"
            )

    check_jobs: list[tuple[str, list[Path], dict[str, set[str]]]] = []
    for name, files in recordings:
        if name in corrupted_recordings:
            continue
        missing = _missing_recording_outputs(
            name,
            args.output,
            args.csv_export,
            topology_enabled,
            recording_available_keys.get(name),
        )
        if missing:
            check_jobs.append((name, files, missing))

    print("\n=== Output Check ===")
    if not check_jobs:
        print("All expected output files are present.")
    else:
        print(
            f"Missing outputs found for {len(check_jobs)} recording(s). "
            "Re-exporting only missing files..."
        )
        if recording_jobs == 1:
            retry_bytes = sum(p.stat().st_size for _, files, _ in check_jobs for p in files)
            with tqdm(total=retry_bytes, unit="B", unit_scale=True, unit_divisor=1024,
                      desc="Re-export missing", smoothing=0.05) as bar:
                for name, files, missing in check_jobs:
                    result = process_recording(
                        name,
                        files,
                        plan,
                        tshark,
                        args.output,
                        jobs,
                        bar,
                        compression,
                        topology_enabled,
                        args.csv_export,
                        trip_split_config,
                        missing_only=missing,
                        exclude_files=recording_bad_files.get(name, set()),
                    )
                    if result.aoecl_failed_paths:
                        recording_bad_files.setdefault(name, set()).update(result.aoecl_failed_paths)
                    miss_desc = ", ".join(
                        f"{key}({','.join(sorted(fmts))})"
                        for key, fmts in sorted(missing.items())
                    )
                    print(f"{name}: retried missing -> {miss_desc}")
                    if result.aoecl_failed_files or result.topology_failed_files:
                        print(
                            f"  retry chunk failures: aoecl={result.aoecl_failed_files}/{len(files)}, "
                            f"topology={result.topology_failed_files}/{len(files)}"
                        )
                    for key, (out_path, rows) in result.written_parquet.items():
                        print(f"  wrote features_{key}.parquet: {rows:,} rows -> {out_path}")
                    for key, (out_path, rows) in result.written_csv.items():
                        print(f"  wrote features_{key}.csv: {rows:,} rows -> {out_path}")
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with tqdm(total=len(check_jobs), unit="rec", desc="Re-export missing") as rec_bar:
                with ThreadPoolExecutor(max_workers=recording_jobs) as pool:
                    futures = {
                        pool.submit(
                            process_recording,
                            name,
                            files,
                            plan,
                            tshark,
                            args.output,
                            jobs,
                            None,
                            compression,
                            topology_enabled,
                            args.csv_export,
                            trip_split_config,
                            missing,
                            recording_bad_files.get(name, set()),
                        ): (name, files, missing)
                        for name, files, missing in check_jobs
                    }
                    for future in as_completed(futures):
                        name, files, missing = futures[future]
                        result = future.result()
                        if result.aoecl_failed_paths:
                            recording_bad_files.setdefault(name, set()).update(result.aoecl_failed_paths)
                        miss_desc = ", ".join(
                            f"{key}({','.join(sorted(fmts))})"
                            for key, fmts in sorted(missing.items())
                        )
                        print(f"{name}: retried missing -> {miss_desc}")
                        if result.aoecl_failed_files or result.topology_failed_files:
                            print(
                                f"  retry chunk failures: aoecl={result.aoecl_failed_files}/{len(files)}, "
                                f"topology={result.topology_failed_files}/{len(files)}"
                            )
                        for key, (out_path, rows) in result.written_parquet.items():
                            print(f"  wrote features_{key}.parquet: {rows:,} rows -> {out_path}")
                        for key, (out_path, rows) in result.written_csv.items():
                            print(f"  wrote features_{key}.csv: {rows:,} rows -> {out_path}")
                        rec_bar.update(1)

        unresolved: list[tuple[str, dict[str, set[str]]]] = []
        for name, _, _ in check_jobs:
            missing = _missing_recording_outputs(
                name,
                args.output,
                args.csv_export,
                topology_enabled,
                recording_available_keys.get(name),
            )
            if missing:
                unresolved.append((name, missing))
        if unresolved:
            print("WARNING: unresolved missing outputs remain:")
            for name, missing in unresolved:
                miss_desc = ", ".join(
                    f"{key}({','.join(sorted(fmts))})"
                    for key, fmts in sorted(missing.items())
                )
                print(f"  {name}: {miss_desc}")

    if args.daily_merge:
        daily_written = merge_daily_outputs(args.output, compression)
        if daily_written:
            print("\n=== Daily Merge ===")
            for day, parquet_path, csv_path, rows, cols in daily_written:
                print(
                    f"{day}: {rows:,} rows, {cols:,} cols -> "
                    f"{parquet_path} | {csv_path}"
                )
    else:
        print("\nDaily merge skipped (--no-daily-merge).")
    print("Done.")


if __name__ == "__main__":
    main()
