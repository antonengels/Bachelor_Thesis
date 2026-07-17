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
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
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
    return ExtractionPlan(master, groups, nid_to_group, valid, missing, display_filter)


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
def process_single_file(tshark: str, path: Path, plan: ExtractionPlan):
    """Stream one pcapng through tshark and route rows into feature groups.

    Returns ``(dict[group_key -> DataFrame], packet_count)``.
    """
    cmd = plan.build_command(tshark, path)
    ts: dict[str, list[str]] = {gp.spec.key: [] for gp in plan.groups}
    raw: dict[str, dict[str, list[str]]] = {
        gp.spec.key: {fld: [] for fld in gp.valid_fields} for gp in plan.groups
    }
    nid_to_group = plan.nid_to_group
    packets = 0

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, bufsize=1 << 20,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        parts = line.rstrip("\r\n").split("\t")
        token = parts[0]
        if not token:
            continue
        try:
            nid = int(token)
        except ValueError:
            continue
        group = nid_to_group.get(nid)
        if group is None:
            continue
        packets += 1
        key = group.spec.key
        ts[key].append(parts[1] if len(parts) > 1 else "")
        buffers = raw[key]
        for fld, idx in group.field_index.items():
            buffers[fld].append(parts[idx] if idx < len(parts) else "")
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"tshark failed with exit code {rc}")

    out: dict[str, pd.DataFrame] = {}
    for group in plan.groups:
        key = group.spec.key
        if ts[key]:
            out[key] = _build_group_df(group, ts[key], raw[key])
    return out, packets


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


def _topology_command(tshark: str, path: Path) -> list[str]:
    cmd = [
        tshark, "-r", str(path),
        "-Y", TOPO_FILTER,
        "-T", "fields",
        "-E", "separator=\t",
        "-E", "occurrence=a",
        "-E", "aggregator=,",
    ]
    for fld in TOPO_FIELDS:
        cmd.extend(("-e", fld))
    return cmd


def process_single_topology_file(tshark: str, path: Path):
    """Extract ERA Subset-126 packet-7 segment profile curve-radius events."""
    cmd = _topology_command(tshark, path)
    rows: list[dict[str, int | None]] = []
    packets = 0

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, bufsize=1 << 20,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        parts = line.rstrip("\r\n").split("\t")
        nid_values = _parse_int_csv(parts[1] if len(parts) > 1 else "")
        if 7 not in nid_values:
            continue
        packets += 1
        rows.extend(parse_segment_profile_radius_rows(
            frame_time_epoch_raw=parts[0] if len(parts) > 0 else "",
            n_segments_raw=parts[2] if len(parts) > 2 else "",
            segment_lengths_raw=parts[3] if len(parts) > 3 else "",
            segment_start_radius_raw=parts[4] if len(parts) > 4 else "",
            curve_change_counts_raw=parts[5] if len(parts) > 5 else "",
            curve_change_locations_raw=parts[6] if len(parts) > 6 else "",
            curve_change_radius_raw=parts[7] if len(parts) > 7 else "",
        ))
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"tshark failed with exit code {rc}")
    return _build_topology_df(rows), packets


def _process_topology_files(files, tshark, jobs, label):
    """Process packet-7 topology extraction for all chunks of one recording."""
    results: list[pd.DataFrame | None] = [None] * len(files)
    total_packets = 0
    failed_files = 0

    if jobs <= 1:
        for i, path in enumerate(files):
            try:
                results[i], n = process_single_topology_file(tshark, path)
            except Exception as exc:  # noqa: BLE001 - keep going on a bad chunk
                tqdm.write(f"WARNING: failed topology parse {path.name}: {exc}")
                results[i], n = pd.DataFrame(columns=list(TOPOLOGY_COLUMNS)), 0
                failed_files += 1
            total_packets += n
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {
                pool.submit(process_single_topology_file, tshark, path): i
                for i, path in enumerate(files)
            }
            for future in as_completed(futures):
                i = futures[future]
                path = files[i]
                try:
                    results[i], n = future.result()
                except Exception as exc:  # noqa: BLE001
                    tqdm.write(f"WARNING: failed topology parse {path.name}: {exc}")
                    results[i], n = pd.DataFrame(columns=list(TOPOLOGY_COLUMNS)), 0
                    failed_files += 1
                total_packets += n

    frames = [df for df in results if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame(columns=list(TOPOLOGY_COLUMNS)), total_packets, failed_files
    return pd.concat(frames, ignore_index=True), total_packets, failed_files


# --------------------------------------------------------------------------- #
# Recording driver
# --------------------------------------------------------------------------- #
def _process_files(files, plan, tshark, jobs, bar, label):
    """Process all chunks of one recording (optionally in parallel)."""
    results: list[dict | None] = [None] * len(files)
    total_packets = 0
    failed_files = 0

    if jobs <= 1:
        for i, path in enumerate(files):
            bar.set_postfix_str(f"{label} \u00b7 {path.name}", refresh=False)
            try:
                results[i], n = process_single_file(tshark, path, plan)
            except Exception as exc:  # noqa: BLE001 - keep going on a bad chunk
                tqdm.write(f"WARNING: failed to process {path.name}: {exc}")
                results[i], n = {}, 0
                failed_files += 1
            total_packets += n
            bar.update(path.stat().st_size)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {
                pool.submit(process_single_file, tshark, path, plan): i
                for i, path in enumerate(files)
            }
            for future in as_completed(futures):
                i = futures[future]
                path = files[i]
                try:
                    results[i], n = future.result()
                except Exception as exc:  # noqa: BLE001
                    tqdm.write(f"WARNING: failed to process {path.name}: {exc}")
                    results[i], n = {}, 0
                    failed_files += 1
                total_packets += n
                bar.set_postfix_str(f"{label} \u00b7 {path.name}", refresh=False)
                bar.update(path.stat().st_size)

    per_group: dict[str, list[pd.DataFrame]] = defaultdict(list)
    for dfs in results:
        if dfs:
            for key, df in dfs.items():
                per_group[key].append(df)
    return per_group, total_packets, failed_files


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
    missing_only: dict[str, set[str]] | None = None,
):
    """Extract one recording and write per-group output files."""
    per_group, packets, aoecl_failed_files = _process_files(files, plan, tshark, jobs, bar, name)
    out_dir = out_root / name
    raw_dir = out_dir / "raw"
    written_parquet: dict[str, tuple[Path, int]] = {}
    written_csv: dict[str, tuple[Path, int]] = {}
    available_keys: set[str] = set()

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

    topology_packets = 0
    topology_failed_files = 0
    if topology_enabled:
        topology_formats_to_write = {"parquet"}
        if csv_export:
            topology_formats_to_write.add("csv")
        if missing_only is not None:
            topology_formats_to_write &= missing_only.get("topology_radius", set())

        topology_df, topology_packets, topology_failed_files = _process_topology_files(
            files, tshark, jobs, name
        )
        if not topology_df.empty and topology_formats_to_write:
            lstm_topology_df = _to_lstm_ready("topology_radius", topology_df)
            if not lstm_topology_df.empty:
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

    return (
        written_parquet,
        written_csv,
        packets,
        topology_packets,
        aoecl_failed_files,
        topology_failed_files,
        available_keys,
    )


# --------------------------------------------------------------------------- #
# Discovery & environment helpers
# --------------------------------------------------------------------------- #
def find_recordings(input_path: Path):
    """Return ``[(recording_name, [pcapng_files])]`` for the given input path."""
    if input_path.is_file():
        if not _is_capture_file(input_path):
            sys.exit("Input file must be .pcapng or .pcapng.gz (excluding .pcapng.txt.gz).")
        return [(_canonical_recording_name(input_path), [input_path])]

    recordings = []
    for sub in sorted(p for p in input_path.iterdir() if p.is_dir()):
        wireshark_dir = sub / "wireshark"
        if not wireshark_dir.is_dir():
            continue
        files = sorted(
            p for p in wireshark_dir.rglob("*")
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


def _collect_day_key_frames(output_root: Path):
    day_key_frames: dict[str, dict[str, list[pd.DataFrame]]] = {}

    for rec_dir in sorted(p for p in output_root.iterdir() if p.is_dir()):
        day = _recording_day(rec_dir.name)
        if day is None:
            continue

        raw_dir = rec_dir / "raw"
        files = sorted(raw_dir.glob("features_*.parquet")) if raw_dir.exists() else []
        if not files:
            continue

        by_key = day_key_frames.setdefault(day, {})
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
        day = _recording_day(rec_dir.name)
        if day is None:
            continue
        day_dirs.setdefault(day, []).append(rec_dir)
    return day_dirs


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
    for day, by_key in sorted(day_key_frames.items()):
        merged = _merge_day_frames(by_key)
        if merged.empty:
            continue
        target_dirs = day_dirs.get(day, [])
        out_dir = target_dirs[0] if target_dirs else output_root
        merged_dir = out_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = merged_dir / f"{day}_merged.parquet"
        csv_path = merged_dir / f"{day}_merged.csv"
        merged.to_parquet(parquet_path, index=False, compression=compression)
        merged.to_csv(csv_path, index=False)
        written.append((day, parquet_path, csv_path, len(merged), len(merged.columns)))
    return written


def resolve_tshark(explicit: str | None) -> str:
    if explicit:
        return explicit
    for candidate in (
        r"C:\Users\Anton\Documents\Data_acq\WiresharkPortable64_v4-4\App\Wireshark\tshark.exe",
        shutil.which("tshark"),
        r"C:\Program Files\Wireshark\tshark.exe",
                      r"C:\Program Files (x86)\Wireshark\tshark.exe"):
        if candidate and Path(candidate).exists():
            return candidate
    sys.exit("ERROR: tshark not found. Install Wireshark or pass --tshark PATH.")


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
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1),
                        help="Parallel tshark workers per recording. Default: "
                             "min(4, CPU count). Use 1 to minimise memory.")
    parser.add_argument("--compression", default="zstd",
                        choices=["zstd", "snappy", "gzip", "brotli", "none"],
                        help="Parquet compression codec. Default: zstd")
    parser.add_argument("--csv-export", dest="csv_export", action="store_true",
                        help="Write CSV files alongside Parquet output (default: enabled).")
    parser.add_argument("--no-csv-export", dest="csv_export", action="store_false",
                        help="Disable CSV output and write only Parquet files.")
    parser.set_defaults(csv_export=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    tshark = resolve_tshark(args.tshark)
    compression = None if args.compression == "none" else args.compression

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

    missing_topology = sorted(TOPO_REQUIRED_FIELDS.difference(fields))
    topology_enabled = not missing_topology
    if topology_enabled:
        print("Topology parser enabled: era-subset-126.nid_packet_ato == 7")
    else:
        print("WARNING: topology parser disabled (missing tshark fields):")
        for fld in missing_topology:
            print(f"  - {fld}")

    recordings = find_recordings(args.input)
    if not recordings:
        sys.exit(f"No .pcapng files found under {args.input}")

    total_bytes = sum(p.stat().st_size for _, files in recordings for p in files)
    n_files = sum(len(files) for _, files in recordings)
    jobs = max(1, args.jobs)
    print(f"Recordings: {len(recordings)} | files: {n_files} | "
          f"{total_bytes / 1e9:.2f} GB | jobs: {jobs}")

    summary = []
    corrupted_recordings: set[str] = set()
    recording_available_keys: dict[str, set[str]] = {}
    with tqdm(total=total_bytes, unit="B", unit_scale=True, unit_divisor=1024,
              desc="Extracting", smoothing=0.05) as bar:
        for name, files in recordings:
            (written_parquet,
             written_csv,
             aoecl_packets,
             topology_packets,
             aoecl_failed_files,
             topology_failed_files,
             available_keys) = process_recording(
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
            )
            recording_available_keys[name] = available_keys
            if files and aoecl_failed_files >= len(files):
                corrupted_recordings.add(name)
                tqdm.write(
                    f"WARNING: skipping corrupted recording {name} "
                    f"({aoecl_failed_files}/{len(files)} chunks unreadable)"
                )
            summary.append(
                (
                    name,
                    aoecl_packets,
                    topology_packets,
                    written_parquet,
                    written_csv,
                    aoecl_failed_files,
                    topology_failed_files,
                    len(files),
                )
            )

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
        retry_bytes = sum(p.stat().st_size for _, files, _ in check_jobs for p in files)
        with tqdm(total=retry_bytes, unit="B", unit_scale=True, unit_divisor=1024,
                  desc="Re-export missing", smoothing=0.05) as bar:
            for name, files, missing in check_jobs:
                (written_parquet,
                 written_csv,
                 _,
                 _,
                 aoecl_failed_files,
                      topology_failed_files,
                      _) = process_recording(
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
                    missing_only=missing,
                )
                miss_desc = ", ".join(
                    f"{key}({','.join(sorted(fmts))})"
                    for key, fmts in sorted(missing.items())
                )
                print(f"{name}: retried missing -> {miss_desc}")
                if aoecl_failed_files or topology_failed_files:
                    print(
                        f"  retry chunk failures: aoecl={aoecl_failed_files}/{len(files)}, "
                        f"topology={topology_failed_files}/{len(files)}"
                    )
                for key, (out_path, rows) in written_parquet.items():
                    print(f"  wrote features_{key}.parquet: {rows:,} rows -> {out_path}")
                for key, (out_path, rows) in written_csv.items():
                    print(f"  wrote features_{key}.csv: {rows:,} rows -> {out_path}")

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

    daily_written = merge_daily_outputs(args.output, compression)
    if daily_written:
        print("\n=== Daily Merge ===")
        for day, parquet_path, csv_path, rows, cols in daily_written:
            print(
                f"{day}: {rows:,} rows, {cols:,} cols -> "
                f"{parquet_path} | {csv_path}"
            )
    print("Done.")


if __name__ == "__main__":
    main()
