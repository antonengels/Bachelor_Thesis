#!/usr/bin/env python3
"""High-performance ETCS/ATO feature extraction from PCAPNG to Parquet.

The input directory is expected to contain one sub-folder per recording, each
holding many sequential ``*.pcapng`` chunks (e.g. ``input/<recording>/wireshark/
*.pcapng``).  For every recording all chunks are streamed through a single
``tshark -T fields`` pass, filtered on the ``aoecl`` application layer, scaled /
validated and written as one compressed Parquet file per ``NID_PACKET`` group
into ``output/<recording>/features_<key>.parquet``.

Using ``subprocess`` + ``tshark -T fields`` (instead of pyshark) keeps the hot
path in native code and is dramatically faster.  Rows are routed to their
feature group by ``aoecl.header.NID_PACKET`` in a single pass; the ``tshark``
field list declared in :data:`FEATURE_SPEC` is validated against the running
``tshark`` build so missing signals degrade gracefully to empty (NaN) columns.

Example:
    python src/extract_features.py --input input --output output --jobs 4
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# aoecl header fields (always extracted, used for routing + time key)
# --------------------------------------------------------------------------- #
HDR_NID = "aoecl.header.NID_PACKET"
HDR_TS = "aoecl.header.T_TIMESTAMP"


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
    error: int | None    # raw sentinel value that maps to NaN
    scale: float | None  # divide valid raw values by this factor
    as_int: bool = False  # keep as nullable integer instead of float


@dataclass(frozen=True)
class Gradient:
    """A comma-separated array field expanded to a fixed number of columns."""

    name: str            # column prefix -> name[0] .. name[width-1]
    field: str
    width: int
    error: int | None
    scale: float | None


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
            Scalar("V_EST", "aoecl.userdata.nid6.V_EST", 65535, 100.0),
            Scalar("V_MRSP", "aoecl.userdata.nid6.V_MRSP", 65535, 100.0),
            Scalar("V_PERMITTED", "aoecl.userdata.nid6.V_PERMITTED", 65535, 100.0),
            Scalar("V_TARGET", "aoecl.userdata.nid6.V_TARGET", 65535, 100.0),
            Scalar("A_EST", "aoecl.userdata.nid6.A_EST", 32767, 1000.0),
        ),
        gradients=(
            Gradient("grad", "aoecl.userdata.nid6.A_GRADIENT", 10, 32767, 1000.0),
        ),
    ),
    # NID 31 / 100 — label (brake lever request)
    Group(
        key="nid31",
        nids=(31, 100),
        scalars=(
            Scalar("M_ATO_RTBRq", "aoecl.userdata.nid31.M_ATO_RTBRq", 32767, 16384.0),
        ),
    ),
    # NID 1 — distance
    Group(
        key="nid1",
        nids=(1,),
        scalars=(
            Scalar("D_STPDISTANCE", "aoecl.userdata.nid1.D_STPDISTANCE",
                   4294967295, 100.0),
        ),
    ),
    # NID 32 — vehicle feedback
    Group(
        key="nid32",
        nids=(32,),
        scalars=(
            Scalar("M_RST_TBsetVal", "aoecl.userdata.nid32.M_RST_TBsetVal",
                   32767, 16384.0),
            Scalar("M_RST_SlipSlide", "aoecl.userdata.nid32.M_RST_SlipSlide",
                   None, None, as_int=True),
        ),
    ),
    # NID 4 / 103 — adhesion
    Group(
        key="nid4",
        nids=(4, 103),
        scalars=(
            Scalar("Q_ADHESION", "aoecl.userdata.nid4.Q_ADHESION",
                   None, None, as_int=True),
        ),
    ),
    # NID 101 / 102 / 107 — driver inputs / status
    Group(
        key="status",
        nids=(101, 102, 107),
        scalars=(
            Scalar("Door_Status", "aoecl.userdata.nid101.Door_Status",
                   None, None, as_int=True),
            Scalar("ATO_State", "aoecl.userdata.nid102.ATO_State",
                   None, None, as_int=True),
            Scalar("Skip_Stop_Pt", "aoecl.userdata.nid107.Skip_Stop_Pt",
                   None, None, as_int=True),
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
    master = [HDR_NID, HDR_TS]
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
    """Convert raw tshark strings into a scaled, validated pandas Series."""
    series = pd.to_numeric(pd.Series(raw, dtype="object"), errors="coerce")
    if spec.error is not None:
        series = series.mask(series == spec.error)
    if spec.scale is not None:
        series = series / spec.scale
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
    if spec.error is not None:
        num = num.mask(num == spec.error)
    if spec.scale is not None:
        num = num / spec.scale
    num.columns = columns
    return num.astype("float64")


def _empty_scalar(spec: Scalar, n: int) -> pd.Series:
    if spec.as_int:
        return pd.Series(pd.array([pd.NA] * n, dtype="Int64"))
    return pd.Series(np.full(n, np.nan), dtype="float64")


def _build_group_df(plan: GroupPlan, ts: list[str], raw: dict[str, list[str]]) -> pd.DataFrame:
    """Assemble one group's DataFrame for a single file."""
    n = len(ts)
    data: dict[str, pd.Series] = {
        "t_timestamp": pd.to_numeric(pd.Series(ts, dtype="object"),
                                     errors="coerce").astype("Int64"),
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
    proc.wait()

    out: dict[str, pd.DataFrame] = {}
    for group in plan.groups:
        key = group.spec.key
        if ts[key]:
            out[key] = _build_group_df(group, ts[key], raw[key])
    return out, packets


# --------------------------------------------------------------------------- #
# Recording driver
# --------------------------------------------------------------------------- #
def _process_files(files, plan, tshark, jobs, bar, label):
    """Process all chunks of one recording (optionally in parallel)."""
    results: list[dict | None] = [None] * len(files)
    total_packets = 0

    if jobs <= 1:
        for i, path in enumerate(files):
            bar.set_postfix_str(f"{label} \u00b7 {path.name}", refresh=False)
            try:
                results[i], n = process_single_file(tshark, path, plan)
            except Exception as exc:  # noqa: BLE001 - keep going on a bad chunk
                tqdm.write(f"WARNING: failed to process {path.name}: {exc}")
                results[i], n = {}, 0
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
                total_packets += n
                bar.set_postfix_str(f"{label} \u00b7 {path.name}", refresh=False)
                bar.update(path.stat().st_size)

    per_group: dict[str, list[pd.DataFrame]] = defaultdict(list)
    for dfs in results:
        if dfs:
            for key, df in dfs.items():
                per_group[key].append(df)
    return per_group, total_packets


def process_recording(name, files, plan, tshark, out_root, jobs, bar, compression):
    """Extract one recording and write its per-NID Parquet files."""
    per_group, packets = _process_files(files, plan, tshark, jobs, bar, name)
    out_dir = out_root / name
    written: dict[str, tuple[Path, int]] = {}
    for group in plan.groups:
        key = group.spec.key
        frames = per_group.get(key)
        if not frames:
            continue
        full = pd.concat(frames, ignore_index=True)
        if full.empty:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"features_{key}.parquet"
        full.to_parquet(out_path, index=False, compression=compression)
        written[key] = (out_path, len(full))
    return written, packets


# --------------------------------------------------------------------------- #
# Discovery & environment helpers
# --------------------------------------------------------------------------- #
def find_recordings(input_path: Path):
    """Return ``[(recording_name, [pcapng_files])]`` for the given input path."""
    if input_path.is_file():
        return [(input_path.stem, [input_path])]

    recordings = []
    for sub in sorted(p for p in input_path.iterdir() if p.is_dir()):
        files = sorted(sub.rglob("*.pcapng"))
        if files:
            recordings.append((sub.name, files))
    loose = sorted(input_path.glob("*.pcapng"))
    if loose:
        recordings.append((input_path.name, loose))
    return recordings


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
        description="Extract ETCS/ATO features from PCAPNG files into Parquet.",
    )
    parser.add_argument("--input", type=Path, default=Path("input"),
                        help="Input directory (recordings as sub-folders) or a "
                             "single .pcapng file. Default: input")
    parser.add_argument("--output", type=Path, default=Path("output"),
                        help="Output directory. Default: output")
    parser.add_argument("--tshark", default=None,
                        help="Path to the tshark executable (auto-detected).")
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1),
                        help="Parallel tshark workers per recording. Default: "
                             "min(4, CPU count). Use 1 to minimise memory.")
    parser.add_argument("--compression", default="zstd",
                        choices=["zstd", "snappy", "gzip", "brotli", "none"],
                        help="Parquet compression codec. Default: zstd")
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

    recordings = find_recordings(args.input)
    if not recordings:
        sys.exit(f"No .pcapng files found under {args.input}")

    total_bytes = sum(p.stat().st_size for _, files in recordings for p in files)
    n_files = sum(len(files) for _, files in recordings)
    jobs = max(1, args.jobs)
    print(f"Recordings: {len(recordings)} | files: {n_files} | "
          f"{total_bytes / 1e9:.2f} GB | jobs: {jobs}")

    summary = []
    with tqdm(total=total_bytes, unit="B", unit_scale=True, unit_divisor=1024,
              desc="Extracting", smoothing=0.05) as bar:
        for name, files in recordings:
            written, packets = process_recording(
                name, files, plan, tshark, args.output, jobs, bar, compression)
            summary.append((name, packets, written))

    print("\n=== Summary ===")
    for name, packets, written in summary:
        print(f"{name}: {packets:,} packets")
        if not written:
            print("  (no matching NID packets)")
        for key, (out_path, rows) in written.items():
            print(f"  features_{key}.parquet: {rows:,} rows -> {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
