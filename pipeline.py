#!/usr/bin/env python
"""Batch pipeline: thoracic measurements from chest CT DICOMs.

For each patient of the input root:
  1. Select the axial CT series with the most slices (>= --min-slices) and
     convert it to output/ct/<id>.nii.gz, with the geometry recomputed from the
     DICOM slice positions and checked against them.
  2. Run each TotalSegmentator task once, saving one mask plus its run report
     under output/masks/<id>/. A mask is reused whenever its report proves it
     was made for the same task, resolution and class list, so nothing is ever
     segmented twice and tasks can be added in later passes.
  3. Measure everything from those masks: craniocaudal lung lengths and the
     vertebral levels at apex and base (from the vertebral BODIES), plus
     parenchymal density, vessel and airway metrics, the carina, effusion
     volumes and the thoracic cavity.
  4. Append one row to <output>/full/results.csv immediately (crash-safe batch).

The input root is treated as strictly read-only.

Usage:
  uv run python pipeline.py --input /Volumes/DRIVE/cts --output ./out
  uv run python pipeline.py --input /Volumes/DRIVE/cts --output ./out --test
  uv run python pipeline.py --input /Volumes/DRIVE/cts --output ./out --dry-run
  uv run python pipeline.py --input /Volumes/DRIVE/cts --output ./out --tasks total,vertebrae_pp_refined
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pydicom
from scipy import ndimage
from tqdm import tqdm

log = logging.getLogger("pipeline")

# Mark a log record for the console. Without it a record still goes to the log
# file, which always keeps everything. --verbose puts everything on screen.
SHOW = {"console": True}
VERBOSE = False


def show_if(cond: bool) -> dict:
    return SHOW if cond else {}


@contextlib.contextmanager
def captured_output(label: str):
    """Send whatever a library prints to the log file instead of the console.

    TotalSegmentator announces each step with plain print(), and torch and
    nnU-Net add their own warnings. On a batch run that buries the one line that
    matters and says nothing the log file cannot hold.
    """
    if VERBOSE:
        yield
        return
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            yield
    finally:
        for line in buf.getvalue().splitlines():
            if line.strip():
                log.debug("    [%s] %s", label, line.rstrip())


def fmt_duration(seconds: float) -> str:
    m, sec = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"

CSV_COLUMNS = [
    "folder_id", "dicom_patient_id", "accession_number", "study_date", "id_mismatch",
    "start_R", "end_R", "start_L", "end_L", "start_both", "end_both",
    "height_R_mm", "height_L_mm", "height_both_mm", "height_sum_mm",
    "diag_R_mm", "diag_L_mm", "diag_both_mm", "diag_sum_mm",
    "width_R_mm", "width_L_mm", "width_both_mm", "width_sum_mm",
    "depth_R_mm", "depth_L_mm", "depth_both_mm", "depth_sum_mm",
    "vol_R_ml", "vol_L_ml", "vol_both_ml",
    "spine_height_mm", "spine_lung_span_mm", "cobb_angle_deg",
    # --- added in the multi-task extension; every one comes from a saved mask ---
    "slice_thickness_mm", "convolution_kernel",
    "lung_mean_hu", "laa950_pct",
    "artery_vol_ml", "vein_vol_ml", "artery_vein_ratio", "small_vessel_vol_ml",
    "airway_lumen_vol_ml", "airway_wall_vol_ml", "airway_lumen_lung_ratio",
    "airway_wall_pct", "airway_branch_count",
    "carina_level", "carina_to_apex_mm", "carina_to_base_mm",
    "pleural_eff_R_ml", "pleural_eff_L_ml", "pericardial_eff_ml",
    "thoracic_cavity_height_mm",
    "status", "runtime_s",
]
IDENTITY_COLUMNS = ("folder_id", "dicom_patient_id", "accession_number", "study_date",
                    "id_mismatch", "status", "runtime_s")

RIGHT_LOBES = ["lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"]
LEFT_LOBES = ["lung_upper_lobe_left", "lung_lower_lobe_left"]
LUNG_LOBES = RIGHT_LOBES + LEFT_LOBES

MIN_LUNG_VOXELS_PER_SLICE = 50  # ignore axial slices with fewer lung voxels at the extremes

# Fallback label ids (TotalSegmentator v2 "total" task) used only if the
# totalsegmentator package is not importable (e.g. measurement-only runs).
_FALLBACK_TOTAL_MAP = {
    10: "lung_upper_lobe_left", 11: "lung_lower_lobe_left", 12: "lung_upper_lobe_right",
    13: "lung_middle_lobe_right", 14: "lung_lower_lobe_right",
    26: "vertebrae_S1", 27: "vertebrae_L5", 28: "vertebrae_L4", 29: "vertebrae_L3",
    30: "vertebrae_L2", 31: "vertebrae_L1", 32: "vertebrae_T12", 33: "vertebrae_T11",
    34: "vertebrae_T10", 35: "vertebrae_T9", 36: "vertebrae_T8", 37: "vertebrae_T7",
    38: "vertebrae_T6", 39: "vertebrae_T5", 40: "vertebrae_T4", 41: "vertebrae_T3",
    42: "vertebrae_T2", 43: "vertebrae_T1", 44: "vertebrae_C7", 45: "vertebrae_C6",
    46: "vertebrae_C5", 47: "vertebrae_C4", 48: "vertebrae_C3", 49: "vertebrae_C2",
    50: "vertebrae_C1",
}


def get_label_maps() -> tuple[dict[str, int], dict[str, int]]:
    """Return (lobe name -> id, vertebra name -> id) for the "total" task."""
    try:
        from totalsegmentator.map_to_binary import class_map
        total = class_map["total"]
    except Exception:
        log.warning("totalsegmentator not importable; using built-in v2 label map fallback")
        total = _FALLBACK_TOTAL_MAP
    name_to_id = {name: idx for idx, name in total.items()}
    lobes = {n: name_to_id[n] for n in LUNG_LOBES}
    verts = {n: i for n, i in name_to_id.items() if n.startswith("vertebrae_")}
    return lobes, verts


VERT_ORDER = (
    [f"vertebrae_C{i}" for i in range(1, 8)]
    + [f"vertebrae_T{i}" for i in range(1, 13)]
    + [f"vertebrae_L{i}" for i in range(1, 6)]
    + ["vertebrae_S1"]
)


def vertebra_rank(name: str) -> int:
    """Anatomical order, 0 = most cranial (C1)."""
    try:
        return VERT_ORDER.index(name)
    except ValueError:
        return len(VERT_ORDER)


def short_name(vert: str | None) -> str:
    return vert.replace("vertebrae_", "") if vert else ""


TRACHEA_NAME = "trachea"

# Label ids of the extra tasks, read from TotalSegmentator when importable.
_FALLBACK_LUNG_VESSELS = {"lung_airways": 1, "lung_airways_wall": 2, "lung_arteries": 3, "lung_veins": 4}
_FALLBACK_EFFUSION = {"lung_pleural": 1, "pleural_effusion": 2, "pericardial_effusion": 3}
_FALLBACK_TRUNK = {"abdominal_cavity": 1, "thoracic_cavity": 2, "pericardium": 3, "mediastinum": 4}


def task_label_ids(task: str, fallback: dict[str, int]) -> dict[str, int]:
    """class name -> label id for one task."""
    try:
        from totalsegmentator.map_to_binary import class_map
        return {n: i for i, n in class_map[task].items()}
    except Exception:
        return dict(fallback)


def get_pp_label_map() -> dict[str, int]:
    """name -> id for vertebrae_pp / vertebrae_pp_refined: 1 = C1 ... 24 = L5.

    These labels are the vertebral BODIES only (no arch, no sacrum), which is
    how a vertebral level is read clinically.
    """
    try:
        from totalsegmentator.map_to_binary import class_map
        return {n: i for i, n in class_map["vertebrae_pp_refined"].items()}
    except Exception:
        return {n: i for i, n in enumerate(VERT_ORDER[:24], start=1)}


# ---------------------------------------------------------------------------
# Segmentation tasks
# ---------------------------------------------------------------------------

# Cohort-fixed inference settings, chosen from the Phase-1 diagnostic measured on
# an Apple M3 Pro and an RTX 4090 (see README).
#
# higher_order_resampling hands the down- and upsampling back to nnU-Net, which
# resamples the network's probability maps onto the CT grid instead of deciding
# on the coarse model grid and snapping labels back. That is the export path
# nnU-Net is built around; TotalSegmentator's default is a memory shortcut that
# systematically thickens thin structures. Measured on a 0.8 mm chest CT: it
# costs 15 s on `total` and 17 s on `lung_vessels` and removes that bias
# (airway wall -6 %, vessels -3 %), while on the other three tasks it costs
# 20-54 s and moves nothing that reaches a column. So it is on only where it
# pays. resampling_order has no effect wherever higher_order_resampling is on.
RESAMPLING_ORDER = 1
ROBUST_CROP = True


@dataclass(frozen=True)
class TaskSpec:
    name: str                    # TotalSegmentator task name
    mandatory: bool = False      # failure aborts the case instead of emptying columns
    uses_roi_subset: bool = False
    higher_order: bool = False   # see the note above


TASK_SPECS: dict[str, TaskSpec] = {
    "total": TaskSpec("total", mandatory=True, uses_roi_subset=True, higher_order=True),
    "vertebrae_pp_refined": TaskSpec("vertebrae_pp_refined", mandatory=True),
    "vertebrae_pp": TaskSpec("vertebrae_pp", mandatory=True),
    "lung_vessels": TaskSpec("lung_vessels", higher_order=True),
    "pleural_pericard_effusion": TaskSpec("pleural_pericard_effusion"),
    "trunk_cavities": TaskSpec("trunk_cavities"),
}
DEFAULT_TASKS = ["total", "vertebrae_pp_refined", "lung_vessels",
                 "pleural_pericard_effusion", "trunk_cavities"]
VERTEBRA_TASKS = ("vertebrae_pp_refined", "vertebrae_pp")


def supports_fast(task: str) -> bool:
    """Only `total` has a low-resolution variant; every other task raises on fast=True."""
    try:
        from totalsegmentator.map_tasks_config import TASK_CONFIGS
        return "sub_modes" in TASK_CONFIGS.get(task, {})
    except Exception:
        return task.startswith("total")


def total_roi_subset(include_vertebrae: bool = False) -> list[str]:
    """Which classes to ask `total` for.

    Lobes and trachea live in the same 1.5 mm sub-model, so the trachea costs no
    extra inference (measured: every other label identical with and without it).
    The vertebrae live in a SECOND sub-model, so asking for them doubles the
    1.5 mm work of this task. The pipeline reads its levels from the vertebral
    bodies of vertebrae_pp_refined, which is both the clinically correct
    definition and a separate model, so the vertebrae here are redundant and are
    left out by default. They can be switched back on with --total-vertebrae,
    which is worth doing when the body model is in doubt (it is the only
    independent check on the levels), and happens automatically when no
    vertebral-body task is in the run.
    """
    lobes, verts = get_label_maps()
    names = list(lobes) + [TRACHEA_NAME]
    if include_vertebrae:
        names += list(verts)
    return names


def totalseg_version() -> str:
    try:
        import importlib.metadata as md
        return md.version("TotalSegmentator")
    except Exception:
        return "unknown"


def masks_dir_for(output: Path, case_id: str) -> Path:
    return output / "masks" / case_id


def mask_files(masks_dir: Path, task: str, fast: bool) -> tuple[Path, Path]:
    """(mask, report) paths. Only `total` gets a separate low-resolution file."""
    stem = f"{task}_fast" if (fast and supports_fast(task)) else task
    return masks_dir / f"{stem}.nii.gz", masks_dir / f"{stem}.report.json"


def check_reusable(mask_path: Path, report_path: Path, ct_path: Path, task: str,
                   was_fast: bool, roi_subset: list[str] | None) -> tuple[bool, list[str]]:
    """Whether a saved mask may be reused, plus any warnings about it.

    A mask counts only if its run report proves it was produced for this task, at
    this resolution, with this roi_subset, and it sits on the saved CT's voxel
    grid. A mask without a report (or a report without a mask) is an interrupted
    run and is always redone.
    """
    if not mask_path.exists() or not report_path.exists():
        return False, []
    try:
        rep = json.loads(report_path.read_text())
    except Exception as e:
        log.warning("  unreadable run report %s (%s) -> re-segmenting", report_path.name, e)
        return False, []
    if rep.get("task") != task or bool(rep.get("fast")) != was_fast:
        return False, []
    # A mask that carries MORE classes than we asked for is still usable; one
    # that is missing any of them is not. This is what lets --total-vertebrae be
    # switched off without invalidating masks that already have them.
    if not set(roi_subset or []).issubset(set(rep.get("roi_subset") or [])):
        return False, []
    try:
        verify_same_grid(mask_path, ct_path)
    except GeometryError as e:
        log.warning("  %s -> re-segmenting", e)
        return False, []
    notes: list[str] = []
    ver, cur = rep.get("totalsegmentator_version"), totalseg_version()
    if ver and cur != "unknown" and ver != cur:
        notes.append(f"{task} mask from TotalSegmentator {ver} (current {cur})")
    pipe = rep.get("pipeline") or {}
    if pipe:
        want_ho = TASK_SPECS[task].higher_order
        if bool(pipe.get("higher_order_resampling")) != want_ho or pipe.get("resampling_order") != RESAMPLING_ORDER:
            notes.append(f"{task} mask made with different resampling settings")
    return True, notes


def find_reusable(masks_dir: Path, spec: TaskSpec, ct_path: Path, fast: bool,
                  roi_subset: list[str] | None) -> tuple[Path | None, list[str]]:
    """An existing mask of equal or better resolution, or None.

    In fast mode a full-resolution mask is better than a fast one, so it wins and
    nothing is recomputed. In full mode a fast mask is never accepted.
    """
    candidates: list[tuple[str, bool]] = [(spec.name, False)]
    if fast and supports_fast(spec.name):
        candidates.append((f"{spec.name}_fast", True))
    for stem, was_fast in candidates:
        mp, rp = masks_dir / f"{stem}.nii.gz", masks_dir / f"{stem}.report.json"
        ok, notes = check_reusable(mp, rp, ct_path, spec.name, was_fast, roi_subset)
        if ok:
            if fast and not was_fast:
                notes.append(f"{spec.name}: full-resolution mask reused")
            return mp, notes
    return None, []


# ---------------------------------------------------------------------------
# DICOM series handling
# ---------------------------------------------------------------------------

@dataclass
class Series:
    uid: str
    files: list[Path] = field(default_factory=list)
    n_slices: int = 0  # frames, not files: enhanced DICOM packs a series into one file
    modality: str = ""
    description: str = ""
    is_axial: bool | None = None  # None = unknown (no orientation tag)
    first_ds: pydicom.Dataset | None = None
    patient_id: str = ""
    study_uid: str = ""
    study_date: str = ""
    # Filled by sort_series_files: how evenly the slices are spaced along the
    # scan axis. None when the positions could not be read for every file.
    gap_median_mm: float | None = None
    gap_max_dev_mm: float | None = None
    gap_min_mm: float | None = None


def _iop_is_axial(ds: pydicom.Dataset) -> bool | None:
    iop = getattr(ds, "ImageOrientationPatient", None)
    if iop is None or len(iop) != 6:
        return None
    row, col = np.array(iop[:3], float), np.array(iop[3:], float)
    normal = np.cross(row, col)
    return bool(abs(normal[2]) > 0.9)


def scan_series(patient_dir: Path) -> dict[str, Series]:
    """Group all readable DICOM files under patient_dir by SeriesInstanceUID."""
    series: dict[str, Series] = {}
    for f in sorted(patient_dir.rglob("*")):
        if not f.is_file() or f.name.startswith(".") or f.name.upper() == "DICOMDIR":
            continue
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
        except Exception:
            continue  # not a DICOM file
        uid = str(getattr(ds, "SeriesInstanceUID", "") or "")
        if not uid or "Rows" not in ds:  # skip non-image objects (SR, presentation states...)
            continue
        s = series.setdefault(uid, Series(uid=uid))
        s.files.append(f)
        s.n_slices += int(getattr(ds, "NumberOfFrames", 0) or 0) or 1
        if s.first_ds is None:
            s.first_ds = ds
            s.modality = str(getattr(ds, "Modality", "") or "")
            s.description = str(getattr(ds, "SeriesDescription", "") or "")
            s.is_axial = _iop_is_axial(ds)
            s.patient_id = str(getattr(ds, "PatientID", "") or "").strip()
            s.study_uid = str(getattr(ds, "StudyInstanceUID", "") or "")
            s.study_date = str(getattr(ds, "StudyDate", "") or "").strip()
    return series


def select_series(source: Path | dict[str, Series], min_slices: int) -> Series:
    """Pick the axial CT series with the most slices; ignore series < min_slices.

    `source` is either a patient folder to scan, or an already-scanned
    {uid: Series} pool (flat export mode).
    """
    series = scan_series(source) if isinstance(source, Path) else source
    if not series:
        raise RuntimeError("no readable DICOM files found")
    if isinstance(source, Path):
        # A folder that holds several PatientIDs is an export, not one patient.
        # Picking the largest series out of it would process one patient and
        # silently drop the rest, so refuse instead.
        pids = sorted({s.patient_id for s in series.values() if s.patient_id})
        if len(pids) > 1:
            raise RuntimeError(
                f"this folder holds {len(pids)} different PatientIDs across {len(series)} series, "
                f"so it is a multi-patient export rather than one patient. Rerun with --flat: "
                f"cases are then grouped by the PatientID stored inside the files, however the "
                f"folders are nested.")
    candidates = [s for s in series.values() if s.n_slices >= min_slices]
    if not candidates:
        raise RuntimeError(
            f"no series with >= {min_slices} slices "
            f"(found {len(series)} series, max {max(s.n_slices for s in series.values())} slices)"
        )
    axial = [s for s in candidates if s.is_axial is not False and s.modality in ("CT", "")]
    pool = axial or candidates
    if not axial:
        log.warning("  no axial CT series among candidates; falling back to largest series")
    chosen = max(pool, key=lambda s: s.n_slices)
    if chosen.first_ds is None and chosen.files:
        # A case restored from the flat index carries no header yet: read the one
        # file we actually need, rather than all of them during the scan.
        try:
            chosen.first_ds = pydicom.dcmread(chosen.files[0], stop_before_pixels=True)
        except Exception as e:
            log.warning("  could not read the header of %s (%s); the identity and acquisition "
                        "columns may stay empty", chosen.files[0].name, e)
    log.info(
        "  series: '%s' (%s, %d slices in %d files, %d series in folder, uid ...%s)",
        chosen.description or "<no description>", chosen.modality or "?",
        chosen.n_slices, len(chosen.files), len(series), chosen.uid[-8:],
    )
    return chosen


def sort_series_files(s: Series) -> list[Path]:
    """Sort slices along the slice normal using ImagePositionPatient."""
    iop = getattr(s.first_ds, "ImageOrientationPatient", None)
    normal = (
        np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
        if iop is not None and len(iop) == 6 else np.array([0.0, 0.0, 1.0])
    )

    proj: dict[Path, float] = {}

    def key(f: Path) -> float:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True, specific_tags=["ImagePositionPatient", "InstanceNumber"])
            ipp = getattr(ds, "ImagePositionPatient", None)
            if ipp is not None and len(ipp) == 3:
                v = float(np.dot(np.array(ipp, float), normal))
                proj[f] = v
                return v
            return float(getattr(ds, "InstanceNumber", 0) or 0)
        except Exception:
            return 0.0

    files = sorted(s.files, key=key)
    # The slice positions were just read for the sort, so measuring how evenly
    # the slices are spaced is free. It matters: the volume is built with one
    # uniform spacing, so an uneven or gappy series is reconstructed slightly
    # stretched or squashed, and every craniocaudal length comes off that grid.
    if len(files) > 2 and len(proj) == len(files):
        gaps = np.diff([proj[f] for f in files])
        med = float(np.median(gaps))
        if med > 1e-6:
            s.gap_median_mm = med
            s.gap_max_dev_mm = float(np.max(np.abs(gaps - med)))
            s.gap_min_mm = float(np.min(gaps))
    return files


def slice_spacing_notes(s: Series) -> list[str]:
    """Warnings about how evenly this series is sampled along the scan axis.

    The volume is reconstructed with a single spacing taken from the first and
    last slice positions, so the two ends are always right, but a gap or a
    duplicate in between displaces the slices around it. SimpleITK prints its
    own "Non uniform sampling" warning for the same thing; this reports it in
    millimetres and puts it in the case's `status`, where it can be filtered.
    """
    med, dev, lo = s.gap_median_mm, s.gap_max_dev_mm, s.gap_min_mm
    if not med or dev is None:
        return []
    notes: list[str] = []
    if dev > 0.25 * med:
        missing = " (a slice looks missing)" if dev > 0.75 * med else ""
        notes.append(f"uneven slice spacing: median {med:.2f} mm, worst gap off by "
                     f"{dev:.2f} mm{missing}")
    if lo is not None and lo < 0.25 * med:
        notes.append(f"overlapping or duplicated slices (smallest gap {lo:.2f} mm "
                     f"against a median of {med:.2f} mm)")
    return notes


def read_identity(s: Series, folder_id: str) -> dict:
    ds = s.first_ds
    pid = str(getattr(ds, "PatientID", "") or "").strip()
    acc = str(getattr(ds, "AccessionNumber", "") or "").strip()
    date = str(getattr(ds, "StudyDate", "") or "").strip()
    return {
        "dicom_patient_id": pid,
        "accession_number": acc,
        "study_date": date,
        "id_mismatch": folder_id.strip() != pid,  # missing PatientID also counts as mismatch
    }


FLAT_INDEX_VERSION = 1


def flat_index_path(output: Path) -> Path:
    return output / "flat_index.json"


def save_flat_index(path: Path, input_root: Path, cases: list[tuple[str, dict[str, Series]]]) -> None:
    """Remember which files belong to which case.

    Grouping a flat export means opening the header of every file, which on an
    external drive is minutes of random reads. The result only changes when the
    data does, so it is written once and reused.
    """
    payload = {
        "version": FLAT_INDEX_VERSION,
        "input_root": str(input_root.resolve()),
        "scanned_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cases": [
            {"case_id": cid,
             "series": [{"uid": s.uid, "files": [str(f) for f in s.files], "n_slices": s.n_slices,
                         "modality": s.modality, "description": s.description,
                         "is_axial": s.is_axial, "patient_id": s.patient_id,
                         "study_uid": s.study_uid, "study_date": s.study_date}
                        for s in pool.values()]}
            for cid, pool in cases
        ],
    }
    tmp = path.with_name("_partial_" + path.name)
    tmp.write_text(json.dumps(payload))
    _replace_retry(tmp, path)


def load_flat_index(path: Path, input_root: Path) -> list[tuple[str, dict[str, Series]]] | None:
    """The saved grouping, or None if there is none that still fits this input."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        log.warning("flat index unreadable (%s) -> rescanning", e)
        return None
    if data.get("version") != FLAT_INDEX_VERSION:
        log.info("flat index was written by an older version -> rescanning")
        return None
    if data.get("input_root") != str(input_root.resolve()):
        log.info("flat index was built for a different input folder -> rescanning")
        return None
    cases: list[tuple[str, dict[str, Series]]] = []
    for c in data.get("cases", []):
        pool = {sd["uid"]: Series(uid=sd["uid"], files=[Path(f) for f in sd["files"]],
                                  n_slices=sd["n_slices"], modality=sd["modality"],
                                  description=sd["description"], is_axial=sd["is_axial"],
                                  patient_id=sd["patient_id"], study_uid=sd["study_uid"],
                                  study_date=sd["study_date"])
                for sd in c["series"]}
        cases.append((c["case_id"], pool))
    if not cases:
        return None
    # Cheap sanity check: if the data moved, one spot check catches it.
    probe = next(iter(cases[0][1].values()))
    if probe.files and not probe.files[0].exists():
        log.warning("flat index points at files that are no longer there -> rescanning")
        return None
    log.info("flat index reused: %d cases from %s, scanned %s (--rescan after adding data)",
             len(cases), path, data.get("scanned_utc", "?"), extra=SHOW)
    return cases


def find_dicomdirs(root: Path, max_depth: int = 3) -> list[Path]:
    """DICOMDIR index files at the root or a few levels below it.

    A DICOMDIR marks a PACS or CD export, which usually holds many patients. The
    search is breadth-first and stops at the first level that has any, so it
    stays fast on a large tree.
    """
    level = [root]
    for _ in range(max_depth + 1):
        found: list[Path] = []
        nxt: list[Path] = []
        for d in level:
            try:
                for child in d.iterdir():
                    if child.name.upper() == "DICOMDIR" and child.is_file():
                        found.append(child)
                    elif child.is_dir() and not child.name.startswith("."):
                        nxt.append(child)
            except OSError:
                continue
        if found:
            return found
        if not nxt:
            break
        level = nxt
    return []


def build_flat_cases(input_root: Path) -> list[tuple[str, dict[str, Series]]]:
    """Flat export mode (DICOMDIR + IMAGES): one case per PatientID + study.

    Scans every DICOM file under the root once and groups the series by the
    PatientID and StudyInstanceUID stored inside the files -- the folder
    structure is ignored entirely. Case ids are the PatientID (a second study
    of the same patient becomes '<PatientID>_study2', ordered by StudyDate).
    """
    log.info("flat export mode: reading headers of every file under %s (may take a while)...", input_root)
    all_series = scan_series(input_root)
    groups: dict[tuple[str, str], dict[str, Series]] = {}
    for s in all_series.values():
        groups.setdefault((s.patient_id, s.study_uid), {})[s.uid] = s

    by_patient: dict[str, list[dict[str, Series]]] = {}
    for (pid, _study), pool in sorted(
        groups.items(),
        key=lambda kv: (kv[0][0], next(iter(kv[1].values())).study_date, kv[0][1]),
    ):
        by_patient.setdefault(pid, []).append(pool)

    cases: list[tuple[str, dict[str, Series]]] = []
    for pid, pools in sorted(by_patient.items()):
        base = re.sub(r"[^\w.-]+", "_", pid) or "unknown_id"
        for j, pool in enumerate(pools, start=1):
            cases.append((base if j == 1 else f"{base}_study{j}", pool))
    log.info("flat export mode: %d DICOM series -> %d patients, %d cases (patient+study)",
             len(all_series), len(by_patient), len(cases))
    return cases


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

class GeometryError(RuntimeError):
    """A volume's world geometry contradicts the DICOM slice positions."""


def _read_ipp_iop(f: Path):
    ds = pydicom.dcmread(f, stop_before_pixels=True,
                         specific_tags=["ImagePositionPatient", "ImageOrientationPatient"])
    ipp = getattr(ds, "ImagePositionPatient", None)
    iop = getattr(ds, "ImageOrientationPatient", None)
    ipp = np.array(ipp, float) if ipp is not None and len(ipp) == 3 else None
    iop = np.array(iop, float) if iop is not None and len(iop) == 6 else None
    return ipp, iop


def _fix_geometry(img, files_sorted: list[Path]) -> None:
    """Overwrite the reader's inferred geometry with one computed from the sorted
    slice positions themselves.

    SimpleITK/GDCM trusts vendor tags for the slice axis -- GE writes a NEGATIVE
    SpacingBetweenSlices (0018,0088), which flips the claimed z direction while
    the pixel slices stay stacked in the order we supplied, silently producing
    an upside-down volume. Slice positions (IPP) are the ground truth, so the
    origin, z-spacing, and direction are recomputed from them.
    """
    if len(files_sorted) < 2:
        return  # single-file (multiframe) input: nothing to recompute from
    ipp0, iop = _read_ipp_iop(files_sorted[0])
    ippN, _ = _read_ipp_iop(files_sorted[-1])
    if ipp0 is None or ippN is None or iop is None:
        return  # tags missing: keep reader's guess; verify_geometry will judge it
    step = (ippN - ipp0) / (len(files_sorted) - 1)
    z_spacing = float(np.linalg.norm(step))
    if z_spacing < 1e-6:
        raise GeometryError("first and last slice share the same position")
    normal = step / z_spacing
    row, col = iop[:3], iop[3:]
    if abs(float(np.dot(normal, np.cross(row, col)))) < 0.99:
        raise GeometryError("slice positions are not perpendicular to the image plane "
                            "(gantry-tilted or non-orthogonal series)")
    sx, sy = img.GetSpacing()[0], img.GetSpacing()[1]
    img.SetSpacing((sx, sy, z_spacing))
    img.SetDirection((row[0], col[0], normal[0],
                      row[1], col[1], normal[1],
                      row[2], col[2], normal[2]))
    img.SetOrigin(tuple(ipp0))


def convert_series_to_nifti(files_sorted: list[Path], out_path: Path) -> None:
    """Convert the DICOM series to a NIfTI with SimpleITK (atomic write),
    with the geometry recomputed from the slice positions (see _fix_geometry)."""
    import SimpleITK as sitk

    if not VERBOSE:
        # ITK warns on the console about non-uniform sampling. slice_spacing_notes
        # measures the same thing in millimetres and puts it in the CSV, where a
        # batch run can actually act on it.
        sitk.ProcessObject_GlobalWarningDisplayOff()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name("_partial_" + out_path.name)
    try:
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames([str(f) for f in files_sorted])
        img = reader.Execute()
        _fix_geometry(img, files_sorted)
        sitk.WriteImage(img, str(tmp))
        os.replace(tmp, out_path)
    finally:
        tmp.unlink(missing_ok=True)


def verify_geometry(nii_path: Path, files_sorted: list[Path], strict: bool = True) -> None:
    """Check a saved volume's world geometry against the DICOM slice positions.

    strict: voxel (0,0,0) and (0,0,nz-1) must map exactly onto the first/last
    sorted slice positions (our own conversions never reorder voxels).
    non-strict (externally converted volumes, which may legally reorder axes):
    only require the volume's claimed z endpoints to match the true slice z
    endpoints -- still catches a z-flip, whose claimed range runs away from the
    true positions. Raises GeometryError on mismatch; a silent flip can never
    reach segmentation or measurement.
    """
    import nibabel as nib

    if len(files_sorted) < 2:
        log.warning("  geometry not verifiable (single-file series)")
        return
    first_ipp, _ = _read_ipp_iop(files_sorted[0])
    last_ipp, _ = _read_ipp_iop(files_sorted[-1])
    if first_ipp is None or last_ipp is None:
        log.warning("  geometry not verifiable (missing ImagePositionPatient)")
        return
    # DICOM positions are LPS; NIfTI affines are RAS: flip x and y.
    first = first_ipp * np.array([-1.0, -1.0, 1.0])
    last = last_ipp * np.array([-1.0, -1.0, 1.0])

    img = nib.load(nii_path)
    nz = img.shape[2]
    if strict and nz != len(files_sorted):
        raise GeometryError(f"{nii_path.name} has {nz} slices but the series has "
                            f"{len(files_sorted)} -- stale or wrong file")
    aff = img.affine
    v0 = (aff @ [0, 0, 0, 1])[:3]
    vN = (aff @ [0, 0, nz - 1, 1])[:3]
    if strict:
        ok = (np.linalg.norm(v0 - first) < 2.0 and np.linalg.norm(vN - last) < 2.0)
    else:
        z_claim, z_true = sorted((v0[2], vN[2])), sorted((first[2], last[2]))
        ok = abs(z_claim[0] - z_true[0]) < 2.0 and abs(z_claim[1] - z_true[1]) < 2.0
    if not ok:
        raise GeometryError(
            f"geometry check FAILED for {nii_path.name}: volume claims slices at "
            f"z {v0[2]:.1f}..{vN[2]:.1f} mm but the DICOMs are at {first[2]:.1f}..{last[2]:.1f} mm "
            f"(flipped volume; if this is a previously saved file, run --recheck-geometry)")


def verify_same_grid(seg_path: Path, ct_path: Path) -> None:
    import nibabel as nib
    a, b = nib.load(seg_path), nib.load(ct_path)
    if a.shape[:3] != b.shape[:3] or not np.allclose(a.affine, b.affine, atol=0.01):
        raise GeometryError(f"{seg_path.name} and the saved CT are on different grids "
                            "-- run --recheck-geometry")


def recheck_geometry(files_sorted: list[Path], ct_path: Path) -> bool:
    """Recovery pass: re-convert the series with the fixed converter and compare.
    Replaces the saved CT only if its geometry actually changed. Returns True
    iff it changed (caller then discards the segmentation of that case)."""
    import nibabel as nib

    tmp = ct_path.with_name("_recheck_" + ct_path.name)
    try:
        convert_series_to_nifti(files_sorted, tmp)
        if ct_path.exists():
            old, new = nib.load(ct_path), nib.load(tmp)
            if old.shape == new.shape and np.allclose(old.affine, new.affine, atol=0.01):
                return False
            log.info("  geometry differs: axcodes %s -> %s",
                     "".join(nib.aff2axcodes(old.affine)), "".join(nib.aff2axcodes(new.affine)))
        os.replace(tmp, ct_path)
        return True
    finally:
        tmp.unlink(missing_ok=True)


def _free_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def _replace_retry(src: Path, dst: Path, attempts: int = 3) -> None:
    """os.replace, retried: on Windows a virus scanner can hold a freshly
    written file open for a moment."""
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except OSError:
            if i == attempts - 1:
                raise
            time.sleep(1.0)


def _link_or_copy(src: Path, dst: Path) -> None:
    """Symlink if allowed (Windows needs Developer Mode), otherwise copy."""
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def device_arg(value: str) -> str:
    if value in ("auto", "cpu", "mps", "gpu") or re.fullmatch(r"gpu:\d+", value):
        return value
    raise argparse.ArgumentTypeError("device must be auto, cpu, mps, gpu or gpu:N")


def resolve_device(arg: str) -> str:
    """auto -> CUDA, else Apple MPS, else CPU. An explicit device that is not
    available is an error: TotalSegmentator would silently fall back to the CPU
    and a batch would crawl for days without saying why."""
    import torch

    cuda = torch.cuda.is_available()
    mps = torch.backends.mps.is_available()
    if arg == "auto":
        chosen = "gpu" if cuda else ("mps" if mps else "cpu")
        log.debug("device auto -> %s (CUDA available %s, MPS available %s)", chosen, cuda, mps)
        return chosen
    if arg.startswith("gpu") and not cuda:
        raise SystemExit("error: --device gpu was requested but CUDA is not available")
    if arg == "mps" and not mps:
        raise SystemExit("error: --device mps was requested but MPS is not available")
    return arg


def disable_usage_stats() -> None:
    """Turn off TotalSegmentator's anonymous usage reporting. Also removes a 5 s
    stall per model call on a machine with no internet access."""
    try:
        from totalsegmentator.config import get_totalseg_dir, setup_totalseg
        setup_totalseg()
        cfg_path = Path(get_totalseg_dir()) / "config.json"
        cfg = json.loads(cfg_path.read_text())
        if cfg.get("send_usage_stats", False):
            cfg["send_usage_stats"] = False
            cfg_path.write_text(json.dumps(cfg, indent=4))
            log.info("TotalSegmentator usage statistics disabled in %s", cfg_path)
    except Exception as e:
        log.warning("could not disable TotalSegmentator usage statistics (%s)", e)


def pipeline_git_sha() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def ensure_ct(files_sorted: list[Path], ct_path: Path):
    """Make sure output/ct/<id>.nii.gz exists and agrees with the DICOM slice
    positions, and return what TotalSegmentator should be fed.

    Returns (input path, ct_saved, tempdir handle). If the conversion fails we
    feed the DICOMs directly; no CT is saved then, so the density columns stay
    empty and the case is flagged.
    """
    try:
        if not ct_path.exists():
            convert_series_to_nifti(files_sorted, ct_path)
    except GeometryError:
        raise  # broken geometry aborts the case; never fall back around it
    except Exception as e:
        log.warning("  NIfTI conversion failed (%s); feeding TotalSegmentator the DICOMs "
                    "directly (no CT saved, density columns stay empty)", e)
        td = tempfile.TemporaryDirectory(prefix="ts_dicom_")
        link_dir = Path(td.name) / "dicom"
        link_dir.mkdir()
        for i, f in enumerate(files_sorted):
            _link_or_copy(f.resolve(), link_dir / f"{i:05d}.dcm")
        return link_dir, False, td
    verify_geometry(ct_path, files_sorted)
    return ct_path, True, None


def run_task(ct_input: Path, spec: TaskSpec, mask_path: Path, report_path: Path, *,
             device: str, fast: bool, roi_subset: list[str] | None, nr_thr_saving: int,
             force_split: bool, ct_name: str) -> list[str]:
    """Run one TotalSegmentator task and save the mask plus its run report.

    Both files are written under a temporary name and renamed only on success,
    the mask first: an interrupted run can then never leave a mask the resume
    logic would trust. On Apple MPS a failure is retried, first split into three
    chunks (which only helps `total`), then on the CPU.
    """
    from totalsegmentator.python_api import totalsegmentator

    eff_fast = fast and supports_fast(spec.name)
    tmp_mask = mask_path.with_name("_partial_" + mask_path.name)
    tmp_report = report_path.with_name("_partial_" + report_path.name)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    attempts: list[tuple[str, bool]] = [(device, force_split)]
    if device == "mps":
        if spec.name == "total" and not force_split:
            attempts.append(("mps", True))
        attempts.append(("cpu", force_split))

    used_dev, used_split, last_error = device, force_split, None
    try:
        for i, (dev, split) in enumerate(attempts):
            tmp_mask.unlink(missing_ok=True)
            tmp_report.unlink(missing_ok=True)
            try:
                with captured_output(spec.name):
                    totalsegmentator(
                        input=ct_input, output=tmp_mask, task=spec.name, ml=True, quiet=True,
                        fast=eff_fast, roi_subset=roi_subset if spec.uses_roi_subset else None,
                        device=dev, robust_crop=ROBUST_CROP, report=str(tmp_report),
                        resampling_order=RESAMPLING_ORDER,
                        higher_order_resampling=spec.higher_order,
                        nr_thr_saving=nr_thr_saving, force_split=split,
                    )
                if not tmp_mask.exists():
                    raise RuntimeError("TotalSegmentator finished but produced no output file")
                used_dev, used_split = dev, split
                if dev != device:
                    notes.append(f"{spec.name} ran on {dev} ({device} failed: {str(last_error)[:80]})")
                elif split != force_split:
                    notes.append(f"{spec.name} needed splitting into 3 chunks on {dev}")
                break
            except Exception as e:
                last_error = e
                if i == len(attempts) - 1:
                    raise
                log.warning("  %s failed on %s (%s); retrying", spec.name, dev, str(e)[:160])
                _free_memory()

        try:
            rep = json.loads(tmp_report.read_text()) if tmp_report.exists() else {}
        except Exception:
            rep = {}
        rep["pipeline"] = {
            "resampling_order": RESAMPLING_ORDER,
            "higher_order_resampling": spec.higher_order,
            "robust_crop": ROBUST_CROP,
            "effective_fast": eff_fast,
            "device_requested": device,
            "device_used": used_dev,
            "retried_on_cpu": used_dev != device,
            "force_split": used_split,
            "nr_thr_saving": nr_thr_saving,
            "ct_file": ct_name,
            "pipeline_git_sha": pipeline_git_sha(),
        }
        tmp_report.write_text(json.dumps(rep, indent=2))
        _replace_retry(tmp_mask, mask_path)
        _replace_retry(tmp_report, report_path)
    finally:
        tmp_mask.unlink(missing_ok=True)
        tmp_report.unlink(missing_ok=True)
    return notes


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

def _largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return labeled == int(np.argmax(sizes))


def _z_extremes(mask: np.ndarray) -> tuple[int, int] | None:
    """(z_min, z_max) over axial slices with >= MIN_LUNG_VOXELS_PER_SLICE lung voxels."""
    counts = mask.sum(axis=(0, 1))
    valid = np.flatnonzero(counts >= MIN_LUNG_VOXELS_PER_SLICE)
    if valid.size == 0:
        return None
    return int(valid[0]), int(valid[-1])


def _vert_level(per_slice_verts: list[list[str]], z_start: int, step: int, pick_cranial: bool) -> str | None:
    """Vertebra overlapping slice z_start; walk in `step` direction until one is found."""
    nz = len(per_slice_verts)
    z = z_start
    while 0 <= z < nz:
        names = per_slice_verts[z]
        if names:
            return min(names, key=vertebra_rank) if pick_cranial else max(names, key=vertebra_rank)
        z += step
    return None


@dataclass
class VertStat:
    centroid: np.ndarray  # (x, y, z) in mm
    n_vox: int
    z_lo: int  # first / last axial slice the vertebra appears on
    z_hi: int


def vertebra_stats(data: np.ndarray, verts: dict[str, int], spacing: np.ndarray) -> dict[str, VertStat]:
    """name -> VertStat for each vertebra with >= 100 voxels."""
    counts = np.bincount(data.ravel())
    bboxes = ndimage.find_objects(data)
    id_to_name = {i: n for n, i in verts.items()}
    ids = [i for i in id_to_name
           if i < counts.size and counts[i] >= 100 and i <= len(bboxes) and bboxes[i - 1] is not None]
    if not ids:
        return {}
    cents = ndimage.center_of_mass(data > 0, labels=data, index=ids)
    return {
        id_to_name[i]: VertStat(np.array(c) * spacing, int(counts[i]),
                                bboxes[i - 1][2].start, bboxes[i - 1][2].stop - 1)
        for i, c in zip(ids, cents)
    }


def cobb_ferguson_angle(stats: dict[str, VertStat], nz: int) -> float | None:
    """Ferguson-style coronal curvature angle in degrees (0 = straight spine).

    Centroids are projected onto the CORONAL plane (left-right vs craniocaudal;
    the AP coordinate is dropped, so pure kyphosis/lordosis measures 0).
    Vertebrae cut by the field of view (touching the first/last slice) or
    abnormally small (< 50% of the median voxel count) are excluded as their
    centroids are unreliable. The angle is between the lines top->apex and
    apex->bottom, the apex being the centroid farthest laterally from the
    top-bottom line. Long lever arms make this robust to per-vertebra centroid
    noise, unlike consecutive-segment tangents.
    """
    usable = {n: s for n, s in stats.items() if s.z_lo > 0 and s.z_hi < nz - 1}
    if usable:
        median_vox = float(np.median([s.n_vox for s in usable.values()]))
        usable = {n: s for n, s in usable.items() if s.n_vox >= 0.5 * median_vox}
    if len(usable) < 5:  # too few complete vertebrae for a meaningful curve
        return None
    pts = np.array(sorted(((s.centroid[0], s.centroid[2]) for s in usable.values()),
                          key=lambda p: p[1]))  # (x, z) in mm, caudal -> cranial
    bottom, top = pts[0], pts[-1]
    axis = top - bottom
    if np.linalg.norm(axis) == 0:
        return None
    rel = pts[1:-1] - bottom
    lateral = rel[:, 0] * axis[1] - rel[:, 1] * axis[0]  # signed lateral offset from the line
    apex = pts[1 + int(np.argmax(np.abs(lateral)))]
    v1, v2 = apex - top, bottom - apex
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return None
    cos_ang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return round(float(np.degrees(np.arccos(cos_ang))), 2)


def vertebra_sanity_flags(data: np.ndarray, verts: dict[str, int],
                          centroids: dict[str, np.ndarray]) -> list[str]:
    """Internal-consistency checks on the vertebra labeling.

    Catches broken, duplicated, or skipped labelings. A labeling that is
    uniformly shifted by one level (transitional anatomy) is internally
    consistent and canNOT be caught here -- that needs visual review.
    """
    if len(centroids) < 3:
        return []
    flags: list[str] = []

    # 1. Sequence contiguity: missing ends are normal (field of view), missing
    #    middles are not.
    ranks = sorted(vertebra_rank(n) for n in centroids)
    missing = [VERT_ORDER[r] for r in range(ranks[0], ranks[-1]) if r not in ranks]
    for name in missing:
        flags.append(f"missing {short_name(name)} in sequence")

    # 2. Spatial order: the physically higher vertebra must carry the more
    #    cranial name (z decreases toward the feet in RAS).
    by_height = sorted(centroids, key=lambda n: -centroids[n][2])
    for above, below in zip(by_height, by_height[1:]):
        if vertebra_rank(above) > vertebra_rank(below):
            flags.append(f"order inverted: {short_name(above)} above {short_name(below)}")

    # 3. Split labels: one name painted on two separate bones.
    bboxes = ndimage.find_objects(data)
    for name in centroids:
        idx = verts[name]
        if idx > len(bboxes) or bboxes[idx - 1] is None:
            continue
        labeled, n = ndimage.label(data[bboxes[idx - 1]] == idx)
        if n > 1:
            sizes = np.sort(np.bincount(labeled.ravel())[1:])
            if sizes[-2] >= 0.3 * sizes.sum():
                flags.append(f"{short_name(name)} split in {n} components")

    # 4. Spacing plausibility: a ~2x gap between name-consecutive vertebrae
    #    means a physical vertebra in between got no label.
    by_rank = sorted(centroids, key=vertebra_rank)
    gaps = [(a, b, centroids[a][2] - centroids[b][2])
            for a, b in zip(by_rank, by_rank[1:])
            if vertebra_rank(b) - vertebra_rank(a) == 1]
    if len(gaps) >= 4:
        median_gap = float(np.median([g for _, _, g in gaps]))
        for a, b, g in gaps:
            if g > 1.7 * median_gap:
                flags.append(f"suspicious gap between {short_name(a)} and {short_name(b)}")
    return flags


class _NoCrossCheck(Exception):
    """The total mask holds no vertebrae, so there is nothing to cross-check against."""


def get_trachea_id() -> int:
    try:
        from totalsegmentator.map_to_binary import class_map
        return {n: i for i, n in class_map["total"].items()}[TRACHEA_NAME]
    except Exception:
        return 16


def _load_mask(path: Path, ref) -> np.ndarray:
    """Load a mask, reorient it exactly like the reference, and insist it is on
    the same voxel grid (all tasks return the input grid, so this must hold)."""
    import nibabel as nib
    m = nib.as_closest_canonical(nib.load(path))
    if m.shape[:3] != ref.shape[:3] or not np.allclose(m.affine, ref.affine, atol=0.01):
        raise GeometryError(f"{path.name} is not on the same grid as the total mask")
    return np.asanyarray(m.dataobj).astype(np.uint8)


def _load_ct(ct_path: Path, ref) -> np.ndarray:
    import nibabel as nib
    c = nib.as_closest_canonical(nib.load(ct_path))
    if c.shape[:3] != ref.shape[:3] or not np.allclose(c.affine, ref.affine, atol=0.01):
        raise GeometryError("the saved CT is not on the same grid as the masks")
    sl, inter = float(c.header["scl_slope"]), float(c.header["scl_inter"])
    if (np.isnan(sl) or sl == 1.0) and (np.isnan(inter) or inter == 0.0):
        return np.asanyarray(c.dataobj)
    return c.get_fdata(dtype=np.float32)


def measure_case(ct_path: Path | None, mask_paths: dict[str, Path],
                 series: Series | None = None) -> tuple[dict, list[str]]:
    """Compute every column from the saved masks (and the CT, for density).

    The lung block and the vertebra block are the original measurements,
    unchanged; only the source of the vertebra labels moved, from the whole
    vertebrae in `total` to the vertebral BODIES in vertebrae_pp_refined, which
    is how a level is read clinically. Everything added afterwards sits in its
    own try/except, so a failure in a new metric can never cost the original
    columns.
    """
    import nibabel as nib

    lobes, verts_total = get_label_maps()
    img = nib.as_closest_canonical(nib.load(mask_paths["total"]))  # RAS: axis 2 = inferior -> superior
    data = np.asanyarray(img.dataobj).astype(np.int16)
    z_spacing = float(img.header.get_zooms()[2])
    nz = data.shape[2]
    log.info("  mask %s, spacing %s mm", data.shape, tuple(round(float(v), 3) for v in img.header.get_zooms()[:3]))

    warnings: list[str] = []

    # Vertebra source: the vertebral-body task when it is available.
    vert_task = next((t for t in VERTEBRA_TASKS if t in mask_paths), None)
    if vert_task is not None:
        vdata = _load_mask(mask_paths[vert_task], img).astype(np.int16)
        verts = get_pp_label_map()
    else:
        vdata, verts = data, verts_total
        warnings.append("levels from total (no vertebral-body mask)")

    # Per-slice vertebra presence (computed once).
    id_to_vert = {i: n for n, i in verts.items()}
    per_slice_verts: list[list[str]] = []
    for z in range(nz):
        counts = np.bincount(vdata[:, :, z].ravel())
        present = [id_to_vert[i] for i in np.flatnonzero(counts) if i in id_to_vert]
        per_slice_verts.append(present)

    def side_mask(names: list[str]) -> np.ndarray:
        m = np.isin(data, [lobes[n] for n in names])
        return _largest_component(m) if m.any() else m

    results = {c: "" for c in CSV_COLUMNS if c not in IDENTITY_COLUMNS}
    spacing = np.array(img.header.get_zooms()[:3], dtype=float)

    masks = {"R": side_mask(RIGHT_LOBES), "L": side_mask(LEFT_LOBES)}
    masks["both"] = masks["R"] | masks["L"]

    raw: dict[str, dict[str, float]] = {"R": {}, "L": {}, "both": {}}
    extremes: dict[str, tuple[int, int] | None] = {}
    for side in ("R", "L", "both"):
        ext = _z_extremes(masks[side])
        extremes[side] = ext
        if ext is None:
            warnings.append(f"no {side} lung voxels")
            continue
        z_min, z_max = ext
        raw[side]["height"] = (z_max - z_min) * z_spacing
        # Diagonal: straight-line 3D distance between the centroids of the apex
        # and base slices (always >= the perpendicular craniocaudal height).
        apex_c = np.argwhere(masks[side][:, :, z_max]).mean(axis=0)
        base_c = np.argwhere(masks[side][:, :, z_min]).mean(axis=0)
        delta_mm = np.append((apex_c - base_c) * spacing[:2], (z_max - z_min) * spacing[2])
        raw[side]["diag"] = float(np.linalg.norm(delta_mm))
        # Bounding-box extents of the cleaned mask along the two other axes
        # (RAS: x = left-right width, y = anteroposterior depth), and volume.
        xs = np.flatnonzero(masks[side].any(axis=(1, 2)))
        ys = np.flatnonzero(masks[side].any(axis=(0, 2)))
        raw[side]["width"] = (xs[-1] - xs[0]) * spacing[0]
        raw[side]["depth"] = (ys[-1] - ys[0]) * spacing[1]
        results[f"height_{side}_mm"] = round(raw[side]["height"], 2)
        results[f"diag_{side}_mm"] = round(raw[side]["diag"], 2)
        results[f"width_{side}_mm"] = round(raw[side]["width"], 2)
        results[f"depth_{side}_mm"] = round(raw[side]["depth"], 2)
        results[f"vol_{side}_ml"] = round(int(masks[side].sum()) * float(spacing.prod()) / 1000.0, 2)
        # Apex: walk downward (toward feet) if no vertebra on the slice;
        # base: walk upward. Ties: most cranial at apex, most caudal at base.
        start = _vert_level(per_slice_verts, z_max, step=-1, pick_cranial=True)
        end = _vert_level(per_slice_verts, z_min, step=+1, pick_cranial=False)
        if start is None or end is None:
            warnings.append(f"no vertebrae found for {side}")
        results[f"start_{side}"] = short_name(start)
        results[f"end_{side}"] = short_name(end)

    # R + L sums (from unrounded values) -- the "total lung tissue" variants,
    # as opposed to the *_both union/envelope variants.
    for metric in ("height", "diag", "width", "depth"):
        if metric in raw["R"] and metric in raw["L"]:
            results[f"{metric}_sum_mm"] = round(raw["R"][metric] + raw["L"][metric], 2)

    if not masks["both"].any():
        raise RuntimeError("segmentation contains no lung voxels")

    # Craniocaudal span of the whole segmented vertebral column, and of just
    # the vertebral levels overlapping the lungs (start_both .. end_both).
    vert_z = [z for z, names in enumerate(per_slice_verts) if names]
    if vert_z:
        results["spine_height_mm"] = round((vert_z[-1] - vert_z[0]) * z_spacing, 2)
    if results["start_both"] and results["end_both"]:
        r_lo = vertebra_rank("vertebrae_" + results["start_both"])
        r_hi = vertebra_rank("vertebrae_" + results["end_both"])
        r_lo, r_hi = min(r_lo, r_hi), max(r_lo, r_hi)
        zs = [z for z, names in enumerate(per_slice_verts)
              if any(r_lo <= vertebra_rank(n) <= r_hi for n in names)]
        if zs:
            results["spine_lung_span_mm"] = round((zs[-1] - zs[0]) * z_spacing, 2)

    stats = vertebra_stats(vdata, verts, spacing)
    cobb = cobb_ferguson_angle(stats, nz)
    if cobb is None:
        warnings.append("cobb angle not computable (<5 usable vertebrae)")
    else:
        results["cobb_angle_deg"] = cobb
    centroids = {n: s.centroid for n, s in stats.items()}
    flags = vertebra_sanity_flags(vdata, verts, centroids)
    warnings += [f"vertebra check: {f}" for f in flags]

    # -----------------------------------------------------------------------
    # Columns added by the multi-task extension. Each block is independent: it
    # fills its own columns or explains itself in `status`, and never raises.
    # -----------------------------------------------------------------------
    try:
        import thorax_metrics as tmx
    except Exception as e:  # scikit-image missing: keep the original columns
        warnings.append(f"extra metrics unavailable ({type(e).__name__}); run `uv sync`")
        return results, warnings

    vox_ml = float(spacing.prod()) / 1000.0
    vol_both_raw = float(masks["both"].sum()) * vox_ml

    # DICOM acquisition settings (no mask involved).
    ds = getattr(series, "first_ds", None) if series is not None else None
    if ds is not None:
        try:
            st = getattr(ds, "SliceThickness", None)
            if st is not None:
                results["slice_thickness_mm"] = round(float(st), 2)
            ck = getattr(ds, "ConvolutionKernel", None)
            if ck is not None:
                results["convolution_kernel"] = str(ck) if isinstance(ck, str) else "/".join(str(v) for v in ck)
        except Exception as e:
            warnings.append(f"DICOM acquisition tags failed: {type(e).__name__}: {e}")

    # Lung touching the edge of the scanned volume: a height measured on it is
    # only a lower bound.
    try:
        faces = tmx.edge_contact(masks["both"], MIN_LUNG_VOXELS_PER_SLICE)
        if faces:
            warnings.append(f"lung touches scan edge ({','.join(faces)})")
    except Exception as e:
        warnings.append(f"edge check failed: {type(e).__name__}: {e}")

    # Cross-check: the same levels read off the whole vertebrae in `total`.
    if vert_task is not None:
        try:
            id_to_total = {i: n for n, i in verts_total.items()}
            per_slice_total = []
            for z in range(nz):
                counts = np.bincount(data[:, :, z].ravel())
                per_slice_total.append([id_to_total[i] for i in np.flatnonzero(counts) if i in id_to_total])
            diffs = []
            if not any(per_slice_total):  # vertebrae were not requested from `total`
                raise _NoCrossCheck
            for side in ("R", "L", "both"):
                ext = extremes[side]
                if ext is None:
                    continue
                a = short_name(_vert_level(per_slice_total, ext[1], step=-1, pick_cranial=True))
                b = short_name(_vert_level(per_slice_total, ext[0], step=+1, pick_cranial=False))
                if a != results[f"start_{side}"]:
                    diffs.append(f"start_{side} {results[f'start_{side}']}/{a}")
                if b != results[f"end_{side}"]:
                    diffs.append(f"end_{side} {results[f'end_{side}']}/{b}")
            if diffs:
                warnings.append("level mismatch total vs vertebrae_pp: " + ", ".join(diffs))
        except _NoCrossCheck:
            pass
        except Exception as e:
            warnings.append(f"level cross-check failed: {type(e).__name__}: {e}")

    # Vessels, airways and the carina, all from the lung_vessels mask.
    lv = None
    if "lung_vessels" in mask_paths:
        try:
            lv = _load_mask(mask_paths["lung_vessels"], img)
        except Exception as e:
            warnings.append(f"lung_vessels unreadable: {type(e).__name__}: {e}")
    if lv is not None:
        ids = task_label_ids("lung_vessels", _FALLBACK_LUNG_VESSELS)
        lumen = lv == ids["lung_airways"]
        wall = lv == ids["lung_airways_wall"]
        z_carina = None
        try:
            car = tmx.find_carina(lumen, spacing)
            source = "lumen"
            if not car.found:
                trachea = data == get_trachea_id()
                if trachea.any():
                    car2 = tmx.find_carina(lumen | trachea, spacing)
                    if car2.found:
                        car, source = car2, "lumen+trachea"
            if car.found:
                z_carina = car.z
                if source != "lumen":
                    warnings.append("carina from lumen+trachea")
                results["carina_level"] = short_name(
                    _vert_level(per_slice_verts, z_carina, step=-1, pick_cranial=True))
                if extremes["both"] is not None:
                    z_min, z_max = extremes["both"]
                    results["carina_to_apex_mm"] = round((z_max - z_carina) * z_spacing, 2)
                    results["carina_to_base_mm"] = round((z_carina - z_min) * z_spacing, 2)
            else:
                warnings.append(f"carina not found ({car.reason})")
        except Exception as e:
            warnings.append(f"carina failed: {type(e).__name__}: {e}")

        if z_carina is not None:
            try:
                air = tmx.airway_metrics(lumen, wall, z_carina, spacing)
                lumen_ml, wall_ml = air.lumen_vox * vox_ml, air.wall_vox * vox_ml
                results["airway_lumen_vol_ml"] = round(lumen_ml, 2)
                results["airway_wall_vol_ml"] = round(wall_ml, 2)
                results["airway_branch_count"] = air.branch_count
                if lumen_ml + wall_ml > 0:
                    results["airway_wall_pct"] = round(100.0 * wall_ml / (lumen_ml + wall_ml), 2)
                if vol_both_raw > 0:
                    results["airway_lumen_lung_ratio"] = round(lumen_ml / vol_both_raw, 4)
            except Exception as e:
                warnings.append(f"airway metrics failed: {type(e).__name__}: {e}")

        try:
            lobe_mask = np.isin(data, [lobes[n] for n in LUNG_LOBES])
            ves = tmx.vessel_metrics(lv, lobe_mask, spacing,
                                     artery_label=ids["lung_arteries"], vein_label=ids["lung_veins"])
            a_ml, v_ml = ves.artery_vox * vox_ml, ves.vein_vox * vox_ml
            results["artery_vol_ml"] = round(a_ml, 2)
            results["vein_vol_ml"] = round(v_ml, 2)
            if v_ml > 0:
                results["artery_vein_ratio"] = round(a_ml / v_ml, 4)
            results["small_vessel_vol_ml"] = round(ves.small_vox * vox_ml, 2)
            if z_spacing > 1.5:
                warnings.append(f"small-vessel volume unreliable at {z_spacing:.1f} mm slices")
            del lobe_mask
        except Exception as e:
            warnings.append(f"vessel metrics failed: {type(e).__name__}: {e}")

        # Parenchymal density: lungs minus vessels and airways, eroded 2 mm.
        try:
            if ct_path is None:
                warnings.append("density skipped (no CT saved)")
            else:
                ct = _load_ct(ct_path, img)
                dens = tmx.parenchyma_density(ct, masks["both"], lv > 0, spacing)
                del ct
                if dens.n_vox:
                    results["lung_mean_hu"] = round(dens.mean_hu, 1)
                    results["laa950_pct"] = round(dens.laa_pct, 2)
                    if not (-1000.0 <= dens.mean_hu <= -500.0):
                        warnings.append(f"lung HU implausible ({dens.mean_hu:.0f}); is the CT in Hounsfield units?")
        except Exception as e:
            warnings.append(f"density failed: {type(e).__name__}: {e}")
        del lv, lumen, wall

    # Effusion, split at the per-slice spine midline.
    if "pleural_pericard_effusion" in mask_paths:
        try:
            eff = _load_mask(mask_paths["pleural_pericard_effusion"], img)
            eids = task_label_ids("pleural_pericard_effusion", _FALLBACK_EFFUSION)
            body = (vdata > 0) if vert_task is not None else np.zeros(data.shape, dtype=bool)
            n_r, n_l, note = tmx.split_effusion_lr(eff == eids["pleural_effusion"], body,
                                                   lungs_R=masks["R"], lungs_L=masks["L"])
            results["pleural_eff_R_ml"] = round(n_r * vox_ml, 2)
            results["pleural_eff_L_ml"] = round(n_l * vox_ml, 2)
            results["pericardial_eff_ml"] = round(
                int((eff == eids["pericardial_effusion"]).sum()) * vox_ml, 2)
            if note and (n_r or n_l):
                warnings.append(note)
            del eff
        except Exception as e:
            warnings.append(f"effusion failed: {type(e).__name__}: {e}")

    # Thoracic cavity height.
    if "trunk_cavities" in mask_paths:
        try:
            trunk = _load_mask(mask_paths["trunk_cavities"], img)
            tid = task_label_ids("trunk_cavities", _FALLBACK_TRUNK)["thoracic_cavity"]
            h = tmx.cavity_height(trunk == tid, spacing, MIN_LUNG_VOXELS_PER_SLICE)
            if h is not None:
                results["thoracic_cavity_height_mm"] = round(h, 2)
            del trunk
        except Exception as e:
            warnings.append(f"thoracic cavity failed: {type(e).__name__}: {e}")

    return results, warnings


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------

def check_csv_header(csv_path: Path) -> None:
    """Refuse to append to a results.csv written with a different set of columns.

    Appending would line every value up under the wrong heading, silently. The
    masks are kept either way, so rebuilding the rows costs minutes.
    """
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return
    with open(csv_path, newline="") as fh:
        header = next(csv.reader(fh), [])
    if header == CSV_COLUMNS:
        return
    missing = [c for c in CSV_COLUMNS if c not in header]
    extra = [c for c in header if c not in CSV_COLUMNS]
    raise SystemExit(
        f"error: {csv_path} was written with a different set of columns, so new rows would\n"
        f"not line up with it.\n"
        f"  expected now but missing there: {', '.join(missing) or 'none'}\n"
        f"  present there but gone now:     {', '.join(extra) or 'none'}\n"
        f"Rename or move that file and run again. The masks are kept, so every row is\n"
        f"recomputed in minutes without re-segmenting anything."
    )


def append_row(csv_path: Path, row: dict) -> None:
    new_file = not csv_path.exists() or csv_path.stat().st_size == 0
    check_csv_header(csv_path)
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        fh.flush()
        os.fsync(fh.fileno())


def parse_tasks(spec_str: str) -> list[TaskSpec]:
    """Task list for this run. `total` is always in it: every original column
    comes from it. Mandatory tasks are ordered first so a failure aborts the case
    before the slow optional ones run."""
    names = [t.strip() for t in spec_str.split(",") if t.strip()]
    unknown = [n for n in names if n not in TASK_SPECS]
    if unknown:
        raise SystemExit(f"error: unknown task(s): {', '.join(unknown)}\n"
                         f"choose from: {', '.join(TASK_SPECS)}")
    if "vertebrae_pp_refined" in names and "vertebrae_pp" in names:
        raise SystemExit("error: choose either vertebrae_pp_refined or vertebrae_pp, not both")
    if "total" not in names:
        names.append("total")
    order = list(TASK_SPECS)
    return [TASK_SPECS[n] for n in sorted(set(names),
                                          key=lambda n: (not TASK_SPECS[n].mandatory, order.index(n)))]


def process_patient(row: dict, source: Path | dict[str, Series], case_id: str, output: Path, *,
                    fast: bool, device: str, min_slices: int, tasks: list[TaskSpec],
                    nr_thr_saving: int, force_split: bool, total_vertebrae: bool = False,
                    flat: bool = False, recheck: bool = False, progress=None) -> None:
    """Fills `row` in place, so fields read before an exception survive into the CSV."""
    series = select_series(source, min_slices)
    row.update(read_identity(series, row["folder_id"]))
    if flat:  # case id IS the PatientID here; mismatch only means the tag was missing
        row["id_mismatch"] = row["dicom_patient_id"] == ""
    log.info("  PatientID=%s  StudyDate=%s  Accession=%s",
             row["dicom_patient_id"] or "<missing>", row["study_date"] or "<missing>",
             row["accession_number"] or "<missing>")

    ct_path = output / "ct" / f"{case_id}.nii.gz"
    masks_dir = masks_dir_for(output, case_id)
    files_sorted = sort_series_files(series)
    notes: list[str] = slice_spacing_notes(series)

    if recheck and (ct_path.exists() or masks_dir.exists()):
        if recheck_geometry(files_sorted, ct_path):
            dropped = [f for f in masks_dir.glob("*") if f.is_file()]
            for f in dropped:
                f.unlink(missing_ok=True)
            if dropped:
                log.info("  geometry CHANGED -> CT replaced, %d mask files dropped, re-segmenting", len(dropped))
                notes.append("geometry was wrong: CT re-converted, case re-segmented")
            else:
                notes.append("geometry was wrong: CT re-converted")
        else:
            log.info("  geometry unchanged -> keeping the existing masks")

    masks_dir.mkdir(parents=True, exist_ok=True)
    for f in masks_dir.glob("_partial_*"):  # leftovers of an interrupted run
        f.unlink(missing_ok=True)

    ct_input, ct_saved, tmpdir = ensure_ct(files_sorted, ct_path)
    available: dict[str, Path] = {}
    try:
        roi = total_roi_subset(total_vertebrae)
        for spec in tasks:
            if progress is not None:
                progress(spec.name)
            roi_subset = roi if spec.uses_roi_subset else None
            mask: Path | None = None
            if ct_saved:
                mask, reuse_notes = find_reusable(masks_dir, spec, ct_path, fast, roi_subset)
                if mask is not None:
                    notes += reuse_notes
                    log.info("  %s: reusing %s", spec.name, mask.name)
            if mask is None:
                mask_path, report_path = mask_files(masks_dir, spec.name, fast)
                log.info("  %s: segmenting (fast=%s, device=%s)...", spec.name,
                         fast and supports_fast(spec.name), device)
                try:
                    notes += run_task(ct_input, spec, mask_path, report_path, device=device,
                                      fast=fast, roi_subset=roi_subset,
                                      nr_thr_saving=nr_thr_saving, force_split=force_split,
                                      ct_name=ct_path.name if ct_saved else "<dicom series>")
                    if ct_saved:
                        verify_same_grid(mask_path, ct_path)
                    else:
                        verify_geometry(mask_path, files_sorted, strict=False)
                    mask = mask_path
                except Exception as e:
                    mask_path.unlink(missing_ok=True)
                    report_path.unlink(missing_ok=True)
                    if spec.mandatory:
                        raise
                    log.warning("  %s FAILED: %s", spec.name, e)
                    notes.append(f"{spec.name} failed: {type(e).__name__}: {str(e)[:120]}")
                    continue
            available[spec.name] = mask
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()

    if not ct_saved:
        notes.append("no CT saved")

    if progress is not None:
        progress("measuring")
    measurements, warnings = measure_case(ct_path if ct_saved else None, available, series)
    row.update(measurements)
    warnings = notes + warnings
    row["status"] = "ok" if not warnings else "ok; " + "; ".join(warnings)


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        tqdm.write(self.format(record), file=sys.stderr)


class ConsoleFilter(logging.Filter):
    """The console shows case-level progress and anything that went wrong. The
    log file keeps every line regardless, so --verbose only changes the screen."""

    def __init__(self, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose

    def filter(self, record: logging.LogRecord) -> bool:
        return self.verbose or record.levelno >= logging.WARNING or getattr(record, "console", False)


def setup_logging(log_path: Path, verbose: bool = False) -> None:
    if log.handlers:  # guard against duplicate handlers if called twice
        return
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    fh.setLevel(logging.DEBUG)
    ch = TqdmLoggingHandler()
    ch.setFormatter(logging.Formatter("%(message)s"))
    ch.addFilter(ConsoleFilter(verbose))
    log.addHandler(fh)
    log.addHandler(ch)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", dest="input_root", type=Path, required=True, metavar="FOLDER",
                    help="Read-only root folder containing one subfolder of DICOMs per patient")
    ap.add_argument("--output", type=Path, required=True, metavar="FOLDER",
                    help="Output folder (created if missing; gets ct/, masks/, full/ or fast/)")
    ap.add_argument("--fast", action="store_true",
                    help="Run `total` with the 3 mm model instead of 1.5 mm. For plumbing tests: "
                         "it shifts vertebral levels by a level. Rows go to <output>/fast/")
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASKS),
                    help="Comma-separated tasks to create when missing (default: all of "
                         + ", ".join(DEFAULT_TASKS) + "). `total` is always included. Masks that "
                         "already exist are reused whatever this is set to, so a cohort can be "
                         "built in passes.")
    ap.add_argument("--device", default="auto", type=device_arg,
                    help="auto|cpu|mps|gpu|gpu:N (default auto: CUDA, else Apple MPS, else CPU)")
    ap.add_argument("--nr-thr-saving", type=int, default=1,
                    help="nnU-Net export worker processes per model call (default 1; more only "
                         "adds process start-up for a single image)")
    ap.add_argument("--total-vertebrae", action="store_true",
                    help="Also ask `total` for the vertebrae. Off by default: the levels come from "
                         "the vertebral bodies of vertebrae_pp_refined, and asking `total` for "
                         "vertebrae runs a second 1.5 mm sub-model for labels nothing reads. Turn it "
                         "on to get the independent cross-check of the levels back (it is switched on "
                         "automatically when no vertebral-body task is in the run).")
    ap.add_argument("--force-split", action="store_true",
                    help="Process `total` in 3 chunks to save memory (do not use on small images)")
    ap.add_argument("--test", action="store_true", help="Process only the first 2 patients")
    ap.add_argument("--dry-run", action="store_true",
                    help="Only list patients and check DICOM series selection; no segmentation, "
                         "no CSV (with --flat it does save the file index, so the real run is fast)")
    ap.add_argument("--flat", action="store_true",
                    help="Input is a single DICOM export (e.g. DICOMDIR + IMAGES folder) with no "
                         "per-patient subfolders: cases are detected by the PatientID inside the files")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Put everything on the console, including what TotalSegmentator, torch "
                         "and ITK print. The log file always has it either way")
    ap.add_argument("--rescan", action="store_true",
                    help="With --flat: rebuild the saved index of which files belong to which "
                         "patient. Needed only after adding or moving data")
    ap.add_argument("--recheck-geometry", action="store_true",
                    help="Recovery pass: re-convert every saved CT with the fixed converter and "
                         "re-segment ONLY the cases whose geometry actually changed")
    ap.add_argument("--min-slices", type=int, default=20, help="Ignore series with fewer slices (default: 20)")
    args = ap.parse_args()

    if not args.input_root.is_dir():
        print(f"error: input root not found: {args.input_root}", file=sys.stderr)
        return 2

    res_dir = args.output / ("fast" if args.fast else "full")
    try:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "ct").mkdir(exist_ok=True)
        (args.output / "masks").mkdir(exist_ok=True)
        res_dir.mkdir(exist_ok=True)
    except OSError as e:
        print(f"error: cannot write to output folder {args.output}: {e}\n"
              "(read-only drive? NTFS drives mount read-only on macOS -- "
              "choose an output folder on a writable disk)", file=sys.stderr)
        return 2
    global VERBOSE
    VERBOSE = args.verbose
    setup_logging(res_dir / "pipeline.log", verbose=args.verbose)
    csv_path = res_dir / "results.csv"

    log.info("INPUT  (read-only): %s", args.input_root.resolve(), extra=SHOW)
    log.info("OUTPUT            : %s", args.output.resolve(), extra=SHOW)
    log.info("resolution        : %s -> %s", "fast (3 mm total)" if args.fast else "full (1.5 mm)",
             res_dir, extra=SHOW)

    try:
        check_csv_header(csv_path)
        tasks = parse_tasks(args.tasks)
        device = resolve_device(args.device)
    except SystemExit as e:
        log.error("%s", e)
        return 2
    disable_usage_stats()
    total_vertebrae = args.total_vertebrae or not any(t.name in VERTEBRA_TASKS for t in tasks)
    log.info("device            : %s%s", device, " (auto)" if args.device == "auto" else "",
             extra=SHOW)
    log.info("tasks             : %s", ", ".join(t.name for t in tasks), extra=SHOW)
    log.info("vertebrae from    : %s%s",
             next((t.name for t in tasks if t.name in VERTEBRA_TASKS), "total"),
             " (+ total, for the level cross-check)" if total_vertebrae and
             any(t.name in VERTEBRA_TASKS for t in tasks) else "")

    if (args.output / "segs").is_dir() or (args.output / "results.csv").exists():
        log.info("note: this folder also holds output of the single-task version (segs/ and/or a "
                 "top-level results.csv). Those have no run reports, so they are ignored and never "
                 "reused. Nothing is deleted.")

    # Guard against the obviously-wrong mode: a DICOMDIR anywhere near the top
    # means these are exports, which usually hold many patients each. Treating
    # such a folder as one patient would segment one of them and silently drop
    # the rest, so refuse before spending any time.
    if not args.flat:
        dicomdirs = find_dicomdirs(args.input_root)
        if dicomdirs:
            where = ", ".join(str(d.parent.relative_to(args.input_root)) or "<the input root>"
                              for d in dicomdirs[:4])
            log.error(
                "Found a DICOMDIR index in: %s%s\n"
                "A DICOMDIR marks a PACS or CD export, and one export usually holds MANY patients. "
                "Treating each of those folders as a single patient would process one patient and "
                "silently skip the others.\n"
                "Rerun with --flat: every file's header is read once and cases are grouped by the "
                "PatientID stored inside the files, no matter how the folders are nested.\n"
                "See the detected patient list first, without segmenting anything:\n"
                "    --flat --dry-run",
                where, f" (and {len(dicomdirs) - 4} more)" if len(dicomdirs) > 4 else "")
            return 2

    # Build the case list. A case is (case_id, source) where source is either a
    # patient folder (default mode) or a pre-scanned series pool (--flat).
    if args.flat:
        index = None if args.rescan else load_flat_index(flat_index_path(args.output), args.input_root)
        if index is None:
            cases = build_flat_cases(args.input_root)
            try:
                save_flat_index(flat_index_path(args.output), args.input_root, cases)
                log.info("flat index saved to %s (%d cases): every later run with this "
                         "--output skips the scan", flat_index_path(args.output), len(cases),
                         extra=SHOW)
            except OSError as e:
                log.warning("could not save the flat index (%s); the next run will scan again", e)
        else:
            cases = index
        kind = "patients (by PatientID + study)"
    else:
        cases = [(p.name, p) for p in sorted(args.input_root.iterdir())
                 if p.is_dir() and not p.name.startswith(".")]
        kind = "patient folders"
    if not cases:
        log.error("no %s found in %s", kind, args.input_root)
        return 2
    if args.test:
        cases = cases[:2]
        log.info("--test: limiting to first %d cases", len(cases))

    # Upfront roster. A case is NEVER silently skipped -- worst case it gets an
    # error row in results.csv.
    n_ready = 0
    detail = show_if(args.verbose or args.dry_run)
    log.info("=== %d %s found ===", len(cases), kind, extra=detail)
    for i, (case_id, _src) in enumerate(cases, start=1):
        md = masks_dir_for(args.output, case_id)
        have = 0
        for spec in tasks:
            stems = [spec.name]
            if args.fast and supports_fast(spec.name):
                stems.append(f"{spec.name}_fast")
            if any((md / f"{st}.nii.gz").exists() and (md / f"{st}.report.json").exists() for st in stems):
                have += 1
        n_ready += have == len(tasks)
        log.info("  %3d/%d  %-40s %d/%d masks present", i, len(cases), case_id, have, len(tasks),
                 extra=detail)
    log.info("=== %d %s: %d already complete, %d to segment ===",
             len(cases), kind, n_ready, len(cases) - n_ready, extra=SHOW)

    if args.dry_run:
        log.info("--dry-run: verifying DICOM series selection per case (no segmentation)",
                 extra=SHOW)
        n_bad = 0
        for i, (case_id, src) in enumerate(cases, start=1):
            try:
                sr = select_series(src, args.min_slices)
                ident = read_identity(sr, case_id)
                log.info("[%d/%d] %-24s PatientID=%s  StudyDate=%s  %d slices", i, len(cases),
                         case_id, ident["dicom_patient_id"] or "<missing>",
                         ident["study_date"] or "<missing>", sr.n_slices, extra=SHOW)
            except Exception as e:
                n_bad += 1
                log.error("[%d/%d] %-24s PROBLEM: %s", i, len(cases), case_id, e)
        log.info("=== dry-run done: %d/%d cases ok, %d with problems ===",
                 len(cases) - n_bad, len(cases), n_bad, extra=SHOW)
        return 0 if n_bad == 0 else 1

    n_ok = 0
    # The bar only makes sense on a terminal. Redirected to a file it would be
    # thousands of redraw lines, so it is switched off and each case announces
    # its start instead.
    on_screen = sys.stderr.isatty()
    bar = tqdm(cases, unit="case", file=sys.stderr, leave=True, disable=not on_screen,
               bar_format="{percentage:3.0f}%|{bar:18}| {n_fmt}/{total_fmt} "
                          "[{elapsed}<{remaining}] {desc}")

    def step(case: str, task: str) -> None:
        # Fixed widths keep the line from jittering as the task names change.
        bar.set_description_str(f"{case[:14]:<14} {task:<26}", refresh=True)

    for i, (case_id, src) in enumerate(bar, start=1):
        t0 = time.monotonic()
        step(case_id, "reading series")
        log.info("[%d/%d] %s", i, len(cases), case_id, extra=show_if(not on_screen))
        row = {c: "" for c in CSV_COLUMNS}
        row["folder_id"] = case_id
        try:
            process_patient(row, src, case_id, args.output, fast=args.fast, device=device,
                            min_slices=args.min_slices, tasks=tasks,
                            nr_thr_saving=args.nr_thr_saving, force_split=args.force_split,
                            total_vertebrae=total_vertebrae,
                            flat=args.flat, recheck=args.recheck_geometry,
                            progress=lambda t: step(case_id, t))
            n_ok += 1
        except Exception as e:
            log.debug("FAILED %s: %s\n%s", case_id, e, traceback.format_exc())
            log.error("[%d/%d] %s  FAILED: %s: %s", i, len(cases), case_id,
                      type(e).__name__, str(e)[:160])
            row["status"] = f"error: {type(e).__name__}: {str(e)[:200]}"
        elapsed = time.monotonic() - t0
        row["runtime_s"] = round(elapsed, 1)
        append_row(csv_path, row)
        log.info("  status: %s", row["status"])  # the log file always keeps it in full
        if not row["status"].startswith("error"):
            status = row["status"]
            short = status if len(status) <= 90 else status[:87] + "..."
            log.info("[%d/%d] %s  %s  (%s)", i, len(cases), case_id, short,
                     fmt_duration(elapsed), extra=SHOW)
    bar.close()

    n_bad = len(cases) - n_ok
    log.info("=== done: %d/%d ok%s | results: %s ===", n_ok, len(cases),
             f", {n_bad} with errors" if n_bad else "", csv_path, extra=SHOW)
    if n_bad:
        log.info("    the failed cases are in the status column; rerunning picks them up again",
                 extra=SHOW)
    return 0


if __name__ == "__main__":
    sys.exit(main())
