#!/usr/bin/env python
"""Batch pipeline: lung craniocaudal length + vertebral level span from chest CT DICOMs.

For each patient subfolder of the input root:
  1. Select the axial CT series with the most slices (>= --min-slices).
  2. Run TotalSegmentator (task "total", multilabel, lung lobes + all vertebrae)
     on the DICOM series, saving output/segs/<folder_id>.nii.gz. If that file
     already exists, segmentation is skipped and only measurements are redone.
  3. Measure per side (right / left / both): craniocaudal lung length in mm and
     the vertebral levels overlapping the lung apex and base slices.
  4. Append one row to output/results.csv immediately (crash-safe batch).

The input root is treated as strictly read-only.

Usage:
  uv run python pipeline.py --input /Volumes/DRIVE/cts --output ./out            # fast (3mm)
  uv run python pipeline.py --input /Volumes/DRIVE/cts --output ./out --full     # full-res
  uv run python pipeline.py --input /Volumes/DRIVE/cts --output ./out --test     # first 2 only
  uv run python pipeline.py --input /Volumes/DRIVE/cts --output ./out --dry-run  # check only
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
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

CSV_COLUMNS = [
    "folder_id", "dicom_patient_id", "accession_number", "study_date", "id_mismatch",
    "start_R", "end_R", "start_L", "end_L", "start_both", "end_both",
    "height_R_mm", "height_L_mm", "height_both_mm", "height_sum_mm",
    "diag_R_mm", "diag_L_mm", "diag_both_mm", "diag_sum_mm",
    "width_R_mm", "width_L_mm", "width_both_mm", "width_sum_mm",
    "depth_R_mm", "depth_L_mm", "depth_both_mm", "depth_sum_mm",
    "vol_R_ml", "vol_L_ml", "vol_both_ml",
    "spine_height_mm", "spine_lung_span_mm", "cobb_angle_deg",
    "status", "runtime_s",
]

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

    def key(f: Path) -> float:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True, specific_tags=["ImagePositionPatient", "InstanceNumber"])
            ipp = getattr(ds, "ImagePositionPatient", None)
            if ipp is not None and len(ipp) == 3:
                return float(np.dot(np.array(ipp, float), normal))
            return float(getattr(ds, "InstanceNumber", 0) or 0)
        except Exception:
            return 0.0

    return sorted(s.files, key=key)


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


def run_segmentation(files_sorted: list[Path], seg_path: Path, ct_path: Path,
                     fast: bool, device: str) -> None:
    """Convert the series to output/ct/<id>.nii.gz, then run TotalSegmentator on
    that NIfTI -- so the saved CT and the mask are on the SAME voxel grid (QC =
    open both in a viewer). Falls back to direct DICOM input if conversion fails
    (no CT saved then)."""
    from totalsegmentator.python_api import totalsegmentator

    lobes, verts = get_label_maps()
    roi_subset = list(lobes) + list(verts)
    seg_path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp name and rename only on success: an interrupted run must
    # not leave a truncated seg file that the resume logic would trust forever.
    tmp_out = seg_path.with_name("_partial_" + seg_path.name)

    def _run(input_path: Path) -> None:
        totalsegmentator(
            input=input_path, output=tmp_out, task="total", ml=True,
            fast=fast, roi_subset=roi_subset, device=device, quiet=True,
        )

    try:
        use_fallback = False
        try:
            if not ct_path.exists():
                convert_series_to_nifti(files_sorted, ct_path)
        except GeometryError:
            raise  # broken geometry aborts the case; never fall back around it
        except Exception as e:
            log.warning("  NIfTI conversion failed (%s); trying TotalSegmentator "
                        "directly on the DICOMs (no CT NIfTI will be saved)", e)
            use_fallback = True

        if not use_fallback:
            verify_geometry(ct_path, files_sorted)  # GeometryError -> error row, no segmentation
            _run(ct_path)
        else:
            with tempfile.TemporaryDirectory(prefix="ts_dicom_") as td:
                # Symlink only the selected series into a clean folder: keeps the
                # input read-only and hides other series from the converter.
                link_dir = Path(td) / "dicom"
                link_dir.mkdir()
                for i, f in enumerate(files_sorted):
                    os.symlink(f.resolve(), link_dir / f"{i:05d}.dcm")
                _run(link_dir)

        if not tmp_out.exists():
            raise RuntimeError("TotalSegmentator finished but produced no output file")
        os.replace(tmp_out, seg_path)
        if use_fallback:
            try:  # TS converted internally: verify its output too (weak check)
                verify_geometry(seg_path, files_sorted, strict=False)
            except GeometryError:
                seg_path.unlink(missing_ok=True)  # never leave a flipped seg for resume
                raise
    finally:
        tmp_out.unlink(missing_ok=True)


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


def measure_case(seg_path: Path) -> tuple[dict, list[str]]:
    """Compute all measurement columns from a saved multilabel segmentation."""
    import nibabel as nib

    lobes, verts = get_label_maps()
    img = nib.as_closest_canonical(nib.load(seg_path))  # RAS: axis 2 = inferior -> superior
    data = np.asanyarray(img.dataobj).astype(np.int16)
    z_spacing = float(img.header.get_zooms()[2])
    nz = data.shape[2]
    log.info("  mask %s, spacing %s mm", data.shape, tuple(round(float(v), 3) for v in img.header.get_zooms()[:3]))

    # Per-slice vertebra presence (computed once).
    id_to_vert = {i: n for n, i in verts.items()}
    per_slice_verts: list[list[str]] = []
    for z in range(nz):
        counts = np.bincount(data[:, :, z].ravel())
        present = [id_to_vert[i] for i in np.flatnonzero(counts) if i in id_to_vert]
        per_slice_verts.append(present)

    def side_mask(names: list[str]) -> np.ndarray:
        m = np.isin(data, [lobes[n] for n in names])
        return _largest_component(m) if m.any() else m

    warnings: list[str] = []
    results = {c: "" for c in CSV_COLUMNS
               if c.startswith(("start_", "end_", "height_", "diag_", "width_", "depth_",
                                "vol_", "spine_", "cobb_"))}
    spacing = np.array(img.header.get_zooms()[:3], dtype=float)

    masks = {"R": side_mask(RIGHT_LOBES), "L": side_mask(LEFT_LOBES)}
    masks["both"] = masks["R"] | masks["L"]

    raw: dict[str, dict[str, float]] = {"R": {}, "L": {}, "both": {}}
    for side in ("R", "L", "both"):
        ext = _z_extremes(masks[side])
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

    stats = vertebra_stats(data, verts, spacing)
    cobb = cobb_ferguson_angle(stats, nz)
    if cobb is None:
        warnings.append("cobb angle not computable (<5 usable vertebrae)")
    else:
        results["cobb_angle_deg"] = cobb
    centroids = {n: s.centroid for n, s in stats.items()}
    flags = vertebra_sanity_flags(data, verts, centroids)
    warnings += [f"vertebra check: {f}" for f in flags]
    return results, warnings


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------

def append_row(csv_path: Path, row: dict) -> None:
    new_file = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        fh.flush()
        os.fsync(fh.fileno())


def process_patient(row: dict, source: Path | dict[str, Series], seg_path: Path,
                    fast: bool, device: str, min_slices: int,
                    flat: bool = False, recheck: bool = False) -> None:
    """Fills `row` in place, so fields read before an exception survive into the CSV."""
    series = select_series(source, min_slices)
    row.update(read_identity(series, row["folder_id"]))
    if flat:  # case id IS the PatientID here; mismatch only means the tag was missing
        row["id_mismatch"] = row["dicom_patient_id"] == ""

    ct_path = seg_path.parent.parent / "ct" / seg_path.name
    files_sorted = sort_series_files(series)
    notes: list[str] = []

    if recheck and (seg_path.exists() or ct_path.exists()):
        if recheck_geometry(files_sorted, ct_path):
            if seg_path.exists():
                seg_path.unlink()
                log.info("  geometry CHANGED -> CT replaced, re-segmenting this case")
                notes.append("geometry was wrong: CT re-converted, case re-segmented")
            else:
                notes.append("geometry was wrong: CT re-converted")
        else:
            log.info("  geometry unchanged -> keeping existing segmentation")

    if seg_path.exists():
        log.info("  segmentation exists, skipping TotalSegmentator: %s", seg_path.name)
        if not ct_path.exists():  # backfill the QC CT for cases segmented before ct/ existed
            try:
                convert_series_to_nifti(files_sorted, ct_path)
                log.info("  saved CT NIfTI for QC: %s", ct_path.name)
            except Exception as e:
                log.warning("  could not save CT NIfTI (%s)", e)
        # Geometry validation on EVERY case, resumed ones included: a flipped
        # volume must surface as an error row, never as silent measurements.
        if ct_path.exists():
            verify_geometry(ct_path, files_sorted)
            verify_same_grid(seg_path, ct_path)
        else:
            verify_geometry(seg_path, files_sorted, strict=False)
    else:
        log.info("  running TotalSegmentator (fast=%s, device=%s)...", fast, device)
        run_segmentation(files_sorted, seg_path, ct_path, fast=fast, device=device)

    measurements, warnings = measure_case(seg_path)
    row.update(measurements)
    warnings = notes + warnings
    row["status"] = "ok" if not warnings else "ok; " + "; ".join(warnings)


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        tqdm.write(self.format(record), file=sys.stderr)


def setup_logging(log_path: Path) -> None:
    if log.handlers:  # guard against duplicate handlers if called twice
        return
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    ch = TqdmLoggingHandler()
    ch.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(fh)
    log.addHandler(ch)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", dest="input_root", type=Path, required=True, metavar="FOLDER",
                    help="Read-only root folder containing one subfolder of DICOMs per patient")
    ap.add_argument("--output", type=Path, required=True, metavar="FOLDER",
                    help="Output folder (created if missing; will contain segs/, results.csv, pipeline.log)")
    ap.add_argument("--full", action="store_true", help="Use the full-resolution model (default: fast 3mm)")
    ap.add_argument("--test", action="store_true", help="Process only the first 2 patients")
    ap.add_argument("--dry-run", action="store_true",
                    help="Only list patients and check DICOM series selection; no segmentation, no CSV")
    ap.add_argument("--flat", action="store_true",
                    help="Input is a single DICOM export (e.g. DICOMDIR + IMAGES folder) with no "
                         "per-patient subfolders: cases are detected by the PatientID inside the files")
    ap.add_argument("--recheck-geometry", action="store_true",
                    help="Recovery pass: re-convert every saved CT with the fixed converter and "
                         "re-segment ONLY the cases whose geometry actually changed")
    ap.add_argument("--device", default="mps", choices=["mps", "cpu", "gpu"], help="Inference device (default: mps)")
    ap.add_argument("--min-slices", type=int, default=20, help="Ignore series with fewer slices (default: 20)")
    args = ap.parse_args()

    if not args.input_root.is_dir():
        print(f"error: input root not found: {args.input_root}", file=sys.stderr)
        return 2

    try:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "segs").mkdir(exist_ok=True)
    except OSError as e:
        print(f"error: cannot write to output folder {args.output}: {e}\n"
              "(read-only drive? NTFS drives mount read-only on macOS -- "
              "choose an output folder on a writable disk)", file=sys.stderr)
        return 2
    setup_logging(args.output / "pipeline.log")
    csv_path = args.output / "results.csv"

    log.info("INPUT  (read-only): %s", args.input_root.resolve())
    log.info("OUTPUT            : %s", args.output.resolve())

    # Guard against the obviously-wrong mode: a DICOMDIR at the input root means
    # this is a single flat export, not a folder-per-patient cohort.
    if not args.flat and any((args.input_root / n).exists() for n in ("DICOMDIR", "dicomdir")):
        log.error("The input root contains a DICOMDIR: this looks like a single DICOM export, "
                  "not one folder per patient. Rerun with --flat to group patients by their "
                  "PatientID inside the files.")
        return 2

    # Build the case list. A case is (case_id, source) where source is either a
    # patient folder (default mode) or a pre-scanned series pool (--flat).
    if args.flat:
        cases: list[tuple[str, Path | dict[str, Series]]] = build_flat_cases(args.input_root)
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
    n_done = 0
    log.info("=== %d %s found ===", len(cases), kind)
    for i, (case_id, _src) in enumerate(cases, start=1):
        has_seg = (args.output / "segs" / f"{case_id}.nii.gz").exists()
        n_done += has_seg
        log.info("  %3d/%d  %-40s %s", i, len(cases), case_id,
                 "seg exists -> measurements only" if has_seg else "needs segmentation")
    log.info("=== %d to segment, %d already segmented | fast=%s | device=%s ===",
             len(cases) - n_done, n_done, not args.full, args.device)

    if args.dry_run:
        log.info("--dry-run: verifying DICOM series selection per case (nothing is written)")
        n_bad = 0
        for i, (case_id, src) in enumerate(cases, start=1):
            log.info("[%d/%d] %s", i, len(cases), case_id)
            try:
                s = select_series(src, args.min_slices)
                ident = read_identity(s, case_id)
                log.info("  PatientID=%s  StudyDate=%s",
                         ident["dicom_patient_id"] or "<missing>", ident["study_date"] or "<missing>")
            except Exception as e:
                n_bad += 1
                log.error("  PROBLEM: %s", e)
        log.info("=== dry-run done: %d/%d cases ok, %d with problems ===",
                 len(cases) - n_bad, len(cases), n_bad)
        return 0 if n_bad == 0 else 1

    n_ok = 0
    for i, (case_id, src) in enumerate(tqdm(cases, unit="case", desc="patients"), start=1):
        t0 = time.monotonic()
        log.info("[%d/%d] %s", i, len(cases), case_id)
        seg_path = args.output / "segs" / f"{case_id}.nii.gz"
        row = {c: "" for c in CSV_COLUMNS}
        row["folder_id"] = case_id
        try:
            process_patient(row, src, seg_path, fast=not args.full,
                            device=args.device, min_slices=args.min_slices,
                            flat=args.flat, recheck=args.recheck_geometry)
            n_ok += 1
        except Exception as e:
            log.error("  FAILED %s: %s\n%s", case_id, e, traceback.format_exc())
            row["status"] = f"error: {type(e).__name__}: {str(e)[:200]}"
        row["runtime_s"] = round(time.monotonic() - t0, 1)
        append_row(csv_path, row)
        log.info("  [%d/%d] %s -> %s (%.1fs)", i, len(cases), case_id, row["status"], row["runtime_s"])

    log.info("=== done: %d/%d ok | results: %s ===", n_ok, len(cases), csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
