#!/usr/bin/env python
"""Generate a synthetic test case for the pipeline (no real data needed).

Creates:
  <root>/input/fake_patient_01/axial/   60-slice axial "CT" DICOM series
  <root>/input/fake_patient_01/scout/    3-slice coronal scout (must be ignored)
  <root>/output/segs/fake_patient_01.nii.gz   fake multilabel segmentation

Because the segmentation file already exists, running the pipeline on this case
exercises everything EXCEPT TotalSegmentator inference (resume logic, DICOM
series selection, identity tags, mask cleaning, measurements, CSV writing):

  uv run python make_fake_case.py
  uv run python pipeline.py --input fake_case/input --output fake_case/output --test

The fake segmentation deliberately contains traps:
  - a detached 9-voxel/slice blob above the right apex   (< 50 voxel/slice rule)
  - a thin connected spike above the left apex           (< 50 voxel/slice rule)
  - a detached >=50 voxel/slice blob below the left base (largest-component rule)
  - an empty slice between C7 and T1 at the right apex   (gap-walking rule)
  - a non-RAS affine (L,P axes flipped)                  (canonical reorientation)
  - a sagittally bowed (kyphotic) vertebral column       (coronal Cobb must be 0)
The expected results are printed at the end.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from pipeline import get_label_maps

SHAPE = (96, 96, 60)          # x, y, z voxels
SPACING = (1.5, 1.5, 2.5)     # mm  (z deliberately NOT 2.0 - header must be read)

R_LUNG = dict(center=(30, 48), radii=(14, 18), z=(10, 53))   # apex at 53 (gap slice)
L_LUNG = dict(center=(66, 48), radii=(12, 16), z=(12, 46))
# Vertebra blocks, top-down: 4 slices each with a 1-slice gap in between.
VERT_NAMES_TOP_DOWN = ["vertebrae_C7"] + [f"vertebrae_T{i}" for i in range(1, 11)]
VERT_TOP_SLICE = 57  # C7 = 54..57, T1 = 49..52, T2 = 44..47, ... T10 = 4..7

EXPECTED = {
    "start_R": "T1", "end_R": "T9", "start_L": "T2", "end_L": "T9",
    "start_both": "T1", "end_both": "T9",
    "height_R_mm": 107.5, "height_L_mm": 85.0, "height_both_mm": 107.5,
    # lungs are straight cylinders and the vertebral column is perfectly
    # aligned, so diagonals equal the heights and the Cobb-like angle is 0
    "diag_R_mm": 107.5, "diag_L_mm": 85.0, "diag_both_mm": 107.5,
    # bounding-box extents of the ellipses: center +/- radius, in voxels * 1.5mm
    "width_R_mm": 42.0, "depth_R_mm": 54.0,
    "width_L_mm": 36.0, "depth_L_mm": 48.0,
    "width_both_mm": 93.0, "depth_both_mm": 54.0,
    # R + L sums
    "height_sum_mm": 192.5, "diag_sum_mm": 192.5,
    "width_sum_mm": 78.0, "depth_sum_mm": 102.0,
    # whole column C7(top 57) .. T10(bottom 4); lung levels T1(top 52) .. T9(bottom 9)
    "spine_height_mm": 132.5, "spine_lung_span_mm": 107.5,
    "cobb_angle_deg": 0.0,
    "id_mismatch": True,
}


def expected_volumes() -> dict[str, float]:
    """Ground-truth volumes from the construction: ellipse voxels x slices
    (+ the connected spike on the left lung, which cleaning must keep)."""
    voxel_ml = SPACING[0] * SPACING[1] * SPACING[2] / 1000.0
    vox_r = int(ellipse_mask(R_LUNG["center"], R_LUNG["radii"]).sum()) * 44   # z 10..53
    vox_l = int(ellipse_mask(L_LUNG["center"], L_LUNG["radii"]).sum()) * 35 + 9 * 6  # z 12..46 + spike
    return {
        "vol_R_ml": round(vox_r * voxel_ml, 2),
        "vol_L_ml": round(vox_l * voxel_ml, 2),
        "vol_both_ml": round((vox_r + vox_l) * voxel_ml, 2),
    }


def ellipse_mask(center: tuple[int, int], radii: tuple[int, int]) -> np.ndarray:
    x, y = np.ogrid[: SHAPE[0], : SHAPE[1]]
    return ((x - center[0]) / radii[0]) ** 2 + ((y - center[1]) / radii[1]) ** 2 <= 1.0


def build_segmentation() -> np.ndarray:
    lobes, verts = get_label_maps()
    lab = np.zeros(SHAPE, dtype=np.uint8)

    # Right lung: lower lobe in bottom half, middle lobe thin band, upper lobe on top.
    rm = ellipse_mask(R_LUNG["center"], R_LUNG["radii"])
    z0, z1 = R_LUNG["z"]
    for z in range(z0, z1 + 1):
        frac = (z - z0) / (z1 - z0)
        name = ("lung_lower_lobe_right" if frac < 0.4
                else "lung_middle_lobe_right" if frac < 0.55 else "lung_upper_lobe_right")
        lab[rm, z] = lobes[name]

    # Left lung: lower / upper lobes.
    lm = ellipse_mask(L_LUNG["center"], L_LUNG["radii"])
    z0, z1 = L_LUNG["z"]
    for z in range(z0, z1 + 1):
        name = "lung_lower_lobe_left" if (z - z0) / (z1 - z0) < 0.5 else "lung_upper_lobe_left"
        lab[lm, z] = lobes[name]

    # Trap 1: detached 3x3 right-lung blob above the apex (9 vox/slice < 50).
    lab[28:31, 18:21, 56:59] = lobes["lung_upper_lobe_right"]
    # Trap 2: thin 3x3 spike CONNECTED to the left lung top (9 vox/slice < 50).
    cx, cy = L_LUNG["center"]
    lab[cx - 1 : cx + 2, cy - 1 : cy + 2, 47:53] = lobes["lung_upper_lobe_left"]
    # Trap 3: detached 12x12 left-lung blob far below the base (144 vox/slice >= 50,
    # only the largest-connected-component rule removes it).
    lab[60:72, 10:22, 3:6] = lobes["lung_lower_lobe_left"]

    # Vertebral column: 8x8 posterior blocks with 1-slice gaps. The column is
    # bowed in the SAGITTAL plane (kyphosis-like anteroposterior curve): the
    # coronal Cobb angle must still measure 0.0 despite it.
    for i, name in enumerate(VERT_NAMES_TOP_DOWN):
        top = VERT_TOP_SLICE - 5 * i
        dy = int(round(8 * math.sin(math.pi * i / (len(VERT_NAMES_TOP_DOWN) - 1))))
        lab[44:52, 66 + dy : 74 + dy, top - 3 : top + 1] = verts[name]
    return lab


def build_ct_volume(lab: np.ndarray) -> np.ndarray:
    lobes, verts = get_label_maps()
    hu = np.zeros(SHAPE, dtype=np.int16)                       # soft tissue-ish 0 HU
    hu[np.isin(lab, list(lobes.values()))] = -800              # air-ish "lungs"
    hu[np.isin(lab, list(verts.values()))] = 300               # bony vertebrae
    rng = np.random.default_rng(0)
    return (hu + rng.integers(-20, 20, SHAPE)).astype(np.int16)


def write_series(out_dir: Path, volume: np.ndarray | None, *, n_slices: int, axial: bool,
                 study_uid: str, series_desc: str, series_num: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    series_uid = generate_uid()
    for k in range(n_slices):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.PatientName = "Fake^Patient"
        ds.PatientID = "FAKE-123"                 # != folder name -> id_mismatch=True
        ds.AccessionNumber = "ACC-0001"
        ds.StudyDate = "20260830"
        ds.Modality = "CT"
        ds.SeriesDescription = series_desc
        ds.SeriesNumber = series_num
        ds.InstanceNumber = k + 1
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0] if axial else [1, 0, 0, 0, 0, -1]
        ds.ImagePositionPatient = [0, 0, k * SPACING[2]] if axial else [0, k * 10.0, 0]
        ds.PixelSpacing = [SPACING[1], SPACING[0]]
        ds.SliceThickness = SPACING[2]
        ds.RescaleIntercept = -1024
        ds.RescaleSlope = 1
        ds.Rows, ds.Columns = SHAPE[1], SHAPE[0]
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0

        if volume is not None and axial:
            stored = (volume[:, :, k].T.astype(np.int32) + 1024).clip(0, 4000)
        else:
            stored = np.full((SHAPE[1], SHAPE[0]), 1024, dtype=np.int32)
        ds.PixelData = stored.astype(np.uint16).tobytes()

        path = out_dir / f"IM{k + 1:04d}.dcm"
        if int(pydicom.__version__.split(".")[0]) >= 3:
            ds.save_as(path, enforce_file_format=True)
        else:
            ds.save_as(path, write_like_original=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("fake_case"), help="output root (default: ./fake_case)")
    args = ap.parse_args()

    patient_dir = args.root / "input" / "fake_patient_01"
    seg_path = args.root / "output" / "segs" / "fake_patient_01.nii.gz"

    lab = build_segmentation()
    vol = build_ct_volume(lab)

    study_uid = generate_uid()
    write_series(patient_dir / "axial", vol, n_slices=SHAPE[2], axial=True,
                 study_uid=study_uid, series_desc="Chest axial 2.5mm", series_num=2)
    write_series(patient_dir / "scout", None, n_slices=3, axial=False,
                 study_uid=study_uid, series_desc="Scout", series_num=1)

    # Non-RAS affine (axes point Left / Posterior) to exercise canonical reorientation.
    affine = np.diag([-SPACING[0], -SPACING[1], SPACING[2], 1.0])
    seg_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(lab, affine), seg_path)

    print(f"fake DICOM series : {patient_dir}  (axial 60 slices + 3-slice scout)")
    print(f"fake segmentation : {seg_path}")
    print("\nExpected results.csv values:")
    for k, v in {**EXPECTED, **expected_volumes()}.items():
        print(f"  {k:16s} = {v}")
    print("\nRun:  uv run python pipeline.py "
          f"--input {args.root / 'input'} --output {args.root / 'output'} --test")


if __name__ == "__main__":
    main()
