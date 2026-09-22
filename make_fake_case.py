#!/usr/bin/env python
"""Generate a synthetic test case for the pipeline (no real data needed).

Creates:
  <root>/input/fake_patient_01/axial/       60-slice axial "CT" DICOM series
  <root>/input/fake_patient_01/scout/        3-slice coronal scout (must be ignored)
  <root>/output/masks/fake_patient_01/<task>.nii.gz  + <task>.report.json

Because every mask already exists and carries a run report, running the pipeline
on this case exercises everything EXCEPT TotalSegmentator inference: the DICOM
series selection, the geometry check, the reuse rule, the mask cleaning, every
measurement and the CSV writing.

  uv run python make_fake_case.py
  uv run python pipeline.py --input fake_case/input --output fake_case/output --test

The masks deliberately contain traps:
  - a detached 9-voxel/slice blob above the right apex   (< 50 voxel/slice rule)
  - a thin connected spike above the left apex           (< 50 voxel/slice rule)
  - a detached >=50 voxel/slice blob below the left base (largest-component rule)
  - an empty slice between C7 and T1 at the right apex   (gap-walking rule)
  - a non-RAS affine (L,P axes flipped)                  (canonical reorientation)
  - a sagittally bowed (kyphotic) vertebral column       (coronal Cobb must be 0)
  - a detached airway-wall blob far from the airway      (wall leak guard)
  - a detached airway-lumen blob                         (largest-component rule)
  - an effusion blob straddling the spine midline        (R/L split, half each)
  - a detached thoracic-cavity blob                      (largest-component rule)
The expected results are printed at the end.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

import pipeline as pl
import thorax_metrics as tmx
from pipeline import get_label_maps

SHAPE = (96, 96, 60)          # x, y, z voxels
SPACING = (1.5, 1.5, 2.5)     # mm  (z deliberately NOT 2.0 - header must be read)
VOX_ML = SPACING[0] * SPACING[1] * SPACING[2] / 1000.0

R_LUNG = dict(center=(30, 48), radii=(14, 18), z=(10, 53))   # apex at 53 (gap slice)
L_LUNG = dict(center=(66, 48), radii=(12, 16), z=(12, 46))
# Vertebra blocks, top-down: 4 slices each with a 1-slice gap in between.
VERT_NAMES_TOP_DOWN = ["vertebrae_C7"] + [f"vertebrae_T{i}" for i in range(1, 11)]
VERT_TOP_SLICE = 57  # C7 = 54..57, T1 = 49..52, T2 = 44..47, ... T10 = 4..7

CARINA_Z = 40                 # the airway tree splits here, by construction
LAA_BLOCK = (slice(26, 35), slice(54, 61), slice(20, 31))    # low-attenuation area in the right lung

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
    # --- columns added by the multi-task extension --------------------------
    "slice_thickness_mm": 2.5,
    "convolution_kernel": "STANDARD",
    "carina_level": "T3",                       # slice 40 sits inside T3 (39..42)
    "carina_to_apex_mm": 32.5,                  # (53 - 40) * 2.5
    "carina_to_base_mm": 75.0,                  # (40 - 10) * 2.5
    "airway_branch_count": 6,                   # 2 main bronchi + 4 terminals below the carina
    "thoracic_cavity_height_mm": 117.5,         # z 8..55
}


def ellipse_mask(center: tuple[int, int], radii: tuple[int, int]) -> np.ndarray:
    x, y = np.ogrid[: SHAPE[0], : SHAPE[1]]
    return ((x - center[0]) / radii[0]) ** 2 + ((y - center[1]) / radii[1]) ** 2 <= 1.0


def disk_mask(center: tuple[int, int], r: int) -> np.ndarray:
    x, y = np.ogrid[: SHAPE[0], : SHAPE[1]]
    return (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r


def expected_volumes() -> dict[str, float]:
    """Ground-truth volumes from the construction: ellipse voxels x slices
    (+ the connected spike on the left lung, which cleaning must keep)."""
    vox_r = int(ellipse_mask(R_LUNG["center"], R_LUNG["radii"]).sum()) * 44   # z 10..53
    vox_l = int(ellipse_mask(L_LUNG["center"], L_LUNG["radii"]).sum()) * 35 + 9 * 6  # z 12..46 + spike
    return {
        "vol_R_ml": round(vox_r * VOX_ML, 2),
        "vol_L_ml": round(vox_l * VOX_ML, 2),
        "vol_both_ml": round((vox_r + vox_l) * VOX_ML, 2),
    }


def build_segmentation() -> np.ndarray:
    """The `total` mask: 5 lung lobes, vertebrae, trachea."""
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
        dy = vert_bow(i)
        lab[44:52, 66 + dy : 74 + dy, top - 3 : top + 1] = verts[name]

    # Trachea, between the two lungs and above the carina. Never touches a lobe
    # or a vertebra label, so no existing measurement can change.
    lab[47:50, 41:44, CARINA_Z + 1 : 60] = pl.get_trachea_id()
    return lab


def vert_bow(i: int) -> int:
    return int(round(8 * math.sin(math.pi * i / (len(VERT_NAMES_TOP_DOWN) - 1))))


def build_vertebrae_pp() -> np.ndarray:
    """The vertebrae_pp_refined mask: the vertebral BODY only, a 6x6 core of each
    block, on exactly the same slices, so every level and span is unchanged."""
    pp = pl.get_pp_label_map()
    lab = np.zeros(SHAPE, dtype=np.uint8)
    for i, name in enumerate(VERT_NAMES_TOP_DOWN):
        top = VERT_TOP_SLICE - 5 * i
        dy = vert_bow(i)
        lab[45:51, 67 + dy : 73 + dy, top - 3 : top + 1] = pp[name]
    return lab


def build_lung_vessels() -> np.ndarray:
    """Airway lumen and wall plus arteries and veins.

    The airway is a trachea that splits at CARINA_Z into two main bronchi, each
    of which splits again into two terminals: exactly 6 skeleton branches below
    the carina. The wall is the one-voxel ring around the lumen on every slice.
    """
    ids = pl.task_label_ids("lung_vessels", pl._FALLBACK_LUNG_VESSELS)
    lumen = np.zeros(SHAPE, dtype=bool)
    lumen[47:50, 41:44, CARINA_Z + 1 : 60] = True          # trachea
    lumen[38:61, 41:44, CARINA_Z] = True                   # carina bar: last single slice
    for x0 in (38, 58):
        lumen[x0 : x0 + 3, 41:44, 30:CARINA_Z] = True      # main bronchus
        lumen[x0 : x0 + 3, 34:51, 29] = True               # splits into two
        for y0 in (34, 48):
            lumen[x0 : x0 + 3, y0 : y0 + 3, 20:29] = True  # terminal
    tree = lumen.copy()
    lumen[70:72, 38:40, 30:34] = True                      # detached noise blob

    # Wall: the 8-neighbour ring around the lumen on each slice.
    from scipy import ndimage
    wall = np.zeros(SHAPE, dtype=bool)
    for z in range(SHAPE[2]):
        if tree[:, :, z].any():
            wall[:, :, z] = ndimage.binary_dilation(tree[:, :, z], structure=np.ones((3, 3))) & ~tree[:, :, z]
    wall[20:23, 60:63, 22:26] = True                       # detached wall blob (leak trap)

    lab = np.zeros(SHAPE, dtype=np.uint8)
    lab[lumen] = ids["lung_airways"]
    lab[wall & ~lumen] = ids["lung_airways_wall"]
    # One thick and one thin vessel per lung: the thin one is a "small vessel".
    for z in range(12, 51):
        lab[:, :, z][disk_mask((30, 48), 3)] = ids["lung_arteries"]
    lab[24, 40, 12:51] = ids["lung_arteries"]
    for z in range(14, 45):
        lab[:, :, z][disk_mask((66, 48), 3)] = ids["lung_veins"]
    lab[70, 54, 14:45] = ids["lung_veins"]
    return lab


def build_effusion() -> np.ndarray:
    """Pleural effusion on both sides plus a blob straddling the spine midline
    (x 44..51 around a midline of 47.5, so it must split exactly in half)."""
    ids = pl.task_label_ids("pleural_pericard_effusion", pl._FALLBACK_EFFUSION)
    lab = np.zeros(SHAPE, dtype=np.uint8)
    lab[14:20, 40:56, 12:25] = ids["pleural_effusion"]      # small x = patient RIGHT
    lab[78:84, 40:56, 14:23] = ids["pleural_effusion"]      # large x = patient LEFT
    lab[44:52, 30:34, 40:43] = ids["pleural_effusion"]      # straddles the midline
    lab[44:53, 30:39, 20:31] = ids["pericardial_effusion"]
    return lab


def build_trunk_cavities() -> np.ndarray:
    ids = pl.task_label_ids("trunk_cavities", pl._FALLBACK_TRUNK)
    lab = np.zeros(SHAPE, dtype=np.uint8)
    lab[10:86, 20:81, 8:56] = ids["thoracic_cavity"]        # z 8..55 -> 47 * 2.5 = 117.5 mm
    lab[2:6, 2:6, 0:2] = ids["thoracic_cavity"]             # detached blob, must be dropped
    return lab


def build_ct_volume(lab: np.ndarray) -> np.ndarray:
    """A CT in Hounsfield units. Noise everywhere except inside the lungs, so the
    expected parenchymal density stays interpretable."""
    lobes, verts = get_label_maps()
    lung = np.isin(lab, list(lobes.values()))
    hu = np.zeros(SHAPE, dtype=np.int32)
    hu[lung] = -800
    hu[np.isin(lab, list(verts.values()))] = 300
    hu[lab == pl.get_trachea_id()] = -900
    rng = np.random.default_rng(0)
    hu += np.where(lung, 0, rng.integers(-20, 20, SHAPE))
    laa = np.zeros(SHAPE, dtype=bool)
    laa[LAA_BLOCK] = True
    hu[laa & lung] = -1000                                   # emphysema-like block
    return hu.astype(np.int16)


def expected_extra(lab: np.ndarray, lv: np.ndarray, eff: np.ndarray, ct: np.ndarray) -> dict:
    """Expected values of the added columns, from the ground-truth arrays.

    Volumes are plain voxel counts, so they are independent of the metric code.
    The density pair is recomputed with the same definition the pipeline uses
    (lungs minus vessels and airways, eroded 2 mm): what this checks is that the
    pipeline pairs the right CT with the right masks in the right orientation.
    """
    lobes, _ = get_label_maps()
    ids = pl.task_label_ids("lung_vessels", pl._FALLBACK_LUNG_VESSELS)
    eids = pl.task_label_ids("pleural_pericard_effusion", pl._FALLBACK_EFFUSION)

    right = pl._largest_component(np.isin(lab, [lobes[n] for n in pl.RIGHT_LOBES]))
    left = pl._largest_component(np.isin(lab, [lobes[n] for n in pl.LEFT_LOBES]))
    both = right | left
    lobe_mask = np.isin(lab, [lobes[n] for n in pl.LUNG_LOBES])

    lumen = lv == ids["lung_airways"]
    wall = lv == ids["lung_airways_wall"]
    tree = tmx.largest_component(lumen, 26)
    below = np.zeros(SHAPE, dtype=bool)
    below[:, :, :CARINA_Z] = True
    attached = wall & (tmx.largest_component(tree | wall, 26)) & below
    lumen_ml = int((tree & below).sum()) * VOX_ML
    wall_ml = int(attached.sum()) * VOX_ML

    artery = (lv == ids["lung_arteries"]) & lobe_mask
    vein = (lv == ids["lung_veins"]) & lobe_mask
    a_ml, v_ml = int(artery.sum()) * VOX_ML, int(vein.sum()) * VOX_ML
    thin = int(lv[24, 40, 12:51].size + lv[70, 54, 14:45].size)  # the two 1-voxel lines

    x_idx = np.arange(SHAPE[0])[:, None, None]
    pleural = eff == eids["pleural_effusion"]
    # stored x grows toward the patient's LEFT, so right = small x
    right_eff = pleural & (x_idx < 47.5)

    dens = tmx.parenchyma_density(ct, both, lv > 0, SPACING)
    return {
        "airway_lumen_vol_ml": round(lumen_ml, 2),
        "airway_wall_vol_ml": round(wall_ml, 2),
        "airway_wall_pct": round(100.0 * wall_ml / (wall_ml + lumen_ml), 2),
        "airway_lumen_lung_ratio": round(lumen_ml / (int(both.sum()) * VOX_ML), 4),
        "artery_vol_ml": round(a_ml, 2),
        "vein_vol_ml": round(v_ml, 2),
        "artery_vein_ratio": round(a_ml / v_ml, 4),
        "small_vessel_vol_ml": round(thin * VOX_ML, 2),
        "pleural_eff_R_ml": round(int(right_eff.sum()) * VOX_ML, 2),
        "pleural_eff_L_ml": round(int((pleural & ~right_eff).sum()) * VOX_ML, 2),
        "pericardial_eff_ml": round(int((eff == eids["pericardial_effusion"]).sum()) * VOX_ML, 2),
        "lung_mean_hu": round(dens.mean_hu, 1),
        "laa950_pct": round(dens.laa_pct, 2),
    }


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
        ds.ConvolutionKernel = "STANDARD"
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


def write_mask(masks_dir: Path, task: str, lab: np.ndarray, roi_subset: list[str] | None) -> None:
    """Save one mask plus the run report the reuse rule needs to accept it."""
    # Non-RAS affine (axes point Left / Posterior), matching what the pipeline's
    # own DICOM converter produces for this series.
    affine = np.diag([-SPACING[0], -SPACING[1], SPACING[2], 1.0])
    masks_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(lab, affine), masks_dir / f"{task}.nii.gz")
    spec = pl.TASK_SPECS[task]
    report = {
        "totalsegmentator_version": pl.totalseg_version(),
        "task": task,
        "device": "fake",
        "fast": False,
        "fastest": False,
        "multilabel": True,
        "roi_subset": roi_subset,
        "runtime_seconds": 0.0,
        "output_files": [],
        "pipeline": {
            "resampling_order": pl.RESAMPLING_ORDER,
            "higher_order_resampling": spec.higher_order,
            "robust_crop": pl.ROBUST_CROP,
            "effective_fast": False,
            "device_requested": "fake",
            "device_used": "fake",
            "retried_on_cpu": False,
            "force_split": False,
            "nr_thr_saving": 1,
            "ct_file": "fake_patient_01.nii.gz",
            "pipeline_git_sha": None,
        },
    }
    (masks_dir / f"{task}.report.json").write_text(json.dumps(report, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("fake_case"), help="output root (default: ./fake_case)")
    args = ap.parse_args()

    patient_dir = args.root / "input" / "fake_patient_01"
    masks_dir = args.root / "output" / "masks" / "fake_patient_01"

    lab = build_segmentation()
    pp = build_vertebrae_pp()
    lv = build_lung_vessels()
    eff = build_effusion()
    trunk = build_trunk_cavities()
    vol = build_ct_volume(lab)

    study_uid = generate_uid()
    write_series(patient_dir / "axial", vol, n_slices=SHAPE[2], axial=True,
                 study_uid=study_uid, series_desc="Chest axial 2.5mm", series_num=2)
    write_series(patient_dir / "scout", None, n_slices=3, axial=False,
                 study_uid=study_uid, series_desc="Scout", series_num=1)

    # `total` is saved WITH the vertebrae so the level cross-check is exercised;
    # the reuse rule accepts it for a run that asks for fewer classes.
    write_mask(masks_dir, "total", lab, pl.total_roi_subset(include_vertebrae=True))
    write_mask(masks_dir, "vertebrae_pp_refined", pp, None)
    write_mask(masks_dir, "lung_vessels", lv, None)
    write_mask(masks_dir, "pleural_pericard_effusion", eff, None)
    write_mask(masks_dir, "trunk_cavities", trunk, None)

    print(f"fake DICOM series : {patient_dir}  (axial 60 slices + 3-slice scout)")
    print(f"fake masks        : {masks_dir}  (5 tasks + run reports)")
    print("\nExpected results.csv values:")
    expected = {**EXPECTED, **expected_volumes(), **expected_extra(lab, lv, eff, vol)}
    for k, v in expected.items():
        print(f"  {k:26s} = {v}")
    print("\nRun:  uv run python pipeline.py "
          f"--input {args.root / 'input'} --output {args.root / 'output'} --test")


if __name__ == "__main__":
    main()
