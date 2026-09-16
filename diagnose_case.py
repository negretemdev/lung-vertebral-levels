#!/usr/bin/env python
"""READ-ONLY diagnostic for one case: dump every geometry-relevant fact of the
DICOM series the pipeline would select, plus the saved ct/seg NIfTI affines.

Usage (run once for a BAD case and once for a GOOD case, paste both outputs):

  uv run python diagnose_case.py "/Volumes/Drive/exports/export_A" \
      --patient-id "12345678" \
      --ct  "/Volumes/Drive/lung_output/ct/12345678.nii.gz" \
      --seg "/Volumes/Drive/lung_output/segs/12345678.nii.gz" > bad_case.txt

--patient-id is only needed for flat exports holding several patients.
Nothing is written or modified; the input stays read-only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pydicom

from pipeline import scan_series, select_series, sort_series_files


def section(title: str) -> None:
    print(f"\n===== {title} =====")


def safe(fn, label: str):
    try:
        fn()
    except Exception as e:
        print(f"[{label} FAILED: {type(e).__name__}: {e}]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dicom_folder", type=Path, help="folder containing this case's DICOMs (an export folder is fine)")
    ap.add_argument("--patient-id", default=None, help="restrict to this PatientID (flat exports with several patients)")
    ap.add_argument("--ct", type=Path, default=None, help="the saved output/ct/<id>.nii.gz of this case")
    ap.add_argument("--seg", type=Path, default=None, help="the saved output/segs/<id>.nii.gz of this case")
    ap.add_argument("--min-slices", type=int, default=20)
    args = ap.parse_args()

    section("SERIES INVENTORY")
    if not args.dicom_folder.is_dir():
        print(f"FOLDER DOES NOT EXIST: {args.dicom_folder}")
        return
    all_series = scan_series(args.dicom_folder)
    if not all_series:
        files = [f for f in args.dicom_folder.rglob("*") if f.is_file() and not f.name.startswith(".")]
        print(f"NO readable DICOM series found, although the folder holds {len(files)} files.")
        for f in files[:8]:
            print(f"  e.g. {f.relative_to(args.dicom_folder)}")
        print("-> Is this the right folder level (the one containing DICOMDIR / IMAGES)?")
        print("-> If these files ARE the DICOMs, they may be unreadable by pydicom; send me one filename and its size.")
        return

    pids = sorted({s.patient_id or "<empty>" for s in all_series.values()})
    print(f"{len(all_series)} series total; PatientIDs present: {pids}")
    series = all_series
    if args.patient_id:
        series = {u: s for u, s in all_series.items() if s.patient_id == args.patient_id}
        if not series:
            print(f"\n--patient-id '{args.patient_id}' MATCHED NOTHING.")
            print("-> Re-run with one of the PatientIDs listed above (exact string, in quotes).")
            return
        print(f"{len(series)} series for PatientID {args.patient_id}")
    for s in series.values():
        mf = "  <-- MULTIFRAME" if s.n_slices > len(s.files) else ""
        print(f"  uid ...{s.uid[-10:]}  '{s.description}'  {s.modality}  patient={s.patient_id or '?'}  "
              f"files={len(s.files)}  slices={s.n_slices}  axial={s.is_axial}{mf}")

    try:
        chosen = select_series(series, args.min_slices)
    except Exception as e:
        print(f"\nSERIES SELECTION FAILED: {e}")
        print("-> This alone may be the bug for this case; paste the output so far.")
        return
    ds0 = chosen.first_ds
    section(f"CHOSEN SERIES uid ...{chosen.uid[-10:]}")
    sop = getattr(ds0, "SOPClassUID", None)
    print(f"SOPClassUID: {sop}  ({pydicom.uid.UID(str(sop)).name if sop else '?'})")
    for tag in ("Manufacturer", "ManufacturerModelName", "PatientPosition",
                "ImageOrientationPatient", "ImagePositionPatient", "InstanceNumber",
                "NumberOfFrames", "SliceThickness", "SpacingBetweenSlices",
                "PixelSpacing", "GantryDetectorTilt", "AcquisitionNumber", "Rows", "Columns"):
        print(f"  {tag}: {getattr(ds0, tag, '<ABSENT>')}")
    print(f"  SharedFunctionalGroupsSequence:   {'present' if 'SharedFunctionalGroupsSequence' in ds0 else 'absent'}")
    print(f"  PerFrameFunctionalGroupsSequence: {'present' if 'PerFrameFunctionalGroupsSequence' in ds0 else 'absent'}")

    def multiframe_detail():
        if "PerFrameFunctionalGroupsSequence" not in ds0:
            return
        ds_full = pydicom.dcmread(chosen.files[0], stop_before_pixels=True)
        pf = ds_full.PerFrameFunctionalGroupsSequence
        print(f"  per-frame groups: {len(pf)} frames; first/second/last frame IPP:")
        for k in (0, 1, len(pf) - 1):
            item = pf[k]
            ipp = iop = "<none>"
            if "PlanePositionSequence" in item:
                ipp = list(item.PlanePositionSequence[0].get("ImagePositionPatient", []))
            if "PlaneOrientationSequence" in item:
                iop = list(item.PlaneOrientationSequence[0].get("ImageOrientationPatient", []))
            print(f"    frame {k}: IPP={ipp}  IOP={iop}")
        if "SharedFunctionalGroupsSequence" in ds_full:
            sh = ds_full.SharedFunctionalGroupsSequence[0]
            if "PlaneOrientationSequence" in sh:
                print(f"    shared IOP: {list(sh.PlaneOrientationSequence[0].get('ImageOrientationPatient', []))}")
    safe(multiframe_detail, "multiframe detail")

    section("SORT-KEY ANALYSIS (the order our pipeline feeds the converter)")

    def sort_analysis():
        files = sort_series_files(chosen)
        iop = getattr(ds0, "ImageOrientationPatient", None)
        normal = (np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
                  if iop is not None and len(iop) == 6 else np.array([0.0, 0.0, 1.0]))
        n_ipp = n_inst = n_none = 0
        projs, iops, acqs, insts = [], set(), set(), []
        for f in files:
            ds = pydicom.dcmread(f, stop_before_pixels=True,
                                 specific_tags=["ImagePositionPatient", "ImageOrientationPatient",
                                                "InstanceNumber", "AcquisitionNumber"])
            ipp = getattr(ds, "ImagePositionPatient", None)
            if ipp is not None and len(ipp) == 3:
                n_ipp += 1
                projs.append(float(np.dot(np.array(ipp, float), normal)))
            elif getattr(ds, "InstanceNumber", None) is not None:
                n_inst += 1
            else:
                n_none += 1
            fiop = getattr(ds, "ImageOrientationPatient", None)
            if fiop is not None:
                iops.add(tuple(round(float(v), 4) for v in fiop))
            acq = getattr(ds, "AcquisitionNumber", None)
            if acq is not None:
                acqs.add(int(acq))
            insts.append(getattr(ds, "InstanceNumber", None))
        print(f"files: {len(files)} | sort key sources: IPP-projection={n_ipp}, "
              f"InstanceNumber-fallback={n_inst}, none(0.0)={n_none}")
        print(f"distinct IOPs in series: {len(iops)}" + ("  <-- INCONSISTENT" if len(iops) > 1 else ""))
        print(f"distinct AcquisitionNumbers: {sorted(acqs) if acqs else '<absent>'}")
        for label, idx in (("first 3", range(min(3, len(files)))), ("last 3", range(max(0, len(files) - 3), len(files)))):
            for i in idx:
                p = f"{projs[i]:.2f}" if i < len(projs) and n_inst == 0 and n_none == 0 else "?"
                print(f"  {label} [{i}] {files[i].name}  InstanceNumber={insts[i]}  proj={p}")
        if len(projs) == len(files) and len(projs) > 1:
            d = np.diff(np.array(projs))
            dup = int(np.sum(np.abs(d) < 0.01))
            rev = int(np.sum(d < -0.01))
            print(f"consecutive position deltas: mean={d.mean():.3f} std={d.std():.4f} "
                  f"min={d.min():.3f} max={d.max():.3f}")
            print(f"duplicate positions: {dup}" + ("  <-- MULTI-PHASE?" if dup else "") +
                  f" | backward steps: {rev}" + ("  <-- NON-MONOTONIC" if rev else ""))
        else:
            print("(projection sequence incomplete -> ordering relied on fallbacks)")
    safe(sort_analysis, "sort analysis")

    section("SIMPLEITK GEOMETRY (in memory only, nothing written)")

    def sitk_check():
        import SimpleITK as sitk
        files = sort_series_files(chosen)
        r = sitk.ImageSeriesReader()
        r.SetFileNames([str(f) for f in files])
        img = r.Execute()
        print(f"our-order conversion : size={img.GetSize()} spacing={tuple(round(v, 3) for v in img.GetSpacing())}")
        print(f"  direction={tuple(round(v, 3) for v in img.GetDirection())}")
        print(f"  origin   ={tuple(round(v, 2) for v in img.GetOrigin())}")
        gd = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(files[0].parent), chosen.uid)
        if gd:
            r2 = sitk.ImageSeriesReader()
            r2.SetFileNames(gd)
            img2 = r2.Execute()
            same_order = [str(f) for f in files] == list(gd)
            print(f"GDCM-order conversion: size={img2.GetSize()} spacing={tuple(round(v, 3) for v in img2.GetSpacing())}")
            print(f"  direction={tuple(round(v, 3) for v in img2.GetDirection())}")
            print(f"  origin   ={tuple(round(v, 2) for v in img2.GetOrigin())}")
            print(f"  same file order as ours: {same_order}"
                  + (f"  (gdcm first={Path(gd[0]).name} last={Path(gd[-1]).name})" if not same_order else ""))
        else:
            print("GDCM found no files for this series uid in that directory (files spread across dirs?)")
    safe(sitk_check, "sitk check")

    section("SAVED NIFTI AFFINES")

    def nifti_check():
        import nibabel as nib
        for label, p in (("ct ", args.ct), ("seg", args.seg)):
            if p is None:
                continue
            if not p.exists():
                print(f"{label}: {p} does not exist")
                continue
            img = nib.load(p)
            print(f"{label}: {p.name} shape={img.shape} zooms={tuple(round(float(v), 3) for v in img.header.get_zooms()[:3])} "
                  f"axcodes={nib.aff2axcodes(img.affine)}")
            print(np.round(img.affine, 2))
    safe(nifti_check, "nifti check")

    print("\ndone. Paste this whole output back.")


if __name__ == "__main__":
    main()
