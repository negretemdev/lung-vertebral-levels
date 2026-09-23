"""Tests for the pipeline's own logic: task selection, the mask reuse rule, the
CSV guard, and one end-to-end run on the synthetic case (no real data, and no
TotalSegmentator inference, because every fake mask carries a run report)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline as pl  # noqa: E402


# ---------------------------------------------------------------------------
# Task selection
# ---------------------------------------------------------------------------

def test_default_tasks_and_order():
    tasks = pl.parse_tasks(",".join(pl.DEFAULT_TASKS))
    assert [t.name for t in tasks] == pl.DEFAULT_TASKS
    assert tasks[0].mandatory and tasks[1].mandatory
    assert not any(t.mandatory for t in tasks[2:])


def test_total_is_always_included():
    tasks = pl.parse_tasks("lung_vessels")
    assert "total" in [t.name for t in tasks]
    assert tasks[0].name == "total"  # mandatory first


def test_unknown_and_conflicting_tasks_are_refused():
    with pytest.raises(SystemExit):
        pl.parse_tasks("total,not_a_task")
    with pytest.raises(SystemExit):
        pl.parse_tasks("vertebrae_pp,vertebrae_pp_refined")


def test_vertebrae_pp_can_replace_the_refined_task():
    assert [t.name for t in pl.parse_tasks("total,vertebrae_pp")] == ["total", "vertebrae_pp"]


def test_only_total_supports_fast():
    assert pl.supports_fast("total")
    for t in ("lung_vessels", "vertebrae_pp_refined", "trunk_cavities", "pleural_pericard_effusion"):
        assert not pl.supports_fast(t)


def test_total_roi_subset_drops_the_vertebrae_by_default():
    without = pl.total_roi_subset(False)
    with_v = pl.total_roi_subset(True)
    assert pl.TRACHEA_NAME in without
    assert not any(n.startswith("vertebrae_") for n in without)
    assert sum(n.startswith("vertebrae_") for n in with_v) == 25
    assert set(without).issubset(set(with_v))


def test_device_argument_validation():
    for good in ("auto", "cpu", "mps", "gpu", "gpu:0", "gpu:3"):
        assert pl.device_arg(good) == good
    for bad in ("cuda", "gpu:", "GPU", "mps:0", ""):
        with pytest.raises(Exception):
            pl.device_arg(bad)


# ---------------------------------------------------------------------------
# results.csv guard
# ---------------------------------------------------------------------------

def test_csv_header_guard(tmp_path):
    good = tmp_path / "ok.csv"
    with open(good, "w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=pl.CSV_COLUMNS).writeheader()
    pl.check_csv_header(good)                      # must not raise
    pl.check_csv_header(tmp_path / "missing.csv")  # absent file is fine

    stale = tmp_path / "stale.csv"
    with open(stale, "w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=pl.CSV_COLUMNS[:-5]).writeheader()
    with pytest.raises(SystemExit) as e:
        pl.check_csv_header(stale)
    assert "different set of columns" in str(e.value)


def test_append_row_refuses_a_stale_file(tmp_path):
    stale = tmp_path / "results.csv"
    with open(stale, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["folder_id", "status"])
        w.writeheader()
        w.writerow({"folder_id": "a", "status": "ok"})
    with pytest.raises(SystemExit):
        pl.append_row(stale, {c: "" for c in pl.CSV_COLUMNS})


# ---------------------------------------------------------------------------
# Mask reuse
# ---------------------------------------------------------------------------

AFF = np.diag([-1.5, -1.5, 2.5, 1.0])


def _nifti(path: Path, affine=AFF) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.uint8), affine), path)


def _report(path: Path, task: str, *, fast=False, roi=None, version=None, ho=None) -> None:
    pl_block = {"resampling_order": pl.RESAMPLING_ORDER,
                "higher_order_resampling": pl.TASK_SPECS[task].higher_order if ho is None else ho}
    path.write_text(json.dumps({
        "task": task, "fast": fast, "roi_subset": roi,
        "totalsegmentator_version": version or pl.totalseg_version(),
        "pipeline": pl_block,
    }))


@pytest.fixture
def case(tmp_path):
    ct = tmp_path / "ct" / "c.nii.gz"
    _nifti(ct)
    masks = tmp_path / "masks" / "c"
    masks.mkdir(parents=True)
    return ct, masks


def test_mask_is_reused_only_with_a_matching_report(case):
    ct, masks = case
    _nifti(masks / "trunk_cavities.nii.gz")
    # no report yet: an interrupted run, never trusted
    ok, _ = pl.check_reusable(masks / "trunk_cavities.nii.gz", masks / "trunk_cavities.report.json",
                              ct, "trunk_cavities", False, None)
    assert not ok
    _report(masks / "trunk_cavities.report.json", "trunk_cavities")
    ok, notes = pl.check_reusable(masks / "trunk_cavities.nii.gz", masks / "trunk_cavities.report.json",
                                  ct, "trunk_cavities", False, None)
    assert ok and notes == []


def test_mask_on_a_different_grid_is_not_reused(case):
    ct, masks = case
    _nifti(masks / "trunk_cavities.nii.gz", affine=np.diag([-2.0, -2.0, 2.5, 1.0]))
    _report(masks / "trunk_cavities.report.json", "trunk_cavities")
    ok, _ = pl.check_reusable(masks / "trunk_cavities.nii.gz", masks / "trunk_cavities.report.json",
                              ct, "trunk_cavities", False, None)
    assert not ok


def test_roi_subset_superset_is_reusable_but_a_subset_is_not(case):
    ct, masks = case
    _nifti(masks / "total.nii.gz")
    _report(masks / "total.report.json", "total", roi=pl.total_roi_subset(True))
    # asking for fewer classes than the mask carries: reuse it
    got, _ = pl.find_reusable(masks, pl.TASK_SPECS["total"], ct, False, pl.total_roi_subset(False))
    assert got is not None
    # a mask that lacks a class we need is not reusable
    _report(masks / "total.report.json", "total", roi=pl.total_roi_subset(False))
    got, _ = pl.find_reusable(masks, pl.TASK_SPECS["total"], ct, False, pl.total_roi_subset(True))
    assert got is None


def test_fast_mode_prefers_the_full_resolution_mask(case):
    ct, masks = case
    roi = pl.total_roi_subset(False)
    _nifti(masks / "total_fast.nii.gz")
    _report(masks / "total_fast.report.json", "total", fast=True, roi=roi)
    got, notes = pl.find_reusable(masks, pl.TASK_SPECS["total"], ct, True, roi)
    assert got is not None and got.name == "total_fast.nii.gz"

    _nifti(masks / "total.nii.gz")
    _report(masks / "total.report.json", "total", fast=False, roi=roi)
    got, notes = pl.find_reusable(masks, pl.TASK_SPECS["total"], ct, True, roi)
    assert got.name == "total.nii.gz"
    assert any("full-resolution mask reused" in n for n in notes)


def test_full_mode_never_accepts_a_fast_mask(case):
    ct, masks = case
    roi = pl.total_roi_subset(False)
    _nifti(masks / "total_fast.nii.gz")
    _report(masks / "total_fast.report.json", "total", fast=True, roi=roi)
    got, _ = pl.find_reusable(masks, pl.TASK_SPECS["total"], ct, False, roi)
    assert got is None


def test_version_and_settings_differences_warn_but_still_reuse(case):
    ct, masks = case
    _nifti(masks / "lung_vessels.nii.gz")
    _report(masks / "lung_vessels.report.json", "lung_vessels", version="2.17.0", ho=False)
    ok, notes = pl.check_reusable(masks / "lung_vessels.nii.gz", masks / "lung_vessels.report.json",
                                  ct, "lung_vessels", False, None)
    assert ok
    assert any("2.17.0" in n for n in notes)
    assert any("resampling settings" in n for n in notes)


def test_mask_file_naming():
    d = Path("/tmp/x")
    assert pl.mask_files(d, "total", False)[0].name == "total.nii.gz"
    assert pl.mask_files(d, "total", True)[0].name == "total_fast.nii.gz"
    # tasks without a fast variant keep one name in both modes
    assert pl.mask_files(d, "lung_vessels", True)[0].name == "lung_vessels.nii.gz"


# ---------------------------------------------------------------------------
# Multi-patient exports must never be mistaken for one patient
# ---------------------------------------------------------------------------

def test_dicomdir_found_below_the_root(tmp_path):
    """The real layout that broke: exports one level down, each with its own
    DICOMDIR, so a check that only looked at the root saw nothing."""
    for day in ("2026-09-16-001", "2026-09-16-002"):
        d = tmp_path / day
        (d / "IMAGES").mkdir(parents=True)
        (d / "DICOMDIR").write_bytes(b"\0")
    found = pl.find_dicomdirs(tmp_path)
    assert {f.parent.name for f in found} == {"2026-09-16-001", "2026-09-16-002"}


def test_dicomdir_at_the_root_and_absent(tmp_path):
    (tmp_path / "DICOMDIR").write_bytes(b"\0")
    assert [f.parent for f in pl.find_dicomdirs(tmp_path)] == [tmp_path]

    clean = tmp_path / "clean"
    (clean / "PAT001" / "series").mkdir(parents=True)
    assert pl.find_dicomdirs(clean) == []


def test_folder_with_several_patients_is_refused(monkeypatch, tmp_path):
    def fake_scan(_root):
        return {f"uid{i}": pl.Series(uid=f"uid{i}", files=[Path(f"{i}.dcm")], n_slices=500,
                                     modality="CT", is_axial=True, patient_id=pid)
                for i, pid in enumerate(("PAT-A", "PAT-B", "PAT-C"))}

    monkeypatch.setattr(pl, "scan_series", fake_scan)
    with pytest.raises(RuntimeError) as e:
        pl.select_series(tmp_path, 20)
    assert "3 different PatientIDs" in str(e.value)
    assert "--flat" in str(e.value)


def test_single_patient_folder_is_accepted(monkeypatch, tmp_path):
    def fake_scan(_root):
        return {"uid1": pl.Series(uid="uid1", files=[Path("a.dcm")], n_slices=500, modality="CT",
                                  is_axial=True, patient_id="PAT-A"),
                "uid2": pl.Series(uid="uid2", files=[Path("b.dcm")], n_slices=60, modality="CT",
                                  is_axial=True, patient_id="")}  # missing tag, not a second patient

    monkeypatch.setattr(pl, "scan_series", fake_scan)
    assert pl.select_series(tmp_path, 20).uid == "uid1"


def test_flat_pools_skip_the_multi_patient_check():
    """In --flat mode the pool is already one patient, so the check must not fire
    even though it is handed a dict rather than a folder."""
    pool = {"uid1": pl.Series(uid="uid1", files=[Path("a.dcm")], n_slices=500, modality="CT",
                              is_axial=True, patient_id="PAT-A")}
    assert pl.select_series(pool, 20).uid == "uid1"


# ---------------------------------------------------------------------------
# End to end on the synthetic case
# ---------------------------------------------------------------------------

def _run(module, argv):
    old = sys.argv
    sys.argv = argv
    try:
        return module.main()
    finally:
        sys.argv = old


@pytest.mark.slow
def test_synthetic_case_end_to_end(tmp_path, capsys):
    import make_fake_case as mk

    _run(mk, ["make_fake_case.py", "--root", str(tmp_path)])
    assert _run(pl, ["pipeline.py", "--input", str(tmp_path / "input"),
                     "--output", str(tmp_path / "output"), "--test"]) == 0

    rows = list(csv.DictReader(open(tmp_path / "output" / "full" / "results.csv")))
    assert len(rows) == 1
    row = rows[0]

    lab = mk.build_segmentation()
    expected = {**mk.EXPECTED, **mk.expected_volumes(),
                **mk.expected_extra(lab, mk.build_lung_vessels(), mk.build_effusion(),
                                    mk.build_ct_volume(lab))}
    for key, want in expected.items():
        got = row[key]
        if isinstance(want, float):
            assert abs(float(got) - want) < 1e-9, f"{key}: expected {want}, got {got}"
        else:
            assert got == str(want), f"{key}: expected {want}, got {got}"

    # every column is filled, and the only warning is the honest one about the
    # 2.5 mm slices of this synthetic scan
    assert [c for c in row if row[c] == ""] == []
    assert row["status"] == "ok; small-vessel volume unreliable at 2.5 mm slices"
    # the masks were reused, so this took no inference at all
    assert float(row["runtime_s"]) < 30


@pytest.mark.slow
def test_deleting_one_mask_only_redoes_that_task(tmp_path):
    import make_fake_case as mk

    _run(mk, ["make_fake_case.py", "--root", str(tmp_path)])
    masks = tmp_path / "output" / "masks" / "fake_patient_01"
    (masks / "trunk_cavities.nii.gz").unlink()
    (masks / "trunk_cavities.report.json").unlink()

    assert _run(pl, ["pipeline.py", "--input", str(tmp_path / "input"),
                     "--output", str(tmp_path / "output"),
                     "--tasks", "total,vertebrae_pp_refined,lung_vessels,pleural_pericard_effusion",
                     "--test"]) == 0
    row = list(csv.DictReader(open(tmp_path / "output" / "full" / "results.csv")))[-1]
    # the cavity column is the only one that went empty; everything else survived
    assert row["thoracic_cavity_height_mm"] == ""
    assert row["airway_branch_count"] == str(mk.EXPECTED["airway_branch_count"])
    assert row["start_R"] == mk.EXPECTED["start_R"]
    assert not (masks / "trunk_cavities.nii.gz").exists()
