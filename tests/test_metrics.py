"""Synthetic-array tests for thorax_metrics (no data, no TotalSegmentator)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import thorax_metrics as tm  # noqa: E402

SP1 = (1.0, 1.0, 1.0)


def disk(shape2d, center, r):
    x, y = np.ogrid[: shape2d[0], : shape2d[1]]
    return (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r


def make_y_tree(shape=(60, 60, 60), carina_z=30, top_z=55, bottom_z=10):
    """Trachea (radius 3) from top_z down to carina_z, then two bronchi (radius 2)
    diverging in x down to bottom_z. The carina slice is carina_z (last single slice)."""
    vol = np.zeros(shape, dtype=bool)
    for z in range(carina_z, top_z + 1):
        vol[:, :, z] = disk(shape[:2], (30, 30), 3)
    for z in range(bottom_z, carina_z):
        off = 3 + (carina_z - 1 - z) // 3
        vol[:, :, z] = disk(shape[:2], (30 - off, 30), 2) | disk(shape[:2], (30 + off, 30), 2)
    return vol


def test_carina_simple_tree():
    vol = make_y_tree()
    res = tm.find_carina(vol, SP1)
    assert res.found and res.z == 30, res
    assert res.z_top == 55 and res.z_bottom == 10


def test_carina_bridges_hole_in_trachea():
    vol = make_y_tree()
    vol[:, :, 45] = False  # one missing slice in the trachea
    res = tm.find_carina(vol, SP1)
    assert res.found and res.z == 30, res


def test_carina_ignores_short_spur_and_hole_split():
    vol = make_y_tree()
    # short spur attached to the trachea for 2 slices (dies quickly)
    vol[34:37, 29:32, 40:42] = True
    # a wall through the trachea for 2 slices splits it into 2 components that re-merge
    vol[30, :, 49:51] = False
    res = tm.find_carina(vol, SP1)
    assert res.found and res.z == 30, res


def test_carina_not_moved_by_interruption_below():
    vol = make_y_tree()
    vol[:30, :, 20] = False  # one bronchus interrupted below the carina
    res = tm.find_carina(vol, SP1)
    assert res.found and res.z == 30, res


def test_carina_scan_starting_below_carina():
    vol = make_y_tree()[:, :, :25]
    res = tm.find_carina(vol, SP1)
    assert not res.found and res.reason == "no single trunk at top"


def test_carina_no_split():
    vol = np.zeros((40, 40, 40), dtype=bool)
    for z in range(5, 35):
        vol[:, :, z] = disk((40, 40), (20, 20), 3)
    res = tm.find_carina(vol, SP1)
    assert not res.found and res.reason == "no split"


def test_carina_empty():
    res = tm.find_carina(np.zeros((10, 10, 10), dtype=bool), SP1)
    assert not res.found and res.reason == "no airway lumen"


def make_branch_tree(shape=(64, 64, 64)):
    """Trunk (z 41..59) -> bar at z=40 -> two bronchi (z 26..39) -> y-bars at z=25 -> four terminals (z 10..24).
    Carina slice = 40. Segments below the carina = 6, terminals = 4."""
    vol = np.zeros(shape, dtype=bool)
    for z in range(41, 60):
        vol[:, :, z] = disk(shape[:2], (30, 30), 2)
    vol[22:39, 29:32, 40] = True                     # carina bar
    for x0 in (21, 36):
        vol[x0:x0 + 3, 29:32, 26:40] = True          # main bronchi
        vol[x0:x0 + 3, 23:38, 25] = True             # y-bar
        for y0 in (23, 35):
            vol[x0:x0 + 3, y0:y0 + 3, 10:25] = True  # terminals
    return vol


def test_branch_tree_carina_and_metrics():
    vol = make_branch_tree()
    # 3x3 tubes = 9 mm^2 at 1 mm spacing: below the default 10 mm^2 island filter
    res = tm.find_carina(vol, SP1, min_area_mm2=5.0)
    assert res.found and res.z == 40, res
    wall = np.zeros_like(vol)
    air = tm.airway_metrics(vol, wall, res.z, SP1)
    assert air.terminal_count == 4
    assert air.branch_count == 6, air
    assert air.lumen_vox == int(vol[:, :, :40].sum())
    assert air.cycle_rank == 0
    z_sk = tm.carina_from_skeleton(vol, SP1)
    assert z_sk is not None and abs(z_sk - 40) <= 2, z_sk


def test_airway_wall_leak_guard():
    vol = make_branch_tree()
    wall = np.zeros_like(vol)
    # ring around the trunk: 1 voxel outside -> counted only below the carina; trunk is above -> 0
    wall[:, :, 50] = disk(vol.shape[:2], (30, 30), 3) & ~vol[:, :, 50]
    # wall touching a bronchus below the carina
    wall[24:26, 29:32, 30] = True
    # detached blob far away
    wall[5:8, 5:8, 30:33] = True
    air = tm.airway_metrics(vol, wall, 40, SP1)
    assert air.wall_vox == int(wall[24:26, 29:32, 30].sum())


def test_vessel_bv5_small_line_vs_thick_cylinder():
    shape = (40, 40, 40)
    lv = np.zeros(shape, dtype=np.uint8)
    lv[10, 10, 5:36] = 3                              # 1-voxel artery line: r = 1 - 0.5 = 0.5 mm -> small
    for z in range(5, 36):
        lv[:, :, z][disk(shape[:2], (25, 25), 3)] = 4  # vein cylinder r ~ 2.5 mm -> large
    lobes = np.ones(shape, dtype=bool)
    res = tm.vessel_metrics(lv, lobes, SP1)
    assert res.artery_vox == 31
    assert res.vein_vox == 29 * 31
    assert res.small_vox == 31, res
    assert res.skeleton_fragments == 2


def test_vessel_restricted_to_lobes():
    lv = np.zeros((20, 20, 20), dtype=np.uint8)
    lv[5:8, 5:8, :] = 3
    lobes = np.zeros((20, 20, 20), dtype=bool)
    lobes[:, :, :10] = True
    res = tm.vessel_metrics(lv, lobes, SP1)
    assert res.artery_vox == 9 * 10


def test_parenchyma_density_erosion_and_laa():
    shape = (40, 40, 40)
    lungs = np.zeros(shape, dtype=bool)
    lungs[10:30, 10:30, 10:30] = True
    ct = np.full(shape, -500, dtype=np.int16)
    ct[10:30, 10:30, 10:30] = -500      # outer 2 layers stay -500 -> must be eroded away
    ct[12:28, 12:28, 12:28] = -800
    ct[14:18, 14:18, 14:18] = -1000     # 64 voxels below -950
    exclude = np.zeros(shape, dtype=bool)
    res = tm.parenchyma_density(ct, lungs, exclude, SP1, erode_mm=2.0)
    assert res.n_vox == 16 ** 3
    expected_mean = (-800 * (16 ** 3 - 64) + -1000 * 64) / 16 ** 3
    assert abs(res.mean_hu - expected_mean) < 1e-3
    assert abs(res.laa_pct - 100 * 64 / 16 ** 3) < 1e-6


def test_parenchyma_density_excludes_vessels():
    shape = (30, 30, 30)
    lungs = np.zeros(shape, dtype=bool)
    lungs[5:25, 5:25, 5:25] = True
    ct = np.full(shape, -800, dtype=np.int16)
    exclude = np.zeros(shape, dtype=bool)
    exclude[14:16, 14:16, :] = True
    ct[exclude] = 50
    res = tm.parenchyma_density(ct, lungs, exclude, SP1, erode_mm=2.0)
    assert res.mean_hu == pytest.approx(-800.0)


def test_effusion_split_sign_and_invariant():
    shape = (60, 40, 30)
    vert = np.zeros(shape, dtype=bool)
    vert[29:32, 5:10, :] = True          # spine column at x ~ 30
    eff = np.zeros(shape, dtype=bool)
    eff[40:46, 15:25, 5:15] = True       # larger x -> patient's RIGHT
    eff[10:16, 15:25, 5:12] = True       # smaller x -> LEFT
    n_r, n_l, note = tm.split_effusion_lr(eff, vert)
    assert note == ""
    assert n_r == 6 * 10 * 10 and n_l == 6 * 10 * 7
    assert n_r + n_l == int(eff.sum())


def test_effusion_split_interpolates_gaps_and_falls_back():
    shape = (40, 40, 20)
    vert = np.zeros(shape, dtype=bool)
    vert[18:22, 5:9, [2, 3, 8, 9, 14, 15]] = True   # bodies with disc gaps
    eff = np.zeros(shape, dtype=bool)
    eff[25:30, 20:25, 5:7] = True                    # in a gap slice, right of the midline
    n_r, n_l, _ = tm.split_effusion_lr(eff, vert)
    assert n_r == int(eff.sum()) and n_l == 0
    # no vertebrae at all: lung midline fallback
    lungs_R = np.zeros(shape, dtype=bool)
    lungs_R[28:36, 10:30, :] = True
    lungs_L = np.zeros(shape, dtype=bool)
    lungs_L[4:12, 10:30, :] = True
    n_r, n_l, note = tm.split_effusion_lr(eff, np.zeros(shape, dtype=bool), lungs_R=lungs_R, lungs_L=lungs_L)
    assert "lung midline" in note and n_r == int(eff.sum())


def test_edge_contact_faces():
    m = np.zeros((30, 30, 30), dtype=bool)
    m[5:25, 5:25, 5:30] = True      # touches the top face (S)
    assert tm.edge_contact(m) == ["S"]
    m2 = np.zeros((30, 30, 30), dtype=bool)
    m2[0:10, 5:25, 5:25] = True     # touches x = 0 face (L)
    assert tm.edge_contact(m2) == ["L"]
    assert tm.edge_contact(np.zeros((5, 5, 5), dtype=bool)) == []


def test_cavity_height_largest_component():
    m = np.zeros((30, 30, 40), dtype=np.uint8)
    m[5:25, 5:25, 5:26] = 2         # z 5..25 -> 20 slices * 2.5 mm
    m[0:3, 0:3, 30:39] = 2          # small detached blob higher up (ignored)
    assert tm.cavity_height(m == 2, (1.0, 1.0, 2.5)) == 50.0


def test_largest_component_and_z_extent():
    m = np.zeros((20, 20, 20), dtype=bool)
    m[2:5, 2:5, 2:5] = True
    m[10:18, 10:18, 10:18] = True
    big = tm.largest_component(m)
    assert big.sum() == 8 ** 3
    assert tm.z_extent_mm(big, (1, 1, 2), min_vox=1) == 14.0
