#!/usr/bin/env python
"""Pure mask -> metric functions for the chest-CT pipeline.

No I/O and no TotalSegmentator in here: every function takes numpy arrays and
returns numbers, so the same code is exercised by the unit tests, by the
Phase-1 diagnostic and by pipeline.py.

Conventions
-----------
* Every array is a 3-D volume already reoriented to canonical RAS
  (``nib.as_closest_canonical``): axis 0 = x, increasing toward the patient's
  RIGHT; axis 1 = y, increasing toward ANTERIOR; axis 2 = z, increasing toward
  SUPERIOR (the head). "Below" therefore means a smaller z index.
* ``spacing`` = (dx, dy, dz) in mm, read from the NIfTI header, never assumed.
* Functions crop to a bounding box internally and return counts / indices in
  full-volume coordinates.
* Nothing in here guesses anatomy. Every number is a property of the masks
  (and of the CT values inside them): the carina is where the airway-lumen
  mask splits, a "small vessel" is a vessel-mask voxel whose local radius is
  small, and so on.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

# 26-neighbourhood kernel (centre excluded) and structuring elements.
K26 = np.ones((3, 3, 3), dtype=np.uint8)
K26[1, 1, 1] = 0
STRUCT_26 = np.ones((3, 3, 3), dtype=bool)
STRUCT_8 = np.ones((3, 3), dtype=bool)

BV_AREA_MM2 = 5.0  # "small vessel" = cross-sectional area below 5 mm^2 (BV5)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def voxel_ml(spacing) -> float:
    """Volume of one voxel in millilitres."""
    return float(np.prod(np.asarray(spacing, dtype=float))) / 1000.0


def largest_component(mask: np.ndarray, connectivity: int = 26) -> np.ndarray:
    """Keep the largest connected component (26- or 6-connectivity)."""
    mask = np.asarray(mask).astype(bool, copy=False)
    if not mask.any():
        return np.zeros_like(mask)
    labeled, n = ndimage.label(mask, structure=STRUCT_26 if connectivity == 26 else None)
    if n <= 1:
        return mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return labeled == int(np.argmax(sizes))


def bbox_slices(mask: np.ndarray, pad: int = 1) -> tuple[slice, slice, slice] | None:
    """Bounding box of the True voxels as a slice tuple (padded, clipped), or None if empty."""
    objs = ndimage.find_objects(np.asarray(mask).astype(np.uint8, copy=False))
    if not objs or objs[0] is None:
        return None
    return tuple(slice(max(0, s.start - pad), min(n, s.stop + pad))
                 for s, n in zip(objs[0], mask.shape))


def z_extent_mm(mask: np.ndarray, spacing, min_vox: int = 50) -> float | None:
    """Craniocaudal extent (mm) over axial slices carrying >= min_vox voxels."""
    counts = np.asarray(mask).sum(axis=(0, 1))
    valid = np.flatnonzero(counts >= min_vox)
    if valid.size == 0:
        return None
    return float((valid[-1] - valid[0]) * float(spacing[2]))


def edge_contact(mask: np.ndarray, min_vox: int = 50) -> list[str]:
    """Faces of the volume the mask touches (>= min_vox voxels on that face).

    Canonical RAS face letters: L/R = x, P/A = y, I/S = z. Contact with I or S
    means a craniocaudal length measured on this mask is only a lower bound.
    """
    m = np.asarray(mask)
    faces = []
    if m[0].sum() >= min_vox:
        faces.append("L")
    if m[-1].sum() >= min_vox:
        faces.append("R")
    if m[:, 0].sum() >= min_vox:
        faces.append("P")
    if m[:, -1].sum() >= min_vox:
        faces.append("A")
    if m[:, :, 0].sum() >= min_vox:
        faces.append("I")
    if m[:, :, -1].sum() >= min_vox:
        faces.append("S")
    return faces


def cavity_height(cavity: np.ndarray, spacing, min_vox: int = 50) -> float | None:
    """Craniocaudal height (mm) of the largest component of a cavity mask."""
    m = largest_component(np.asarray(cavity).astype(bool, copy=False), 26)
    return z_extent_mm(m, spacing, min_vox)


def _skeleton_neighbours(sk: np.ndarray) -> np.ndarray:
    """Number of 26-neighbours of every skeleton voxel (0 outside the skeleton)."""
    nb = ndimage.convolve(sk.astype(np.uint8), K26, mode="constant", cval=0)
    return nb * sk


def _geodesic_length_mm(coords: np.ndarray, start_idx: int, spacing) -> float:
    """Length of a skeleton chain: BFS from the start voxel over 26-adjacency,
    summing the physical distance between consecutively visited voxels."""
    if len(coords) < 2:
        return 0.0
    sp = np.asarray(spacing, dtype=float)
    key = {tuple(c): i for i, c in enumerate(coords)}
    seen = np.zeros(len(coords), dtype=bool)
    order = []
    q = deque([start_idx])
    seen[start_idx] = True
    while q:
        i = q.popleft()
        order.append(i)
        c = coords[i]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    j = key.get((c[0] + dx, c[1] + dy, c[2] + dz))
                    if j is not None and not seen[j]:
                        seen[j] = True
                        q.append(j)
    pts = coords[order] * sp
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


# ---------------------------------------------------------------------------
# Carina: where the airway lumen mask splits
# ---------------------------------------------------------------------------

@dataclass
class CarinaResult:
    z: int                  # full-volume z index of the carina slice (last slice where the trunk is ONE component); -1 = not found
    reason: str             # "" when found, otherwise why not
    n2d: np.ndarray         # per axial slice: number of 2-D lumen components of the tree (full-volume length)
    z_top: int              # highest slice holding airway lumen (largest component); -1 if none
    z_bottom: int           # lowest such slice

    @property
    def found(self) -> bool:
        return self.z >= 0


def find_carina(lumen: np.ndarray, spacing, *, min_area_mm2: float = 10.0,
                persist_mm: float = 8.0, min_child_frac: float = 0.10,
                max_gap_slices: int = 1) -> CarinaResult:
    """Locate the carina from the airway-lumen mask alone.

    Rule: take the largest 26-connected lumen component (the tracheobronchial
    tree). Label 8-connected 2-D components on every axial slice (islands
    smaller than `min_area_mm2` are ignored). Start from the single trunk on
    the top slice and walk toward the feet. The carina is the last slice on
    which the trunk is still one component such that, on the slices below,
    it has >= 2 children (each >= `min_child_frac` of the parent area) which
    stay separate and do not re-merge for at least `persist_mm`. Holes in the
    mask (a missing slice) are bridged up to `max_gap_slices`; an oblique
    trachea is followed with an adaptive in-plane dilation. Interruptions of
    a bronchus BELOW the carina cannot move the result because the sweep stops
    at the first persistent split.

    Failure reasons: "no airway lumen", "no single trunk at top" (the scan
    starts below the carina, or the mask holds two trunks), "trunk lost at
    z=N" (mask broken for more than max_gap_slices), "no split".
    """
    lumen = np.asarray(lumen).astype(bool, copy=False)
    nz_full = lumen.shape[2]
    empty = np.zeros(nz_full, dtype=int)
    if not lumen.any():
        return CarinaResult(-1, "no airway lumen", empty, -1, -1)
    # Connected components with vertical gaps of up to max_gap_slices bridged:
    # a missing slice in the trachea must not cut the tree in two.
    if max_gap_slices > 0:
        z_struct = np.zeros((3, 3, 3), dtype=bool)
        z_struct[1, 1, :] = True
        bridged = ndimage.binary_dilation(lumen, structure=z_struct, iterations=max_gap_slices)
    else:
        bridged = lumen
    labeled, ncomp = ndimage.label(bridged, structure=STRUCT_26)
    sizes = np.bincount(labeled[lumen].ravel(), minlength=ncomp + 1)  # original voxels per component
    sizes[0] = 0
    main = int(np.argmax(sizes))
    tree = lumen & (labeled == main)
    if ncomp > 1:
        # A scan that starts below the carina leaves the two main bronchi as two
        # SEPARATE components that both reach the top of the airway region and
        # overlap in z. A bronchus broken further down forms a component whose
        # top lies well below; a piece above a gap wider than max_gap_slices
        # does not overlap the main component's top.
        objs = ndimage.find_objects(labeled)
        z_top_main = objs[main - 1][2].stop - 1
        for c in range(1, ncomp + 1):
            if c == main or sizes[c] < 0.05 * sizes[main]:
                continue
            z_lo, z_hi = objs[c - 1][2].start, objs[c - 1][2].stop - 1
            if z_hi >= z_top_main - 1 and z_lo <= z_top_main:
                n2d = empty.copy()
                zs = np.flatnonzero(tree.any(axis=(0, 1)))
                n2d[zs] = 1
                return CarinaResult(-1, "no single trunk at top", n2d, int(zs[-1]), int(zs[0]))

    sl = bbox_slices(tree, pad=0)
    sub = tree[sl]
    z0 = sl[2].start
    dx, dy, dz = (float(v) for v in spacing)
    min_vox = max(1, int(round(min_area_mm2 / (dx * dy))))
    dil_max = max(1, int(math.ceil(dz / min(dx, dy))))
    nz = sub.shape[2]

    lab = np.zeros(sub.shape, dtype=np.int32)
    areas: list[dict[int, int]] = []
    for z in range(nz):
        l2, n = ndimage.label(sub[:, :, z], structure=STRUCT_8)
        if n:
            counts = np.bincount(l2.ravel())
            keep = counts >= min_vox
            keep[0] = False
            l2 = np.where(keep[l2], l2, 0)
            counts = np.bincount(l2.ravel())
            areas.append({i: int(c) for i, c in enumerate(counts) if i and c})
        else:
            areas.append({})
        lab[:, :, z] = l2
    n2d = empty.copy()
    n2d[z0:z0 + nz] = [len(a) for a in areas]
    nonempty = [z for z in range(nz) if areas[z]]
    if not nonempty:
        return CarinaResult(-1, "no airway lumen", n2d, -1, -1)
    top, bottom = nonempty[-1], nonempty[0]
    z_top_full, z_bottom_full = z0 + top, z0 + bottom

    def children(z_from: int, comps: set[int], z_to: int, dil: int) -> set[int]:
        fp = np.isin(lab[:, :, z_from], list(comps))
        if dil > 0:
            fp = ndimage.binary_dilation(fp, structure=STRUCT_8, iterations=dil)
        return {int(v) for v in np.unique(lab[:, :, z_to][fp])} - {0}

    def step_down(z_from: int, comps: set[int]) -> tuple[int | None, set[int]]:
        """Components continuing `comps` below z_from (bridging small gaps)."""
        for gap in range(0, max_gap_slices + 1):
            z_to = z_from - 1 - gap
            if z_to < 0:
                return None, set()
            for dil in sorted({1, dil_max + gap}):
                kids = children(z_from, comps, z_to, dil)
                if kids:
                    return z_to, kids
        return None, set()

    k_slices = max(2, int(math.ceil(persist_mm / dz)))

    def persistent(z_split: int, kids: list[int]) -> bool:
        fronts: list[tuple[int, set[int]]] = [(z_split, {k}) for k in kids]
        for _ in range(k_slices - 1):
            new_fronts: list[tuple[int, set[int]]] = []
            for zf, comps in fronts:
                if zf == 0:  # tree reaches the bottom of the volume: cannot disprove, keep alive
                    new_fronts.append((zf, comps))
                    continue
                z_to, nxt = step_down(zf, comps)
                if z_to is None:
                    return False  # this child dies within persist_mm -> spur / noise, not a bronchus
                new_fronts.append((z_to, nxt))
            for i in range(len(new_fronts)):
                for j in range(i + 1, len(new_fronts)):
                    if new_fronts[i][0] == new_fronts[j][0] and new_fronts[i][1] & new_fronts[j][1]:
                        return False  # the fronts re-merge -> it was a hole in the trunk
            fronts = new_fronts
        return True

    comps_top = sorted(areas[top].items(), key=lambda kv: -kv[1])
    if len(comps_top) > 1 and comps_top[1][1] >= min_child_frac * comps_top[0][1]:
        return CarinaResult(-1, "no single trunk at top", n2d, z_top_full, z_bottom_full)
    trunk, z = comps_top[0][0], top
    while True:
        z_to, kids = step_down(z, {trunk})
        if z_to is None:
            reason = "no split" if z - 1 - max_gap_slices < 0 else f"trunk lost at z={z0 + z}"
            return CarinaResult(-1, reason, n2d, z_top_full, z_bottom_full)
        parent_area = areas[z][trunk]
        big = [k for k in kids if areas[z_to][k] >= min_child_frac * parent_area]
        if len(big) >= 2 and persistent(z_to, big):
            return CarinaResult(z0 + z, "", n2d, z_top_full, z_bottom_full)
        trunk = max(kids, key=lambda k: areas[z_to][k])
        z = z_to


def carina_from_skeleton(lumen: np.ndarray, spacing, *, min_branch_mm: float = 15.0) -> int | None:
    """Independent cross-check of find_carina using the skeleton graph.

    Root = the skeleton endpoint with the largest lumen radius (top of the
    trachea). Walk down the graph; the first junction with >= 2 branches whose
    subtrees are each longer than `min_branch_mm` is the carina. Returns the
    full-volume z index of that junction, or None.
    """
    lumen = np.asarray(lumen).astype(bool, copy=False)
    tree = largest_component(lumen, 26)
    if not tree.any():
        return None
    sl = bbox_slices(tree, pad=1)
    T = tree[sl]
    z0 = sl[2].start
    sk = skeletonize(ndimage.binary_fill_holes(T), method="lee").astype(bool)
    if sk.sum() < 3:
        return None
    nb = _skeleton_neighbours(sk)
    J = nb >= 3
    E = nb == 1
    seg, nseg = ndimage.label(sk & ~J, structure=STRUCT_26)
    clusters, ncl = ndimage.label(J, structure=STRUCT_26)
    if nseg == 0:
        return None
    sp = np.asarray(spacing, dtype=float)
    mean_step = float(sp.mean())
    seg_sizes = np.bincount(seg.ravel(), minlength=nseg + 1)
    seg_len = {s: max(0.0, (seg_sizes[s] - 1) * mean_step) for s in range(1, nseg + 1)}
    # adjacency: cluster -> segments touching it; segment -> clusters
    cl_segs: dict[int, set[int]] = {c: set() for c in range(1, ncl + 1)}
    seg_cls: dict[int, set[int]] = {s: set() for s in range(1, nseg + 1)}
    cl_objs = ndimage.find_objects(clusters)
    for c, obj in enumerate(cl_objs, start=1):
        if obj is None:
            continue
        pad = tuple(slice(max(0, o.start - 1), min(n, o.stop + 1)) for o, n in zip(obj, sk.shape))
        fp = ndimage.binary_dilation(clusters[pad] == c, structure=STRUCT_26)
        touching = {int(s) for s in np.unique(seg[pad][fp])} - {0}
        cl_segs[c] = touching
        for s in touching:
            seg_cls[s].add(c)
    # root: the endpoint whose skeleton segment has the largest MEAN lumen radius
    # (the trachea). The radius at the endpoint voxel itself is useless: at the
    # open end of any tube the distance to the background is one voxel.
    edt = ndimage.distance_transform_edt(T, sampling=spacing)
    ends = np.argwhere(E)
    if len(ends) == 0:
        return None
    seg_ids = np.arange(1, nseg + 1)
    seg_mean_r = dict(zip(seg_ids.tolist(), np.atleast_1d(ndimage.mean(edt, labels=seg, index=seg_ids)).tolist()))
    end_segs = [int(seg[tuple(v)]) for v in ends]
    root_seg = end_segs[int(np.argmax([seg_mean_r.get(s, 0.0) for s in end_segs]))]
    if root_seg == 0:
        return None

    def subtree_length(s: int, from_cluster: int, seen: set[int]) -> float:
        total = seg_len[s]
        for c in seg_cls[s] - {from_cluster}:
            if c in seen:
                continue
            seen.add(c)
            for s2 in cl_segs[c] - {s}:
                total += subtree_length(s2, c, seen)
        return total

    seg_cur, cl_prev = root_seg, 0
    visited: set[int] = set()
    for _ in range(nseg + 1):
        nxt = seg_cls[seg_cur] - {cl_prev}
        if not nxt:
            return None
        c = min(nxt)
        if c in visited:
            return None
        visited.add(c)
        branches = cl_segs[c] - {seg_cur}
        lengths = {s: subtree_length(s, c, {c}) for s in branches}
        long = [s for s, L in lengths.items() if L >= min_branch_mm]
        if len(long) >= 2:
            zs = np.argwhere(clusters == c)[:, 2]
            return int(z0 + zs.max())
        if not lengths:
            return None
        seg_cur, cl_prev = max(lengths, key=lengths.get), c
    return None


# ---------------------------------------------------------------------------
# Airways below the carina
# ---------------------------------------------------------------------------

@dataclass
class AirwayResult:
    lumen_vox: int          # lumen voxels of the tree strictly below the carina slice
    wall_vox: int           # wall voxels attached to the tree (<= max_wall_mm from the lumen), below the carina
    branch_count: int       # skeleton segments below the carina (>= min_seg_vox voxels)
    terminal_count: int     # skeleton endpoints below the carina
    junction_count: int     # junction clusters below the carina
    cycle_rank: int         # loops in the whole skeleton graph (0 for a clean tree)
    skeleton_vox: int       # skeleton voxels below the carina
    prune_rounds: int       # spur-pruning rounds that removed something


def airway_metrics(lumen: np.ndarray, wall: np.ndarray, z_carina: int, spacing, *,
                   spur_mm: float = 3.0, max_wall_mm: float = 3.0, min_seg_vox: int = 2,
                   max_prune_rounds: int = 5) -> AirwayResult:
    """Lumen / wall volumes and skeleton branch statistics below the carina.

    The skeleton is computed on the WHOLE tree and cut afterwards (cutting
    first would create artificial endpoints at the cut). Wall voxels count
    only if they belong to the connected object containing the tree and lie
    within `max_wall_mm` of the lumen (a leak guard against the model
    painting neighbouring structures).
    """
    lumen = np.asarray(lumen).astype(bool, copy=False)
    wall = np.asarray(wall).astype(bool, copy=False)
    tree = largest_component(lumen, 26)
    if not tree.any():
        return AirwayResult(0, 0, 0, 0, 0, 0, 0, 0)
    sl = bbox_slices(tree | wall, pad=1)
    T = tree[sl]
    W = wall[sl]
    z0 = sl[2].start
    zc = z_carina - z0
    below = np.zeros(T.shape, dtype=bool)
    below[:, :, :max(0, min(zc, T.shape[2]))] = True

    # wall attached to the tree, leak-guarded
    att, n = ndimage.label(T | W, structure=STRUCT_26)
    tree_label = int(np.bincount(att[T].ravel()).argmax()) if n else 0
    W_tree = W & (att == tree_label)
    if W_tree.any():
        W_tree &= ndimage.distance_transform_edt(~T, sampling=spacing) <= max_wall_mm

    sk = skeletonize(ndimage.binary_fill_holes(T), method="lee").astype(bool)
    min_sp = float(min(spacing))
    prune_rounds = 0
    for _ in range(max_prune_rounds):
        nb = _skeleton_neighbours(sk)
        remove = (sk & (nb == 0))
        J = nb >= 3
        E = nb == 1
        seg, nseg = ndimage.label(sk & ~J, structure=STRUCT_26)
        if nseg:
            sizes = np.bincount(seg.ravel(), minlength=nseg + 1)
            leaf_ids = np.unique(seg[E])
            objs = ndimage.find_objects(seg)
            for sid in leaf_ids:
                if sid == 0:
                    continue
                if (sizes[sid] - 1) * min_sp >= spur_mm:
                    continue  # cannot be shorter than spur_mm
                obj = objs[sid - 1]
                local = seg[obj] == sid
                coords = np.argwhere(local)
                ends_local = np.argwhere(local & E[obj])
                start = int(np.flatnonzero((coords == ends_local[0]).all(axis=1))[0]) if len(ends_local) else 0
                if _geodesic_length_mm(coords, start, spacing) < spur_mm:
                    remove[obj] |= local
        if not remove.any():
            break
        sk &= ~remove
        prune_rounds += 1

    nb = _skeleton_neighbours(sk)
    J = nb >= 3
    E = nb == 1
    seg, nseg = ndimage.label(sk & ~J & below, structure=STRUCT_26)
    branch_count = int((np.bincount(seg.ravel())[1:] >= min_seg_vox).sum()) if nseg else 0
    terminal_count = int((E & below).sum())
    _, n_j_below = ndimage.label(J & below, structure=STRUCT_26)
    _, n_edges = ndimage.label(sk & ~J, structure=STRUCT_26)
    _, n_jall = ndimage.label(J, structure=STRUCT_26)
    _, n_comp = ndimage.label(sk, structure=STRUCT_26)
    cycle_rank = max(0, int(n_edges) - (int(n_jall) + int(E.sum())) + int(n_comp))
    return AirwayResult(
        lumen_vox=int((T & below).sum()),
        wall_vox=int((W_tree & below).sum()),
        branch_count=branch_count,
        terminal_count=terminal_count,
        junction_count=int(n_j_below),
        cycle_rank=cycle_rank,
        skeleton_vox=int((sk & below).sum()),
        prune_rounds=prune_rounds,
    )


# ---------------------------------------------------------------------------
# Vessels: volumes and BV5 (small-vessel volume)
# ---------------------------------------------------------------------------

@dataclass
class VesselResult:
    artery_vox: int
    vein_vox: int
    small_vox: int               # BV5 voxels (artery + vein) using the half-voxel-corrected radius
    small_vox_uncorrected: int   # same without the correction (sensitivity check)
    skeleton_vox: int
    skeleton_fragments: int      # 26-connected components of the vessel skeleton (fragmentation indicator)
    radius_p50_mm: float         # median local radius along the skeleton (corrected)
    radius_p90_mm: float


def _nearest_skeleton_values(V: np.ndarray, sk: np.ndarray, value_vols: list[np.ndarray], spacing,
                             max_block_voxels: int, halo_mm: float) -> list[np.ndarray]:
    """For every True voxel of V, the value(s) stored at its nearest skeleton voxel.

    Exact nearest-skeleton assignment via distance_transform_edt(return_indices)
    on z-blocks with a halo (bounded memory: the index array alone is 12 B/voxel).
    """
    outs = [np.zeros(V.shape, dtype=np.float32) for _ in value_vols]
    nz = V.shape[2]
    plane = V.shape[0] * V.shape[1]
    step = nz if V.size <= max_block_voxels else max(1, int(max_block_voxels // plane))
    halo = int(math.ceil(halo_mm / float(spacing[2])))
    for a in range(0, nz, step):
        b = min(a + step, nz)
        if not V[:, :, a:b].any():
            continue
        lo, hi = max(0, a - halo), min(nz, b + halo)
        skb = sk[:, :, lo:hi]
        if not skb.any():
            continue
        idx = ndimage.distance_transform_edt(~skb, sampling=spacing, return_distances=False,
                                             return_indices=True)
        core = slice(a - lo, b - lo)
        sel = V[:, :, a:b]
        for out, vol in zip(outs, value_vols):
            mapped = vol[:, :, lo:hi][idx[0], idx[1], idx[2]][:, :, core]
            out[:, :, a:b] = np.where(sel, mapped, 0)
    return outs


def vessel_metrics(lung_vessels: np.ndarray, lobes: np.ndarray, spacing, *, artery_label: int = 3,
                   vein_label: int = 4, bv_area_mm2: float = BV_AREA_MM2,
                   max_block_voxels: int = 40_000_000, halo_mm: float = 15.0) -> VesselResult:
    """Artery / vein volumes inside the lungs and the BV5 small-vessel volume.

    Local radius = Euclidean distance transform at the nearest skeleton voxel
    minus half an in-plane voxel (the EDT measures centre-to-centre, the
    physical boundary lies half a voxel closer). A voxel is "small vessel" if
    pi * r^2 < bv_area_mm2. Vessels are restricted to the lung-lobe labels,
    i.e. hilar / mediastinal portions are excluded.
    """
    lv = np.asarray(lung_vessels)
    lobes = np.asarray(lobes).astype(bool, copy=False)
    A = (lv == artery_label) & lobes
    Vn = (lv == vein_label) & lobes
    V = A | Vn
    res = VesselResult(int(A.sum()), int(Vn.sum()), 0, 0, 0, 0, float("nan"), float("nan"))
    if not V.any():
        return res
    sl = bbox_slices(V, pad=2)
    Vc = V[sl]
    edt = ndimage.distance_transform_edt(Vc, sampling=spacing).astype(np.float32)
    sk = skeletonize(Vc, method="lee").astype(bool)
    if not sk.any():
        return res
    half = 0.5 * float(min(spacing[0], spacing[1]))
    r_unc = np.zeros(Vc.shape, dtype=np.float32)
    r_unc[sk] = edt[sk]
    r_cor = np.zeros(Vc.shape, dtype=np.float32)
    r_cor[sk] = np.maximum(edt[sk] - half, 0.1)
    r_map_cor, r_map_unc = _nearest_skeleton_values(Vc, sk, [r_cor, r_unc], spacing,
                                                    max_block_voxels, halo_mm)
    small = Vc & (math.pi * r_map_cor ** 2 < bv_area_mm2)
    small_unc = Vc & (math.pi * r_map_unc ** 2 < bv_area_mm2)
    _, n_frag = ndimage.label(sk, structure=STRUCT_26)
    rs = r_cor[sk]
    res.small_vox = int(small.sum())
    res.small_vox_uncorrected = int(small_unc.sum())
    res.skeleton_vox = int(sk.sum())
    res.skeleton_fragments = int(n_frag)
    res.radius_p50_mm = float(np.percentile(rs, 50))
    res.radius_p90_mm = float(np.percentile(rs, 90))
    return res


# ---------------------------------------------------------------------------
# Parenchymal density
# ---------------------------------------------------------------------------

@dataclass
class DensityResult:
    n_vox: int
    mean_hu: float
    laa_pct: float      # % of parenchyma voxels below laa_thr (default -950 HU)
    p15_hu: float       # 15th percentile HU


def parenchyma_density(ct: np.ndarray, lungs_clean: np.ndarray, exclude: np.ndarray, spacing, *,
                       erode_mm: float = 2.0, laa_thr: float = -950.0) -> DensityResult:
    """Mean HU and LAA% inside the lungs minus `exclude` (vessels + airways), eroded by erode_mm.

    Erosion uses the spacing-aware Euclidean distance transform (> erode_mm from
    any excluded / non-lung voxel), so slice thickness is respected. `ct` must
    be on the same canonical grid as the masks.
    """
    P0 = np.asarray(lungs_clean).astype(bool, copy=False) & ~np.asarray(exclude).astype(bool, copy=False)
    nan = float("nan")
    if not P0.any():
        return DensityResult(0, nan, nan, nan)
    sl = bbox_slices(P0, pad=1)
    P = ndimage.distance_transform_edt(P0[sl], sampling=spacing) > erode_mm
    n = int(P.sum())
    if n == 0:
        return DensityResult(0, nan, nan, nan)
    hu = np.asarray(ct[sl])[P].astype(np.float32)
    return DensityResult(n, float(hu.mean()), float(100.0 * (hu < laa_thr).mean()),
                         float(np.percentile(hu, 15)))


# ---------------------------------------------------------------------------
# Pleural effusion: right / left split at the spine midline
# ---------------------------------------------------------------------------

def _per_slice_centroid_x(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(valid, x_centroid) per axial slice of a boolean mask."""
    nz = mask.shape[2]
    xs, _, zs = np.nonzero(mask)
    cnt = np.bincount(zs, minlength=nz).astype(float)
    sx = np.bincount(zs, weights=xs, minlength=nz)
    valid = cnt > 0
    cx = np.zeros(nz)
    cx[valid] = sx[valid] / cnt[valid]
    return valid, cx


def spine_midline(vert_body: np.ndarray, *, lungs_R: np.ndarray | None = None,
                  lungs_L: np.ndarray | None = None) -> tuple[np.ndarray, str]:
    """Per-slice x position of the spine midline (float, full nz length) and a note.

    Uses the x-centroid of the vertebral-body mask on each slice, linearly
    interpolated across slices without a vertebra (disc gaps) and clamped
    beyond the column. Falls back to the midpoint of the two lung centroids,
    then to the volume centre.
    """
    nz = vert_body.shape[2]
    valid, cx = _per_slice_centroid_x(np.asarray(vert_body).astype(bool, copy=False))
    if valid.any():
        return np.interp(np.arange(nz), np.flatnonzero(valid), cx[valid]), ""
    if lungs_R is not None and lungs_L is not None:
        vr, cr = _per_slice_centroid_x(np.asarray(lungs_R).astype(bool, copy=False))
        vl, cl = _per_slice_centroid_x(np.asarray(lungs_L).astype(bool, copy=False))
        both = vr & vl
        if both.any():
            mid = (cr + cl) / 2.0
            return np.interp(np.arange(nz), np.flatnonzero(both), mid[both]), "effusion split without vertebrae (lung midline)"
    return np.full(nz, (vert_body.shape[0] - 1) / 2.0), "effusion split without vertebrae (volume centre)"


def split_effusion_lr(effusion: np.ndarray, vert_body: np.ndarray, *, lungs_R: np.ndarray | None = None,
                      lungs_L: np.ndarray | None = None) -> tuple[int, int, str]:
    """(right voxels, left voxels, note). Canonical RAS: larger x index = patient's RIGHT.
    Voxels exactly on the midline count as left. right + left == effusion.sum()."""
    eff = np.asarray(effusion).astype(bool, copy=False)
    mid, note = spine_midline(vert_body, lungs_R=lungs_R, lungs_L=lungs_L)
    x_idx = np.arange(eff.shape[0])[:, None, None]
    right = eff & (x_idx > mid[None, None, :])
    n_r = int(right.sum())
    return n_r, int(eff.sum()) - n_r, note
