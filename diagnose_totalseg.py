#!/usr/bin/env python
"""Phase-1 diagnostic: run every TotalSegmentator task of the extended pipeline
ONCE on ONE case, time each task, and check the resulting masks.

The printed output is identifier-free by construction (the case is always
called "case"; patient / study tags, UIDs, descriptions and the input path
are never printed and are scrubbed from any library output), so the whole
output can be pasted back. The --out folder itself receives the CT and the
masks of this case: it is patient data, keep it with the data.

Usage (MacBook, Apple GPU):
  uv run python diagnose_totalseg.py --dicom "<folder with this case's DICOMs>" --out ./diag_case1
  uv run python diagnose_totalseg.py --dicom "<flat export folder>" --patient-id "<PatientID>" --out ./diag_case1
Windows / 4090:
  uv run python diagnose_totalseg.py --dicom "<folder>" --out .\\diag_case1 --device gpu

Options:
  --fast-too        also time the 3 mm "fast" total model
  --variants        also time resampling variants (resampling_order=3, higher_order_resampling, total without trachea)
  --tasks a,b,c     subset of: total vertebrae_pp_refined trunk_cavities lung_vessels pleural_pericard_effusion
  --skip-inference  do not run TotalSegmentator; analyse the masks already in --out
  --json FILE       also write the identifier-free summary as JSON
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import math
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
from scipy import ndimage

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import pipeline as pl  # noqa: E402
import thorax_metrics as tm  # noqa: E402

ALL_TASKS = ["total", "vertebrae_pp_refined", "trunk_cavities", "lung_vessels", "pleural_pericard_effusion"]

# dataset ids to pre-download (untimed) so the runtime table measures inference only
WEIGHTS_FOR_TASK = {
    "total": [297, 291, 292, 293, 294, 295],
    "total_fast": [297],
    "vertebrae_pp_refined": [803, 305],
    "trunk_cavities": [343],
    "lung_vessels": [297, 117],
    "pleural_pericard_effusion": [297, 315],
}

ALLOWED_TAGS = [
    "Manufacturer", "ManufacturerModelName", "KVP", "SliceThickness", "SpacingBetweenSlices",
    "PixelSpacing", "ConvolutionKernel", "RescaleSlope", "RescaleIntercept", "RescaleType",
    "BitsStored", "PixelRepresentation", "PatientPosition", "ImageOrientationPatient",
    "Rows", "Columns", "NumberOfFrames", "ReconstructionDiameter", "Modality",
]
IDENT_TAGS = [
    "PatientID", "PatientName", "OtherPatientIDs", "PatientBirthDate", "AccessionNumber",
    "StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate", "StudyTime", "StudyID",
    "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID", "FrameOfReferenceUID",
    "InstitutionName", "InstitutionAddress", "ReferringPhysicianName", "PerformingPhysicianName",
    "OperatorsName", "StationName", "StudyDescription", "SeriesDescription", "ProtocolName",
    "DeviceSerialNumber", "RequestingPhysician", "PatientAddress",
]
GENERIC_PATH_PARTS = {"users", "volumes", "documents", "desktop", "home", "downloads", "mnt", "media"}

LEVEL_SIDES = ("R", "L", "both")


# ---------------------------------------------------------------------------
# Identifier-free printing
# ---------------------------------------------------------------------------

class Redactor:
    def __init__(self) -> None:
        self.secrets: list[str] = []

    def add(self, value) -> None:
        s = str(value).strip()
        if len(s) < 3 or s in self.secrets:
            return
        self.secrets.append(s)
        self.secrets.sort(key=len, reverse=True)

    def add_path(self, p: Path, last_component_always: bool = False) -> None:
        """Redact the path as a whole plus any component that looks like an
        identifier (contains a digit, or is a long name such as a person's
        name). Plain words like "output" are left alone so normal text stays
        readable; the last component (a case folder = often the MRN) is always
        redacted when asked."""
        self.add(str(p))
        self.add(str(p.resolve()))
        parts = p.resolve().parts
        for i, part in enumerate(parts):
            if part == p.anchor or part.lower() in GENERIC_PATH_PARTS:
                continue
            looks_like_id = any(ch.isdigit() for ch in part) or len(part) >= 12
            if looks_like_id or (last_component_always and i == len(parts) - 1):
                self.add(part)

    def __call__(self, text: str) -> str:
        for s in self.secrets:
            if s in text:
                text = text.replace(s, "<redacted>")
        return text


RED = Redactor()


def say(*parts) -> None:
    print(RED(" ".join(str(p) for p in parts)), flush=True)


class RedactingStream(io.TextIOBase):
    def __init__(self, target) -> None:
        self.target = target

    def write(self, s: str) -> int:  # type: ignore[override]
        self.target.write(RED(s))
        return len(s)

    def flush(self) -> None:
        self.target.flush()


@contextlib.contextmanager
def redacted_output():
    out, err = RedactingStream(sys.__stdout__), RedactingStream(sys.__stderr__)
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        yield


@contextlib.contextmanager
def quiet_output():
    """Discard library output entirely (download progress bars)."""
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        yield


def section(title: str) -> None:
    say(f"\n===== {title} =====")


def safe(fn, label: str, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001
        say(f"[{label} FAILED: {type(e).__name__}: {str(e)[:300]}]")
        say(RED("".join(traceback.format_exception_only(type(e), e))).strip())
        return None


def fmt(v, nd=2):
    if v is None:
        return "-"
    if isinstance(v, float):
        return "-" if math.isnan(v) else f"{v:.{nd}f}"
    return str(v)


def peak_rss_mb() -> float:
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 1e6 if sys.platform == "darwin" else r / 1e3
    except ImportError:
        try:
            import psutil
            return psutil.Process().memory_info().peak_wset / 1e6
        except Exception:  # noqa: BLE001
            return float("nan")


class GpuSampler:
    """Samples nvidia-smi utilisation / memory every 0.5 s in a background thread."""

    def __init__(self) -> None:
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if shutil.which("nvidia-smi") is None:
            return

        def loop() -> None:
            while not self._stop.is_set():
                try:
                    out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                          "--format=csv,noheader,nounits"],
                                         capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
                    u, m = out[0].split(",")
                    self.samples.append((float(u), float(m)))
                except Exception:  # noqa: BLE001
                    pass
                self._stop.wait(0.5)

        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def stop(self) -> tuple[float, float, float]:
        """(mean utilisation %, max utilisation %, max memory used GB); nan without nvidia-smi."""
        self._stop.set()
        if self.thread is not None:
            self.thread.join(timeout=3)
        if not self.samples:
            return float("nan"), float("nan"), float("nan")
        u = [x[0] for x in self.samples]
        m = [x[1] for x in self.samples]
        return float(np.mean(u)), float(max(u)), float(max(m) / 1024)


def resolve_device(arg: str) -> str:
    import torch
    if arg == "auto":
        if torch.cuda.is_available():
            return "gpu"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if arg.startswith("gpu") and not torch.cuda.is_available():
        sys.exit("error: --device gpu requested but CUDA is not available (torch.cuda.is_available() is False)")
    if arg == "mps" and not torch.backends.mps.is_available():
        sys.exit("error: --device mps requested but MPS is not available")
    return arg


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def section_environment(device_arg: str, device: str) -> dict:
    import importlib.metadata as md
    import torch

    info: dict = {}
    section("ENVIRONMENT")
    say(f"OS: {platform.system()} {platform.release()} ({platform.machine()}) | Python {platform.python_version()}")
    for pkg in ("TotalSegmentator", "nnunetv2", "torch", "numpy", "scipy", "scikit-image", "nibabel", "SimpleITK", "pydicom"):
        try:
            info[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            info[pkg] = "not installed"
    say("versions: " + ", ".join(f"{k} {v}" for k, v in info.items()))
    cuda = torch.cuda.is_available()
    say(f"CUDA available: {cuda}" + (f" ({torch.cuda.get_device_name(0)}, torch cuda {torch.version.cuda})" if cuda else ""))
    say(f"MPS available: {torch.backends.mps.is_available()}")
    say(f"device requested: {device_arg} -> resolved: {device}")
    try:
        import psutil
        say(f"RAM: {psutil.virtual_memory().total / 1e9:.1f} GB total | CPUs: {psutil.cpu_count()}")
        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:  # noqa: BLE001
        say("RAM: (psutil not available)")
    info.update(cuda=cuda, mps=torch.backends.mps.is_available(), device=device)

    try:
        from totalsegmentator.config import get_config, get_weights_dir
        wdir = Path(get_weights_dir())
        say(f"weights dir: {wdir}")
        folds: dict[str, int] = {}
        for ds in sorted(p for p in wdir.glob("Dataset*") if p.is_dir()):
            n = sum(1 for tr in ds.iterdir() if tr.is_dir() for f in tr.glob("fold_*") if f.is_dir())
            folds[ds.name] = n
            say(f"  {ds.name}: {n} fold(s)")
        if not folds:
            say("  (no weights downloaded yet)")
        info["weights_folds"] = folds
        try:
            cfg = get_config()
            say(f"config: send_usage_stats={cfg.get('send_usage_stats')} | license set: {'license_number' in cfg}")
            info["send_usage_stats"] = cfg.get("send_usage_stats")
        except Exception as e:  # noqa: BLE001
            say(f"config: unreadable ({type(e).__name__})")
    except Exception as e:  # noqa: BLE001
        say(f"weights/config: {type(e).__name__}: {e}")
    return info


def section_registry(tasks: list[str]) -> dict:
    section("TASK REGISTRY (from the installed TotalSegmentator)")
    from totalsegmentator.map_tasks_config import TASK_CONFIGS
    from totalsegmentator.registry import get_task_classes

    out: dict = {}
    lobes, verts = pl.get_label_maps()
    for t in tasks + ["vertebrae_pp", "vertebrae_body"]:
        cfg = TASK_CONFIGS.get(t, {})
        fast = "sub_modes" in cfg
        base = cfg["sub_modes"]["default"] if fast else cfg
        classes = get_task_classes(t)
        line = (f"{t}: fast={'yes' if fast else 'no'}{' (fast raises)' if cfg.get('disallow_fast') else ''}"
                f" | resample={base.get('resample')} | task_id={base.get('task_id')} | trainer={base.get('trainer')}"
                f" | crop={'lungs' if cfg.get('crop') else None} | crop_addon={cfg.get('crop_addon', [3, 3, 3])}"
                f" | folds={cfg.get('folds', [0])} | robust_crop_forced={cfg.get('robust_crop', False)}"
                f" | n_classes={len(classes)}")
        say(line)
        if t == "total":
            keep = {i: n for i, n in classes.items() if n in lobes or n in verts or n in ("trachea", "sacrum")}
            say("   classes used: " + ", ".join(f"{i}:{n}" for i, n in sorted(keep.items())))
        else:
            say("   classes: " + ", ".join(f"{i}:{n}" for i, n in sorted(classes.items())))
        out[t] = {"fast": fast, "resample": base.get("resample"), "task_id": base.get("task_id"),
                  "classes": {str(i): n for i, n in classes.items()}}
    return out


def prepare_ct(args, out: Path) -> tuple[Path, dict]:
    """Convert the selected DICOM series (or copy the given NIfTI) to out/case.nii.gz."""
    section("CT")
    ct_path = out / "case.nii.gz"
    tags: dict = {}
    if args.ct:
        RED.add_path(args.ct)
        if ct_path.resolve() != args.ct.resolve():
            shutil.copy2(args.ct, ct_path)
        say("input: NIfTI (copied to case.nii.gz); no DICOM tags available")
    elif args.dicom:
        RED.add_path(args.dicom, last_component_always=True)
        if args.patient_id:
            RED.add(args.patient_id)
            pool = {u: s for u, s in pl.scan_series(args.dicom).items() if s.patient_id == args.patient_id}
            if not pool:
                sys.exit("error: --patient-id matched no series in that folder")
            series = pl.select_series(pool, args.min_slices)
        else:
            series = pl.select_series(args.dicom, args.min_slices)
        ds = series.first_ds
        for tag in IDENT_TAGS:
            v = getattr(ds, tag, None)
            if v:
                RED.add(v)
                for piece in str(v).replace("^", " ").split():
                    RED.add(piece)
        for f in series.files[:1]:
            RED.add_path(f.parent)
        say(f"series: {series.n_slices} slices in {len(series.files)} files | modality {series.modality or '?'} | axial={series.is_axial}")
        for tag in ALLOWED_TAGS:
            v = getattr(ds, tag, None)
            if v is not None:
                tags[tag] = str(v)
        say("DICOM tags: " + " | ".join(f"{k}={v}" for k, v in tags.items()))
        files = pl.sort_series_files(series)
        t0 = time.monotonic()
        pl.convert_series_to_nifti(files, ct_path)
        pl.verify_geometry(ct_path, files)
        say(f"converted + geometry verified in {time.monotonic() - t0:.1f} s")
    elif not ct_path.exists():
        sys.exit("error: give --dicom or --ct (or use --skip-inference with an existing case.nii.gz in --out)")

    import nibabel as nib
    img = nib.load(ct_path)
    hdr = img.header
    data = np.asanyarray(img.dataobj)
    say(f"case.nii.gz: shape={img.shape} zooms={tuple(round(float(z), 3) for z in hdr.get_zooms()[:3])} mm"
        f" dtype={data.dtype} axcodes={''.join(nib.aff2axcodes(img.affine))}"
        f" scl_slope={hdr['scl_slope']} scl_inter={hdr['scl_inter']}")
    pct = np.percentile(data, [0.5, 5, 25, 50, 75, 95, 99.5])
    say("HU percentiles 0.5/5/25/50/75/95/99.5: " + " / ".join(f"{p:.0f}" for p in pct)
        + f" | min={data.min()} max={data.max()} | fraction < -900: {(data < -900).mean():.3f}")
    tags["_shape"] = list(img.shape)
    tags["_zooms"] = [round(float(z), 4) for z in hdr.get_zooms()[:3]]
    tags["_hu_percentiles"] = [round(float(p), 1) for p in pct]
    return ct_path, tags


def build_runs(args, roi: list[str], roi_no_trachea: list[str], cuda: bool) -> list[tuple[str, dict]]:
    base = {
        "total": dict(task="total", roi_subset=roi),
        "vertebrae_pp_refined": dict(task="vertebrae_pp_refined"),
        "trunk_cavities": dict(task="trunk_cavities"),
        "lung_vessels": dict(task="lung_vessels"),
        "pleural_pericard_effusion": dict(task="pleural_pericard_effusion"),
    }
    runs = [(t, base[t]) for t in ALL_TASKS if t in args.tasks]
    if args.fast_too:
        runs.append(("total_fast", dict(task="total", roi_subset=roi, fast=True)))
    if args.variants:
        if "total" in args.tasks:
            runs.append(("total_ro3", dict(task="total", roi_subset=roi, resampling_order=3)))
            runs.append(("total_no_trachea", dict(task="total", roi_subset=roi_no_trachea)))
            if cuda:
                runs.append(("total_ho", dict(task="total", roi_subset=roi, higher_order_resampling=True)))
            else:
                say("(skipping total with higher_order_resampling on a non-CUDA device: logits at input resolution "
                    "x 25 classes would need many GB of RAM)")
        if "lung_vessels" in args.tasks:
            runs.append(("lung_vessels_ro3", dict(task="lung_vessels", resampling_order=3)))
            runs.append(("lung_vessels_ho", dict(task="lung_vessels", higher_order_resampling=True)))
    return runs


def download_weights(runs: list[tuple[str, dict]]) -> None:
    section("WEIGHTS (download once, untimed)")
    try:
        from totalsegmentator.libs import download_pretrained_weights
    except Exception as e:  # noqa: BLE001
        say(f"cannot import download_pretrained_weights: {type(e).__name__}: {e}")
        return
    ids: list[int] = []
    for name, kw in runs:
        key = "total_fast" if kw.get("fast") else kw["task"]
        for i in WEIGHTS_FOR_TASK.get(key, []):
            if i not in ids:
                ids.append(i)
    for i in ids:
        t0 = time.monotonic()
        try:
            with quiet_output():
                download_pretrained_weights(i)
            say(f"  dataset {i}: ready ({time.monotonic() - t0:.0f} s)")
        except Exception as e:  # noqa: BLE001
            say(f"  dataset {i}: DOWNLOAD FAILED ({type(e).__name__}: {str(e)[:120]}) -- offline? the timed run will retry")


def run_tasks(runs: list[tuple[str, dict]], ct_path: Path, out: Path, device: str, skip: bool,
              nr_thr_saving: int = 6, nr_thr_resamp: int = 1) -> list[dict]:
    section("RUNTIME PER TASK" + (" (skip-inference: reading existing masks)" if skip else ""))
    from totalsegmentator.python_api import totalsegmentator
    say(f"  settings: nr_thr_saving={nr_thr_saving} (nnU-Net export worker processes per model call) | "
        f"nr_thr_resamp={nr_thr_resamp} | robust_crop=True")

    rows: list[dict] = []
    for name, kw in runs:
        mask = out / f"case_{name}.nii.gz"
        report = out / f"case_{name}.report.json"
        row = dict(name=name, task=kw["task"], device=device, seconds=float("nan"), ok=False,
                   retried_cpu=False, peak_rss_mb=float("nan"), error="", fast=bool(kw.get("fast", False)),
                   resampling_order=kw.get("resampling_order", 1),
                   higher_order_resampling=bool(kw.get("higher_order_resampling", False)),
                   nr_thr_saving=nr_thr_saving, nr_thr_resamp=nr_thr_resamp,
                   gpu_util_mean=float("nan"), gpu_util_max=float("nan"), gpu_mem_gb=float("nan"),
                   torch_gpu_peak_gb=float("nan"))
        if skip:
            row["ok"] = mask.exists()
            if report.exists():
                try:
                    rep = json.loads(report.read_text())
                    row.update(seconds=float(rep.get("runtime_seconds", float("nan"))), device=rep.get("device", "?"))
                except Exception:  # noqa: BLE001
                    pass
            rows.append(row)
            say(f"  {name:28s} {'mask present' if row['ok'] else 'MISSING'} | {fmt(row['seconds'], 1)} s (from report)")
            continue
        devices = [device] + (["cpu"] if device == "mps" else [])
        for dev in devices:
            mask.unlink(missing_ok=True)
            say(f"  running {name} on {dev} ...")
            sampler = GpuSampler()
            if dev.startswith("gpu"):
                sampler.start()
                try:
                    import torch
                    torch.cuda.reset_peak_memory_stats()
                except Exception:  # noqa: BLE001
                    pass
            t0 = time.monotonic()
            try:
                with redacted_output():
                    totalsegmentator(
                        input=str(ct_path), output=str(mask), task=kw["task"], ml=True, quiet=True,
                        robust_crop=True, device=dev, report=str(report),
                        fast=bool(kw.get("fast", False)), roi_subset=kw.get("roi_subset"),
                        resampling_order=kw.get("resampling_order", 1),
                        higher_order_resampling=bool(kw.get("higher_order_resampling", False)),
                        nr_thr_saving=nr_thr_saving, nr_thr_resamp=nr_thr_resamp,
                    )
                if not mask.exists():
                    raise RuntimeError("TotalSegmentator finished but produced no output file")
                row.update(seconds=round(time.monotonic() - t0, 1), ok=True, device=dev, retried_cpu=(dev != device), error="")
                um, ux, mg = sampler.stop()
                row.update(gpu_util_mean=um, gpu_util_max=ux, gpu_mem_gb=mg)
                if dev.startswith("gpu"):
                    try:
                        import torch
                        row["torch_gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
                    except Exception:  # noqa: BLE001
                        pass
                break
            except Exception as e:  # noqa: BLE001
                sampler.stop()
                row["error"] = f"{type(e).__name__}: {str(e)[:160]}"
                row["seconds"] = round(time.monotonic() - t0, 1)
                say(f"  {name} FAILED on {dev} after {row['seconds']} s: {row['error']}")
                if dev != devices[-1]:
                    say("  -> retrying on cpu")
                    try:
                        import gc
                        import torch
                        gc.collect()
                        if hasattr(torch, "mps"):
                            torch.mps.empty_cache()
                    except Exception:  # noqa: BLE001
                        pass
        row["peak_rss_mb"] = peak_rss_mb()
        if report.exists():
            try:
                rep = json.loads(report.read_text())
                row["report_fast"] = rep.get("fast")
                row["report_version"] = rep.get("totalsegmentator_version")
                row["report_device"] = rep.get("device")
            except Exception:  # noqa: BLE001
                pass
        rows.append(row)
        status = "ok" if row["ok"] else "FAILED"
        say(f"  {name:28s} {status:6s} {fmt(row['seconds'], 1):>8s} s = {fmt(row['seconds'] / 60 if row['ok'] else float('nan'), 1):>5s} min"
            f" | device {row['device']}{' (cpu retry)' if row['retried_cpu'] else ''} | peak RSS so far {fmt(row['peak_rss_mb'], 0)} MB"
            + (f" | GPU util mean {fmt(row['gpu_util_mean'], 0)} % max {fmt(row['gpu_util_max'], 0) } % | GPU mem max {fmt(row['gpu_mem_gb'], 1)} GB"
               f" | torch peak {fmt(row['torch_gpu_peak_gb'], 1)} GB" if not math.isnan(row['gpu_util_mean']) else ""))
    return rows


def print_runtime_table(rows: list[dict]) -> None:
    section("RUNTIME TABLE")
    say(f"{'run':28s} {'task':26s} {'device':8s} {'seconds':>8s} {'min':>6s} {'ro':>3s} {'ho':>3s} {'fast':>5s} {'nts':>4s} {'peakRSS_MB':>10s} {'gpu%mean':>8s} {'gpu%max':>7s} {'gpuGB':>6s}  note")
    for r in rows:
        note = r.get("error", "") or ("cpu retry" if r.get("retried_cpu") else "")
        say(f"{r['name']:28s} {r['task']:26s} {str(r['device']):8s} {fmt(r['seconds'], 1):>8s} "
            f"{fmt(r['seconds'] / 60 if r['ok'] and not math.isnan(r['seconds']) else float('nan'), 1):>6s} "
            f"{r['resampling_order']:>3d} {str(r['higher_order_resampling'])[0]:>3s} {str(r['fast'])[0]:>5s} {r.get('nr_thr_saving', 6):>4d} "
            f"{fmt(r['peak_rss_mb'], 0):>10s} {fmt(r.get('gpu_util_mean', float('nan')), 0):>8s} "
            f"{fmt(r.get('gpu_util_max', float('nan')), 0):>7s} {fmt(r.get('gpu_mem_gb', float('nan')), 1):>6s}  {note}")
    say("(peak RSS is the process high-water mark up to that task, not a per-task figure; gpu% = nvidia-smi utilisation "
        "sampled every 0.5 s during the task, mean and max; gpuGB = max GPU memory in use)")


# ---------------------------------------------------------------------------
# Mask analysis
# ---------------------------------------------------------------------------

def per_slice_vertebrae(data: np.ndarray, id_to_name: dict[int, str]) -> list[list[str]]:
    out: list[list[str]] = []
    for z in range(data.shape[2]):
        counts = np.bincount(data[:, :, z].ravel())
        out.append([id_to_name[i] for i in np.flatnonzero(counts) if i in id_to_name])
    return out


def levels_from(per_slice: list[list[str]], extremes: dict[str, tuple[int, int] | None]) -> dict[str, str]:
    res: dict[str, str] = {}
    for side in LEVEL_SIDES:
        ext = extremes.get(side)
        if ext is None:
            res[f"start_{side}"] = res[f"end_{side}"] = ""
            continue
        z_min, z_max = ext
        res[f"start_{side}"] = pl.short_name(pl._vert_level(per_slice, z_max, step=-1, pick_cranial=True))
        res[f"end_{side}"] = pl.short_name(pl._vert_level(per_slice, z_min, step=+1, pick_cranial=False))
    return res


def analyze(ct_path: Path, mask_paths: dict[str, Path]) -> dict:
    import nibabel as nib
    from totalsegmentator.map_to_binary import class_map

    summary: dict = {}
    section("MASK CHECKS")
    ct_img = nib.load(ct_path)
    loaded: dict[str, object] = {}
    for name, p in mask_paths.items():
        if not p.exists():
            say(f"  {name}: mask missing -> skipped")
            continue
        img = nib.load(p)
        same = img.shape[:3] == ct_img.shape[:3] and np.allclose(img.affine, ct_img.affine, atol=1e-3)
        say(f"  {name}: shape={img.shape} same grid as CT: {same}")
        summary[f"{name}_same_grid"] = bool(same)
        loaded[name] = img
    if "total" not in loaded:
        say("no total mask: cannot analyse further")
        return summary

    ct_c = nib.as_closest_canonical(ct_img)
    spacing = tuple(float(z) for z in ct_c.header.get_zooms()[:3])
    vox_ml = tm.voxel_ml(spacing)
    nz = ct_c.shape[2]

    def canon(name: str, dtype=np.uint8) -> np.ndarray:
        return np.asanyarray(nib.as_closest_canonical(loaded[name]).dataobj).astype(dtype)

    total = canon("total", np.int16)
    lobes, verts = pl.get_label_maps()
    total_map = class_map["total"]
    trachea_id = {n: i for i, n in total_map.items()}.get("trachea")
    right_ids = [lobes[n] for n in pl.RIGHT_LOBES]
    left_ids = [lobes[n] for n in pl.LEFT_LOBES]
    lobe_ids = right_ids + left_ids

    # --- lungs (pipeline cleaning rules) ---
    section("LUNGS (total)")
    counts = np.bincount(total.ravel(), minlength=max(total_map) + 1)
    say("  voxels per lobe: " + ", ".join(f"{n}={counts[lobes[n]]}" for n in pl.LUNG_LOBES)
        + (f", trachea={counts[trachea_id]}" if trachea_id is not None else ""))
    lung_R = pl._largest_component(np.isin(total, right_ids))
    lung_L = pl._largest_component(np.isin(total, left_ids))
    lung_both = lung_R | lung_L
    extremes = {"R": pl._z_extremes(lung_R), "L": pl._z_extremes(lung_L), "both": pl._z_extremes(lung_both)}
    for side, ext in extremes.items():
        if ext:
            say(f"  {side}: base z={ext[0]} apex z={ext[1]} height={(ext[1] - ext[0]) * spacing[2]:.1f} mm")
    faces = tm.edge_contact(lung_both)
    say(f"  lung touches scan edge: {faces if faces else 'no'}")
    summary["lung_edge_faces"] = faces
    vol_both_ml = float(lung_both.sum()) * vox_ml
    say(f"  vol_both = {vol_both_ml:.1f} ml")

    # --- HU sanity ---
    section("CT STORED IN HU?")
    ct = np.asanyarray(ct_c.dataobj)
    lobe_mask = np.isin(total, lobe_ids)
    vert_ids_total = list(verts.values())
    vert_mask_total = np.isin(total, vert_ids_total)
    hu_lung = float(ct[lobe_mask].mean()) if lobe_mask.any() else float("nan")
    hu_vert = float(ct[vert_mask_total].mean()) if vert_mask_total.any() else float("nan")
    hu_ok = (-950 <= hu_lung <= -600) and (hu_vert > 100)
    say(f"  mean HU inside lung lobes: {hu_lung:.0f} (expect ~ -700..-900) | inside vertebrae: {hu_vert:.0f} (expect > 100)")
    if not lobe_mask.any() or not vert_mask_total.any():
        say("  verdict: cannot judge (no lung / vertebra voxels segmented)")
    else:
        say(f"  verdict: {'HU, yes' if hu_ok else 'NOT plausible HU -> check RescaleIntercept / conversion'}")
    summary.update(hu_lung_mean=round(hu_lung, 1), hu_vert_mean=round(hu_vert, 1), hu_plausible=bool(hu_ok))

    # --- vertebrae: total vs pp ---
    section("VERTEBRAE: total vs vertebrae_pp_refined (bodies only?)")
    pp = canon("vertebrae_pp_refined") if "vertebrae_pp_refined" in loaded else None
    pp_map = class_map["vertebrae_pp_refined"]
    per_slice_total = per_slice_vertebrae(total, {i: n for n, i in verts.items()})
    per_slice_pp = per_slice_vertebrae(pp, dict(pp_map)) if pp is not None else None

    def level_stats(data: np.ndarray, name_to_id: dict[str, int]):
        st = {}
        objs = ndimage.find_objects(data)
        for name, idx in name_to_id.items():
            if idx - 1 < len(objs) and objs[idx - 1] is not None:
                sl = objs[idx - 1]
                n = int((data[sl] == idx).sum())
                st[name] = dict(z=(sl[2].start, sl[2].stop - 1), y=(sl[1].start, sl[1].stop - 1), n=n)
        return st

    st_total = level_stats(total, verts)
    st_pp = level_stats(pp, {n: i for i, n in pp_map.items()}) if pp is not None else {}
    ratios, ap_ratios = [], []
    say(f"  {'level':5s} {'total z':>11s} {'pp z':>11s} {'total vox':>10s} {'pp vox':>8s} {'ratio':>6s} {'AP ratio':>8s}")
    for name in pl.VERT_ORDER:
        a, b = st_total.get(name), st_pp.get(name)
        if a is None and b is None:
            continue
        r = (b["n"] / a["n"]) if (a and b and a["n"]) else float("nan")
        ap = ((b["y"][1] - b["y"][0] + 1) / (a["y"][1] - a["y"][0] + 1)) if (a and b) else float("nan")
        if a and b:
            ratios.append(r)
            ap_ratios.append(ap)
        say(f"  {pl.short_name(name):5s} {str(a['z']) if a else '-':>11s} {str(b['z']) if b else '-':>11s} "
            f"{a['n'] if a else '-':>10} {b['n'] if b else '-':>8} {fmt(r):>6s} {fmt(ap):>8s}")
    if ratios:
        med_r, med_ap = float(np.median(ratios)), float(np.median(ap_ratios))
        say(f"  median voxel ratio pp/total = {med_r:.2f}, median AP-extent ratio = {med_ap:.2f}"
            f" -> {'bodies only (as documented in the TS source)' if med_r < 0.75 and med_ap < 0.85 else 'NOT clearly bodies-only, look at the overlay'}")
        summary.update(pp_total_voxel_ratio=round(med_r, 3), pp_total_ap_ratio=round(med_ap, 3))

    # --- levels from both sources ---
    section("VERTEBRAL LEVELS: total vs vertebrae_pp_refined")
    lv_total = levels_from(per_slice_total, extremes)
    lv_pp = levels_from(per_slice_pp, extremes) if per_slice_pp is not None else {}
    mismatches = []
    for k in lv_total:
        a, b = lv_total[k], lv_pp.get(k, "?")
        flag = "" if a == b else "  <-- MISMATCH"
        if a != b and lv_pp:
            mismatches.append(f"{k} {b}/{a}")
        say(f"  {k:10s} total={a or '-':4s} pp={b or '-':4s}{flag}")
    summary["level_mismatches"] = mismatches

    # --- airways / carina ---
    section("AIRWAYS AND CARINA (lung_vessels)")
    carina_z = None
    lv = canon("lung_vessels") if "lung_vessels" in loaded else None
    if lv is not None:
        lcounts = np.bincount(lv.ravel(), minlength=5)
        say(f"  voxels: lumen={lcounts[1]} wall={lcounts[2]} arteries={lcounts[3]} veins={lcounts[4]}")
        lumen = lv == 1
        wall = lv == 2
        tree = tm.largest_component(lumen, 26)
        zs = np.flatnonzero(tree.any(axis=(0, 1)))
        if zs.size and extremes["both"]:
            apex = extremes["both"][1]
            say(f"  lumen tree: z {zs[0]}..{zs[-1]}; top is {(zs[-1] - apex) * spacing[2]:+.1f} mm relative to the lung apex"
                f" ({'reaches above the apex' if zs[-1] > apex else 'does NOT reach above the apex'})")
            summary["lumen_top_minus_apex_mm"] = round(float((zs[-1] - apex) * spacing[2]), 1)
        t0 = time.monotonic()
        res = tm.find_carina(lumen, spacing)
        used = "lumen"
        if not res.found and trachea_id is not None and (total == trachea_id).any():
            res2 = tm.find_carina(lumen | (total == trachea_id), spacing)
            if res2.found:
                res, used = res2, "lumen+trachea"
        say(f"  carina (slice rule, {used}): {'z=' + str(res.z) if res.found else 'NOT FOUND (' + res.reason + ')'}"
            f" [{time.monotonic() - t0:.1f} s]")
        summary.update(carina_found=res.found, carina_reason=res.reason, carina_source=used)
        centre = res.z if res.found else res.z_top
        if centre >= 0:
            lo, hi = max(0, centre - 10), min(nz - 1, centre + 10)
            say("  2-D lumen components per slice (z: n) around it: "
                + " ".join(f"{z}:{res.n2d[z]}" for z in range(hi, lo - 1, -1)))
        if res.found:
            carina_z = res.z
            lvl_pp = pl.short_name(pl._vert_level(per_slice_pp, carina_z, step=-1, pick_cranial=True)) if per_slice_pp else "-"
            lvl_tot = pl.short_name(pl._vert_level(per_slice_total, carina_z, step=-1, pick_cranial=True))
            say(f"  carina level: pp={lvl_pp} total={lvl_tot}")
            if extremes["both"]:
                say(f"  carina to apex: {(extremes['both'][1] - carina_z) * spacing[2]:.1f} mm | to base: {(carina_z - extremes['both'][0]) * spacing[2]:.1f} mm")
            summary.update(carina_z=int(carina_z), carina_level_pp=lvl_pp, carina_level_total=lvl_tot)
            t0 = time.monotonic()
            zsk = safe(tm.carina_from_skeleton, "skeleton carina", lumen, spacing)
            say(f"  carina (skeleton cross-check): {'z=' + str(zsk) if zsk is not None else 'not found'}"
                + (f" (dz = {(zsk - carina_z) * spacing[2]:+.1f} mm)" if zsk is not None else "") + f" [{time.monotonic() - t0:.1f} s]")
            summary["carina_skeleton_z"] = zsk
            t0 = time.monotonic()
            air = safe(tm.airway_metrics, "airway metrics", lumen, wall, carina_z, spacing)
            if air:
                lumen_ml, wall_ml = air.lumen_vox * vox_ml, air.wall_vox * vox_ml
                say(f"  below carina: lumen {lumen_ml:.2f} ml | wall {wall_ml:.2f} ml | wall% {100 * wall_ml / (wall_ml + lumen_ml) if lumen_ml + wall_ml else float('nan'):.1f}"
                    f" | lumen/lung {lumen_ml / vol_both_ml if vol_both_ml else float('nan'):.4f}")
                say(f"  skeleton: branches {air.branch_count} | terminals {air.terminal_count} | junctions {air.junction_count}"
                    f" | loops {air.cycle_rank} | prune rounds {air.prune_rounds} [{time.monotonic() - t0:.1f} s]")
                summary.update(airway_lumen_ml=round(lumen_ml, 2), airway_wall_ml=round(wall_ml, 2),
                               airway_branches=air.branch_count, airway_terminals=air.terminal_count, airway_loops=air.cycle_rank)
    else:
        say("  lung_vessels mask missing")

    # --- vessels ---
    section("VESSELS (lung_vessels)")
    if lv is not None:
        t0 = time.monotonic()
        vr = safe(tm.vessel_metrics, "vessel metrics", lv, lobe_mask, spacing)
        if vr:
            a_ml, v_ml = vr.artery_vox * vox_ml, vr.vein_vox * vox_ml
            say(f"  arteries {a_ml:.1f} ml | veins {v_ml:.1f} ml | ratio {a_ml / v_ml if v_ml else float('nan'):.3f}")
            say(f"  BV5 small-vessel volume: {vr.small_vox * vox_ml:.2f} ml (corrected radius) | {vr.small_vox_uncorrected * vox_ml:.2f} ml (uncorrected)"
                f" | radius p50 {vr.radius_p50_mm:.2f} mm p90 {vr.radius_p90_mm:.2f} mm")
            say(f"  skeleton fragments: {vr.skeleton_fragments} ({vr.skeleton_fragments / (a_ml + v_ml) if a_ml + v_ml else float('nan'):.1f} per ml of vessel;"
                f" high values = broken thin vessels, expected on thick slices) [{time.monotonic() - t0:.1f} s]")
            summary.update(artery_ml=round(a_ml, 2), vein_ml=round(v_ml, 2), bv5_ml=round(vr.small_vox * vox_ml, 2),
                           bv5_uncorrected_ml=round(vr.small_vox_uncorrected * vox_ml, 2), vessel_fragments=vr.skeleton_fragments)
    else:
        say("  lung_vessels mask missing")

    # --- density ---
    section("PARENCHYMAL DENSITY (CT + total + lung_vessels)")
    if lv is not None:
        t0 = time.monotonic()
        dr = safe(tm.parenchyma_density, "density", ct, lung_both, lv > 0, spacing)
        if dr:
            say(f"  parenchyma voxels after 2 mm erosion: {dr.n_vox} ({dr.n_vox * vox_ml:.0f} ml of {vol_both_ml:.0f} ml)"
                f" | mean HU {dr.mean_hu:.1f} | LAA-950 {dr.laa_pct:.2f} % | p15 {dr.p15_hu:.0f} HU [{time.monotonic() - t0:.1f} s]")
            summary.update(lung_mean_hu=round(dr.mean_hu, 1), laa950_pct=round(dr.laa_pct, 2))
    else:
        say("  needs lung_vessels")

    # --- effusion ---
    section("EFFUSION (pleural_pericard_effusion)")
    if "pleural_pericard_effusion" in loaded:
        eff = canon("pleural_pericard_effusion")
        ecounts = np.bincount(eff.ravel(), minlength=4)
        say(f"  voxels: lung_pleura={ecounts[1]} pleural_effusion={ecounts[2]} pericardial_effusion={ecounts[3]}")
        n_r, n_l, note = tm.split_effusion_lr(eff == 2, pp > 0 if pp is not None else np.zeros_like(eff, dtype=bool),
                                              lungs_R=lung_R, lungs_L=lung_L)
        say(f"  pleural R {n_r * vox_ml:.1f} ml | L {n_l * vox_ml:.1f} ml | pericardial {ecounts[3] * vox_ml:.1f} ml {('(' + note + ')') if note else ''}")
        summary.update(pleural_R_ml=round(n_r * vox_ml, 2), pleural_L_ml=round(n_l * vox_ml, 2),
                       pericardial_ml=round(ecounts[3] * vox_ml, 2))
    else:
        say("  mask missing")

    # --- thoracic cavity ---
    section("THORACIC CAVITY (trunk_cavities)")
    if "trunk_cavities" in loaded:
        tc = canon("trunk_cavities")
        tcounts = np.bincount(tc.ravel(), minlength=5)
        say(f"  voxels: abdominal_cavity={tcounts[1]} thoracic_cavity={tcounts[2]} pericardium={tcounts[3]} mediastinum={tcounts[4]}")
        h = tm.cavity_height(tc == 2, spacing)
        lung_h = (extremes["both"][1] - extremes["both"][0]) * spacing[2] if extremes["both"] else float("nan")
        say(f"  thoracic cavity height {fmt(h, 1)} mm vs lung height {lung_h:.1f} mm")
        summary["thoracic_cavity_height_mm"] = h
    else:
        say("  mask missing")
    return summary


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--dicom", type=Path, help="folder containing this case's DICOMs (an export folder is fine)")
    src.add_argument("--ct", type=Path, help="an already converted CT NIfTI")
    ap.add_argument("--patient-id", default=None, help="restrict to this PatientID (flat exports with several patients)")
    ap.add_argument("--min-slices", type=int, default=20)
    ap.add_argument("--out", type=Path, required=True, help="scratch folder for case.nii.gz and the masks (patient data!)")
    ap.add_argument("--device", default="auto", help="auto|cpu|mps|gpu|gpu:X (default auto = CUDA, then MPS, then CPU)")
    ap.add_argument("--tasks", default=",".join(ALL_TASKS), help="comma list of tasks to run")
    ap.add_argument("--fast-too", action="store_true", help="also time the 3 mm fast total model")
    ap.add_argument("--variants", action="store_true", help="also time resampling variants")
    ap.add_argument("--skip-inference", action="store_true", help="analyse existing masks in --out only")
    ap.add_argument("--no-download", action="store_true", help="skip the untimed weight pre-download")
    ap.add_argument("--nr-thr-saving", type=int, default=6,
                    help="TotalSegmentator nr_thr_saving = nnU-Net export worker processes per model call "
                         "(TotalSegmentator default 6; 1 avoids starting 6 python processes per model call)")
    ap.add_argument("--nr-thr-resamp", type=int, default=1, help="TotalSegmentator nr_thr_resamp (default 1)")
    ap.add_argument("--json", type=Path, default=None, help="write the identifier-free summary here")
    args = ap.parse_args()
    args.tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    bad = [t for t in args.tasks if t not in ALL_TASKS]
    if bad:
        sys.exit(f"error: unknown task(s) {bad}; choose from {ALL_TASKS}")

    logging.getLogger("pipeline").addHandler(logging.NullHandler())
    args.out.mkdir(parents=True, exist_ok=True)
    RED.add_path(args.out)
    t_all = time.monotonic()
    say("diagnose_totalseg.py | identifier-free output: paste it back whole")

    device = resolve_device(args.device)
    env = section_environment(args.device, device)
    reg = safe(section_registry, "registry", args.tasks) or {}

    ct_path, tags = prepare_ct(args, args.out)

    lobes, verts = pl.get_label_maps()
    roi = list(lobes) + list(verts) + ["trachea"]
    roi_no_trachea = list(lobes) + list(verts)
    runs = build_runs(args, roi, roi_no_trachea, cuda=bool(env.get("cuda")))
    if not args.skip_inference and not args.no_download:
        download_weights(runs)
    rows = run_tasks(runs, ct_path, args.out, device, args.skip_inference,
                     nr_thr_saving=args.nr_thr_saving, nr_thr_resamp=args.nr_thr_resamp)

    mask_paths = {name: args.out / f"case_{name}.nii.gz" for name, _ in runs if name in ALL_TASKS}
    summary = safe(analyze, "analysis", ct_path, mask_paths) or {}
    print_runtime_table(rows)

    say(f"\ntotal wall time: {(time.monotonic() - t_all) / 60:.1f} min")
    if args.json:
        payload = {"environment": env, "registry": reg, "ct": tags, "runs": rows, "analysis": summary}
        args.json.write_text(RED(json.dumps(payload, indent=2, default=str)))
        say(f"summary written to {args.json.name}")
    say("done. Paste this whole output back.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
