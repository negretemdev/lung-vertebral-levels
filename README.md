# lung-vertebral-levels

Batch pipeline for chest CTs. For each patient it runs a set of
TotalSegmentator tasks once, saves every mask, and measures craniocaudal lung
length and the vertebral levels at the lung apex and base — separately for the
right lung, left lung, and both combined — plus parenchymal density, vessel and
airway metrics, the carina, effusion volumes and the thoracic cavity.

Masks are never recomputed: each one is stored with the run report that proves
what produced it, so a cohort can be built in passes and a second run over the
same folder only fills in what is missing. Runs on Apple Silicon (Metal) and on
NVIDIA GPUs; `--device auto` picks whichever is there.

## Setup

Requires [uv](https://docs.astral.sh/uv/). From the repo root:

```bash
uv sync                 # pipeline only
uv sync --extra viz     # + napari for the QC viewer (test.py); not needed on the batch machine
```

### Windows / NVIDIA machine

- `uv sync` on Windows installs the CUDA 13.0 build of torch from the PyTorch
  index (declared in `pyproject.toml`; PyPI's Windows torch is CPU-only). It
  needs an NVIDIA driver 580 or newer: run `nvidia-smi` and check that the
  "CUDA Version" in its header is 13.0 or higher. Older driver: update it.
- Model weights: TotalSegmentator downloads them on first use into
  `%USERPROFILE%\.totalsegmentator` (about 3 GB for all tasks used here). For
  an offline machine copy the whole `~/.totalsegmentator` folder from the Mac
  after running the diagnostic there, which downloads every needed model.
- `--device auto` picks CUDA when available, then Apple MPS, then CPU.

## Phase 1: diagnostic on one case (before any batch)

`diagnose_totalseg.py` runs every TotalSegmentator task of the extended
pipeline once on ONE case, times each task, and checks the masks (does the
airway mask reach the carina, are the `vertebrae_pp` labels bodies only, is
the CT stored in HU, how long does each model take on this machine, ...).
Its printed output contains no identifiers and can be pasted back whole. The
`--out` folder receives the case's CT and masks: that is patient data, keep it
with the data and never inside the repo.

```bash
# MacBook (Apple GPU): one patient folder, or a flat export + PatientID
uv run python diagnose_totalseg.py --dicom "/Volumes/DRIVE/cohort/PATIENT_FOLDER" --out "/Volumes/DRIVE/diag_case1"
uv run python diagnose_totalseg.py --dicom "/Volumes/DRIVE/export" --patient-id "ID" --out "/Volumes/DRIVE/diag_case1"

# Windows / 4090: same command with --device gpu
uv run python diagnose_totalseg.py --dicom "D:\cohort\PATIENT_FOLDER" --out "D:\diag_case1" --device gpu
```

Options: `--fast-too` (also time the 3 mm model), `--variants` (resampling
variants; adds several full model runs), `--tasks total,lung_vessels` (subset),
`--skip-inference` (re-analyse masks already in `--out`), `--json summary.json`.
The first run also downloads the model weights (untimed).

## Choosing the input and output folders

Both folders are given as explicit named flags, so the order never matters:

```bash
uv run python pipeline.py --input <patients folder> --output <results folder>
```

Type the paths, tab-complete them, or **drag the folder from Finder onto the
Terminal window** and macOS pastes its full path for you.

External drives on macOS live under `/Volumes/<drive name>`. Check what's
mounted with `ls /Volumes`. If a path contains spaces, wrap it in quotes:

```bash
uv run python pipeline.py --input "/Volumes/My Passport/chest_cts" \
                          --output "/Volumes/My Passport/lung_output"
```

### Two supported input layouts

**Layout 1 — one folder per patient (default).** `--input` is the root whose
immediate subfolders are each one patient:

```
cohort/                  <- --input
├── PAT001/  ... DICOM slices anywhere inside (nesting is fine)
├── PAT002/
└── ...
```

**Layout 2 — a single DICOM export (`--flat`).** A PACS/CD-style export with a
`DICOMDIR` index and one big pile of image files, no per-patient folders,
often with extensionless names (macOS shows them as "Unix Executable File" —
they are normal DICOM files):

```
exports/                 <- --input, with --flat
├── DICOMDIR
└── IMAGES/
    ├── IM000001
    ├── IM000002
    └── ...
```

Nesting does not matter. Several exports side by side, one per download date,
each holding dozens of patients, is the same layout and the same flag:

```
cohort/                  <- --input, with --flat
├── 2026-09-16-001/      (one export, ~30 patients)
│   ├── DICOMDIR
│   └── IMAGES/
├── 2026-09-16-002/
│   ├── DICOMDIR
│   └── IMAGES/
└── ...
```

With `--flat`, folder structure is ignored: every file's header is read and
cases are grouped by the **PatientID and study stored inside the DICOMs** —
this works whether the export holds one patient or two hundred. The case id
(`folder_id` column, and the folder name under `masks/`) is the PatientID; a
second study of the same patient becomes `<PatientID>_study2`; `id_mismatch` is
only True if a file has no PatientID at all.

**If you forget `--flat`**, the pipeline looks for a DICOMDIR at the root and a
few levels below it, and refuses to start when it finds one. Without that check
each export folder would be taken for a single patient: one patient would be
segmented and the other thirty silently skipped. As a second line of defence,
any folder that turns out to hold more than one PatientID produces an `error:`
row naming the count, rather than a quietly wrong result.

Run `--flat --dry-run` first to see the detected patient list before segmenting
anything.

- **Input root** (layout 1) = the folder that directly contains one subfolder
  per patient (not a single patient's folder). Every immediate subfolder is
  treated as a patient.
- **Output folder** can be anywhere you can write — on the same external drive
  is fine. It is created automatically if it doesn't exist.
- The input is only ever read. If the drive is formatted NTFS, macOS mounts it
  read-only — fine for the input, but the output must then go elsewhere (e.g.
  `~/Documents/lung_output`).

## Usage

```bash
# optional first check: also opens every patient's DICOMs to verify series
# selection and PatientID. Writes nothing, segments nothing.
uv run python pipeline.py --input /Volumes/MyDrive/chest_cts --output ./out --dry-run

# smoke test: fully process only the first 2 patients
uv run python pipeline.py --input /Volumes/MyDrive/chest_cts --output ./out --test

# the normal run: all five tasks at full resolution
uv run python pipeline.py --input /Volumes/MyDrive/chest_cts --output ./out

# build the cohort in passes: a later run only adds what is missing
uv run python pipeline.py --input /Volumes/MyDrive/chest_cts --output ./out --tasks total,vertebrae_pp_refined
uv run python pipeline.py --input /Volumes/MyDrive/chest_cts --output ./out

# single DICOM export (DICOMDIR + IMAGES, no per-patient folders): add --flat
uv run python pipeline.py --input /Volumes/MyDrive/exports --output ./out --flat --dry-run
uv run python pipeline.py --input /Volumes/MyDrive/exports --output ./out --flat
```

**Every run** (with or without `--dry-run`) starts by instantly printing the
full patient roster — every folder found, numbered, with whether it will be
segmented or only re-measured (resume) — so you can confirm the count before
any heavy work starts:

```
INPUT  (read-only): /Volumes/MyDrive/chest_cts
OUTPUT            : /Volumes/MyDrive/lung_output
resolution        : full (1.5 mm) -> /Volumes/MyDrive/lung_output/full
device: auto -> gpu (CUDA True, MPS False)
tasks             : total, vertebrae_pp_refined, lung_vessels, pleural_pericard_effusion, trunk_cavities
vertebrae from    : vertebrae_pp_refined
=== 143 patient folders found ===
    1/143  PAT0001                     0/5 masks present
    2/143  PAT0002                     5/5 masks present
    ...
=== 1 case(s) complete, 142 to segment | device=gpu | 5 task(s) ===
```

Patient discovery and series selection are independent: series selection can
never cause a patient to be skipped. If a folder has no usable series, that
patient gets an `error: ...` row in `results.csv` — it is never silently
dropped, and the roster count always equals the number of CSV rows written.

`--dry-run` additionally opens every patient's DICOMs, reports the series that
would be used and the PatientID, flags problems, and exits without segmenting
or writing anything.

- **Input** (read-only, never modified): one subfolder per patient containing
  that patient's DICOM slices. Multiple series per folder are fine — the axial
  CT series with the most slices is used (series with < 20 slices are ignored;
  the chosen series is logged).
- **Output**:

  ```
  output/
  ├── ct/<id>.nii.gz                   the CT that was segmented
  ├── masks/<id>/<task>.nii.gz         one mask per task
  │                <task>.report.json  what produced it
  ├── full/results.csv, pipeline.log   normal runs
  └── fast/results.csv, pipeline.log   --fast runs
  ```

  Budget roughly 150–250 MB per case for the CT and its masks.
- **QC in ITK-SNAP**: open `ct/<id>.nii.gz` as the main image, then any
  `masks/<id>/<task>.nii.gz` via *Segmentation → Open Segmentation* — they
  overlay voxel-for-voxel. `uv run python test.py --ct ... --seg ...` does the
  same in napari with class names on hover (needs `uv sync --extra viz`).
- **Resume**: a mask is reused when its run report shows it was made for the
  same task, at the same resolution, with the same class list, and it sits on
  the saved CT's voxel grid. Anything else is redone. A mask without a report is
  an interrupted run and is always redone. Re-runs append to `results.csv`;
  delete or rename it for a clean file.
- **Columns added later**: if `results.csv` was written by an older version its
  header no longer matches, and the pipeline stops rather than write rows that
  would not line up. Rename that file and run again — the masks are kept, so
  every row is rebuilt in minutes without re-segmenting anything.
- One failing case never stops the batch: its row gets an error message in the
  `status` column and processing continues.
- **Geometry safety**: converted volumes never trust vendor metadata for the
  slice direction (GE writes a negative SpacingBetweenSlices that flips naive
  converters); geometry is recomputed from the slice positions, and every case
  — resumed ones included — is validated against the DICOM positions before
  segmentation/measurement. A mismatch becomes an `error:` row; a silently
  flipped volume cannot pass through.
- **`--recheck-geometry`** (recovery pass): re-converts every saved CT with the
  fixed converter and re-segments ONLY the cases whose geometry actually
  changed (unchanged cases keep their segmentation and are just re-measured).
  Use once after upgrading past the flip bug, then never needed again.

### What runs, and what it costs

Measured on one 470-slice chest CT (0.9 × 0.9 × 0.8 mm), per case:

| task | what it segments | columns it fills | RTX 4090 | M3 Pro |
|---|---|---|---|---|
| `total` | 5 lung lobes, trachea | every length, width, depth and volume column | 67 s | 85 s |
| `vertebrae_pp_refined` | vertebral bodies C1–L5 | levels, spine spans, Cobb angle | 44 s | 155 s |
| `lung_vessels` | arteries, veins, airway lumen and wall | 15 columns: vessels, airways, carina, density | 55 s | 238 s |
| `pleural_pericard_effusion` | pleural and pericardial effusion | the 3 effusion columns | 35 s | 58 s |
| `trunk_cavities` | thoracic cavity | thoracic cavity height | 19 s | 69 s |

Roughly 4 minutes per case on the 4090 and 11 on the MacBook, so the laptop is
for tests and the GPU machine for the batch.

`--fast` runs `total` with the 3 mm model instead of 1.5 mm and writes its rows
to `<output>/fast/`. It is for checking that the plumbing works, not for
results: on the test case it shifted the vertebral level at the lung base by a
whole level while lung volumes moved under 1 %.

Other flags:

| flag | meaning |
|---|---|
| `--tasks a,b,c` | which tasks to create when missing (default: all five). `total` is always included |
| `--device` | `auto` (default), `cpu`, `mps`, `gpu`, `gpu:N`. An unavailable choice is an error, never a silent fall back to the CPU |
| `--fast` | 3 mm `total`; rows go to `fast/` |
| `--total-vertebrae` | also ask `total` for the vertebrae. Off by default: the levels come from the vertebral bodies, and asking `total` for vertebrae runs a second model for labels nothing reads. Turning it on restores the independent cross-check of the levels |
| `--force-split` | process `total` in 3 chunks to use less memory |
| `--nr-thr-saving N` | nnU-Net export worker processes per model call (default 1) |
| `--min-slices N` | ignore series with fewer slices (default 20) |

## Measurement guide — every column of results.csv

### Before anything is measured: mask cleaning

All lung measurements share the same preparation. The saved segmentation is
reoriented to canonical patient axes (RAS — so "z" is always craniocaudal
regardless of how the scan was acquired) and voxel sizes are read from the
file header, never assumed. Then, per side: the lobes are merged into one
lung, only the largest 3D connected component is kept (removes mislabeled
islands), and when locating the apex/base, axial slices with fewer than 50
lung voxels are ignored (a handful of stray voxels cannot fake an apex).
"R"/"L" come from the *labels*, not from position in the image.

### The three (or four) variants of each measurement

| suffix | meaning |
|---|---|
| `_R`, `_L` | right lung alone, left lung alone (after cleaning) |
| `_both` | the two lungs **merged into one object** and measured as such — an *envelope*. E.g. `width_both_mm` runs from the outer edge of one lung to the outer edge of the other, mediastinal gap included; `height_both_mm` runs from the higher apex to the lower base (so it is ≥ each single-lung value, but never their sum) |
| `_sum` | simply **R + L**, added before rounding — "total lung tissue", gaps excluded |

### Identity (from the DICOM header)

| column | meaning |
|---|---|
| `folder_id` | the patient's folder name; also names the segmentation file in `segs/` |
| `dicom_patient_id` | DICOM tag (0010,0020) Patient ID — normally the MRN |
| `accession_number` | (0008,0050) — the study's RIS accession |
| `study_date` | (0008,0020), as `YYYYMMDD` |
| `id_mismatch` | `True` when `folder_id` ≠ PatientID (or PatientID is missing) — a data-organization alarm, check those folders |

### Vertebral levels (labels, not distances)

| column | meaning |
|---|---|
| `start_R`, `start_L`, `start_both` | the vertebra (e.g. `T1`) overlapping the axial slice of that lung's **apex**. If that slice has no vertebra (intervertebral gap), the search walks toward the feet until one is found; two on one slice → the more cranial |
| `end_R`, `end_L`, `end_both` | same for the **base** slice, walking toward the head if needed; ties → the more caudal |

`_both` uses the combined mask's extremes: the higher apex of either lung, the
lower base of either lung.

**These come from the vertebral BODY**, segmented by `vertebrae_pp_refined`,
because that is how a level is read clinically: the body, not the arch. It
matters. A whole vertebra's posterior arch reaches higher than its body, so
reading the level off whole vertebrae can name the vertebra below. On the test
case the lung base read T12 from whole vertebrae and T11 from the bodies.
Anything measured with the single-task version of this pipeline used whole
vertebrae and may sit one level lower.

With `--total-vertebrae` the same levels are also read off the whole vertebrae
from `total` and any disagreement is reported in `status` as
`level mismatch total vs vertebrae_pp: ...`, with the body-based value first.

### Craniocaudal lengths (mm)

| column | meaning |
|---|---|
| `height_*_mm` | **perpendicular** apex-to-base length: distance between the apex and base *axial planes*. The standard "craniocaudal lung length". Robust to patient tilt |
| `diag_*_mm` | **straight 3D line** between the lung centroids on the apex and base slices. Always ≥ height; the height-vs-diag gap grows with obliquity, so the pair is itself a tilt indicator |

### The other two axes (mm)

| column | meaning |
|---|---|
| `width_*_mm` | left–right (transverse) extent of the cleaned mask (bounding box) |
| `depth_*_mm` | anteroposterior extent (bounding box) |

### Volumes (mL = cm³)

| column | meaning |
|---|---|
| `vol_R_ml`, `vol_L_ml`, `vol_both_ml` | voxel count × voxel volume; `vol_both_ml` is the exact R + L sum (computed before rounding, so it may differ from adding the two rounded columns by 0.01) |

Volumes are at the **scanned respiratory phase**: inspiration vs expiration
scans are not comparable.

### Spine (mm)

| column | meaning |
|---|---|
| `spine_height_mm` | craniocaudal span of the whole segmented vertebral column visible in the scan (field-of-view dependent!) |
| `spine_lung_span_mm` | span of only the vertebral levels from `start_both` to `end_both` — the stretch of spine the lungs sit against |

### Acquisition (from the DICOM header, no model involved)

| column | meaning |
|---|---|
| `slice_thickness_mm` | (0018,0050). Thick slices make the vessel columns unreliable; the pipeline says so in `status` above 1.5 mm |
| `convolution_kernel` | (0018,1210), multiple values joined with `/`. Sharp and soft kernels shift density and vessel numbers, so cases should be grouped by kernel before comparing |

### Parenchymal density (needs `lung_vessels`)

Measured inside the lungs after removing every vessel and airway voxel and
eroding 2 mm, so neither the vessels nor the pleural surface contribute.

| column | meaning |
|---|---|
| `lung_mean_hu` | mean attenuation of the remaining parenchyma |
| `laa950_pct` | percent of it below −950 HU, the usual emphysema index |

### Vessels (needs `lung_vessels`)

Arteries and veins are counted only inside the lung lobes, so the hilar and
mediastinal parts are excluded.

| column | meaning |
|---|---|
| `artery_vol_ml`, `vein_vol_ml` | intraparenchymal arterial and venous volume |
| `artery_vein_ratio` | artery ÷ vein |
| `small_vessel_vol_ml` | volume in vessels whose local cross-section is under 5 mm², the usual BV5. The local radius comes from the distance transform at the nearest skeleton voxel. Not comparable with published BV5 above about 1.5 mm slices, and `status` says so |

### Airways (needs `lung_vessels`)

Everything here is counted strictly **below the carina**, so the trachea never
contributes and cases with different scan ranges stay comparable.

| column | meaning |
|---|---|
| `airway_lumen_vol_ml` | air inside the bronchi of the largest connected tree |
| `airway_wall_vol_ml` | the wall around that tree, limited to 3 mm from the lumen so a leaking label cannot inflate it |
| `airway_wall_pct` | wall ÷ (wall + lumen), a marker of airway remodelling |
| `airway_lumen_lung_ratio` | lumen ÷ `vol_both_ml` |
| `airway_branch_count` | branches of the skeletonised tree below the carina, after pruning spurs under 3 mm. It depends on how much tree the model resolved, so compare it only within one scan protocol |

### Carina (needs `lung_vessels`)

The carina is found from the airway mask alone: the most cranial slice where the
trachea splits into two branches that stay separate for at least 8 mm. Nothing
is assumed about where it ought to be. If the mask never splits, the columns
stay empty and `status` says `carina not found` with the reason.

| column | meaning |
|---|---|
| `carina_level` | the vertebral body at the carina — a fixed landmark, independent of the scan range |
| `carina_to_apex_mm`, `carina_to_base_mm` | its distance to the lung apex and base |

### Effusion (needs `pleural_pericard_effusion`)

| column | meaning |
|---|---|
| `pleural_eff_R_ml`, `pleural_eff_L_ml` | pleural fluid per side, split at the spine midline taken slice by slice from the vertebral bodies and interpolated across the disc gaps |
| `pericardial_eff_ml` | pericardial fluid |

### Thoracic cavity (needs `trunk_cavities`)

| column | meaning |
|---|---|
| `thoracic_cavity_height_mm` | craniocaudal height of the thoracic cavity. Compared with `height_both_mm` it says how much of the cavity the lungs fill |

### Quality / confounder columns

| column | meaning |
|---|---|
| `cobb_angle_deg` | **Ferguson-style coronal curvature angle** from vertebral centroids: centroids are projected onto the coronal plane (AP dropped — kyphosis/lordosis reads 0°), vertebrae cut by the field of view or abnormally small are excluded, and the angle is measured between the lines top→apex and apex→bottom (apex = most laterally deviated centroid). 0° = straight spine. A scoliosis *flag*, NOT a clinical endplate Cobb angle (centroid methods underestimate it, and only in-FOV vertebrae contribute). Empty if < 5 usable vertebrae |
| `status` | `ok`, or `ok; <warnings>`, or `error: <message>` (case failed but the batch continued). Warnings include the **vertebra sanity flags** (`vertebra check: ...`): missing level mid-sequence, spatially inverted labels, one label split across two bones, or a ~2× centroid gap (a physical vertebra the model skipped). Any flagged case deserves visual review. The checks are *internal* consistency only — a labeling uniformly shifted one level (transitional anatomy, T13/L6) passes them and is only caught by overlay review |
| `runtime_s` | wall-clock seconds for the case. Segmentation dominates; a case whose masks are all reused still costs 10–30 s for the distance transforms and skeletons |

Besides the vertebra checks, `status` can carry:

| warning | what it means |
|---|---|
| `lung touches scan edge (S)` | the lung reaches the top or bottom of the scanned volume, so its height is only a lower bound. Other letters are the other faces |
| `carina not found (<reason>)` | the airway mask never splits, so the carina and airway columns stay empty. The reason says whether the trunk was lost, never split, or the scan starts below it |
| `carina from lumen+trachea` | the airway mask alone did not reach the carina, so the trachea from `total` was added to find it |
| `level mismatch total vs vertebrae_pp: ...` | the body-based and whole-vertebra levels disagree (body value first). Only with `--total-vertebrae` |
| `lung HU implausible (...)` | the mean parenchymal attenuation is outside −1000 to −500 HU, so the scan may not be in Hounsfield units |
| `small-vessel volume unreliable at X mm slices` | slices thicker than 1.5 mm; the BV5 column is not comparable with published values |
| `no CT saved` | the DICOMs could not be converted, so TotalSegmentator was fed them directly and the density columns stay empty |
| `<task> failed: ...` | an optional task failed; its columns are empty and the next run retries it. A failure of `total` or the vertebra task makes the whole case an `error:` row |
| `<task> ran on cpu (mps failed ...)` | the Apple GPU could not run that task, so it ran on the CPU |
| `<task> mask from TotalSegmentator <v>` | the mask was reused but made by a different version |

All numeric values have 2 decimals; with ~2 mm slices the real uncertainty is
about one slice thickness, so treat sub-millimeter digits as noise.

## Testing without real data

```bash
uv run python make_fake_case.py
uv run python pipeline.py --input fake_case/input --output fake_case/output --test
```

This creates a synthetic patient (a 60-slice axial DICOM series plus a scout
series that must be ignored) and a full set of fake masks with known ground
truth, including traps for every cleaning rule: detached blobs, a connected
spike, a gap between vertebrae, a non-canonical orientation, a bowed spine, a
leaking airway wall, and an effusion blob straddling the midline. Every mask
carries a run report, so the pipeline reuses them all and runs no inference; it
must reproduce the printed values exactly.

`uv run pytest` checks the same thing automatically, along with the metric
functions on synthetic arrays, the reuse rule and the CSV guard.

Delete one mask, say `fake_case/output/masks/fake_patient_01/trunk_cavities.nii.gz`
together with its `.report.json`, to watch only that task get recomputed. Real
inference on the synthetic images finds no anatomy, which is the point: it
proves the DICOM → segmentation → measurement path runs end to end.
