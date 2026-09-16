# lung-vertebral-levels

Batch pipeline for chest CTs: runs TotalSegmentator (lung lobes + all vertebrae)
on each patient's DICOM folder, then measures craniocaudal lung length and the
vertebral levels at the lung apex and base — separately for the right lung, left
lung, and both combined. Built for Apple Silicon (`device="mps"`).

## Setup

Requires [uv](https://docs.astral.sh/uv/). From the repo root:

```bash
uv sync
```

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

With `--flat`, folder structure is ignored: every file's header is read and
cases are grouped by the **PatientID and study stored inside the DICOMs** —
this works whether the export holds one patient or two hundred. The case id
(`folder_id` column, segmentation filename) is the PatientID; a second study
of the same patient becomes `<PatientID>_study2`; `id_mismatch` is only True
if a file has no PatientID at all. If you forget `--flat` on such a folder,
the pipeline detects the DICOMDIR and refuses with a hint instead of
mis-treating `IMAGES` as one patient. Run `--flat --dry-run` first to see the
detected patient list before segmenting anything.

- **Input root** (layout 1) = the folder that directly contains one subfolder
  per patient (not a single patient's folder). Every immediate subfolder is
  treated as a patient.
- **Output folder** can be anywhere you can write — on the same external drive
  is fine. It is created automatically (including `segs/`) if it doesn't exist.
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

# fast (3 mm) model — recommended first pass
uv run python pipeline.py --input /Volumes/MyDrive/chest_cts --output ./out

# full-resolution model
uv run python pipeline.py --input /Volumes/MyDrive/chest_cts --output ./out --full

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
=== 143 patient folders found ===
    1/143  PAT0001    needs segmentation
    2/143  PAT0002    seg exists -> measurements only
    ...
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
- **Output**: `output/segs/<folder_id>.nii.gz` (multilabel segmentation),
  `output/ct/<folder_id>.nii.gz` (the CT volume that was actually segmented,
  on the **same voxel grid** as the mask), `output/results.csv` (one row per
  patient, appended as each case finishes), `output/pipeline.log`.
- **QC in ITK-SNAP**: open `ct/<id>.nii.gz` as the main image, then
  `segs/<id>.nii.gz` via *Segmentation → Open Segmentation* — they overlay
  voxel-for-voxel. (Budget disk space: each CT is roughly 30–100 MB.)
- **Resume**: if `segs/<folder_id>.nii.gz` already exists, segmentation is
  skipped and only the (instant) measurements are recomputed. Note that re-runs
  append to `results.csv`; delete or rename it if you want a clean file.
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

Other flags: `--device {mps,cpu,gpu}` (default `mps`), `--min-slices N`.

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

### Quality / confounder columns

| column | meaning |
|---|---|
| `cobb_angle_deg` | **Ferguson-style coronal curvature angle** from vertebral centroids: centroids are projected onto the coronal plane (AP dropped — kyphosis/lordosis reads 0°), vertebrae cut by the field of view or abnormally small are excluded, and the angle is measured between the lines top→apex and apex→bottom (apex = most laterally deviated centroid). 0° = straight spine. A scoliosis *flag*, NOT a clinical endplate Cobb angle (centroid methods underestimate it, and only in-FOV vertebrae contribute). Empty if < 5 usable vertebrae |
| `status` | `ok`, or `ok; <warnings>`, or `error: <message>` (case failed but the batch continued). Warnings include the **vertebra sanity flags** (`vertebra check: ...`): missing level mid-sequence, spatially inverted labels, one label split across two bones, or a ~2× centroid gap (a physical vertebra the model skipped). Any flagged case deserves visual review. The checks are *internal* consistency only — a labeling uniformly shifted one level (transitional anatomy, T13/L6) passes them and is only caught by overlay review |
| `runtime_s` | wall-clock seconds for the case (segmentation dominates; measurement-only reruns take ~0.1 s) |

All numeric values have 2 decimals; with ~2 mm slices the real uncertainty is
about one slice thickness, so treat sub-millimeter digits as noise.

## Testing without real data

```bash
uv run python make_fake_case.py
uv run python pipeline.py --input fake_case/input --output fake_case/output --test
```

This creates a synthetic patient (60-slice axial DICOM series + a scout series
that must be ignored) and a fake segmentation with known ground truth, including
traps for every cleaning rule. The script prints the expected `results.csv`
values; the pipeline skips TotalSegmentator (the fake segmentation already
exists) and must reproduce them exactly. Delete
`fake_case/output/segs/fake_patient_01.nii.gz` to also exercise real
TotalSegmentator inference on the fake DICOMs (it will find no anatomy in the
synthetic images — expect an `error: ... no lung voxels` row — but it proves the
DICOM → segmentation path runs end to end).
