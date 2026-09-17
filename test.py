#!/usr/bin/env python
"""Quick napari QC viewer: overlay a multilabel TotalSegmentator mask on its CT.

Usage:
  uv run python test.py --ct output/ct/<id>.nii.gz --seg output/masks/<id>/total.nii.gz [--task total]

Requires the optional viewer extra:  uv sync --extra viz
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ct", type=Path, required=True, help="CT NIfTI (same grid as the mask)")
    ap.add_argument("--seg", type=Path, required=True, help="multilabel mask NIfTI")
    ap.add_argument("--task", default="total", help="TotalSegmentator task name of the mask, for label names")
    args = ap.parse_args()

    import napari
    import nibabel as nib
    import numpy as np
    import pandas as pd
    from totalsegmentator.map_to_binary import class_map

    ct = nib.load(args.ct).get_fdata()
    seg = np.asanyarray(nib.load(args.seg).dataobj).astype(int)

    names = class_map.get(args.task, {})
    feat = pd.DataFrame({"name": [names.get(i, "") for i in range(int(seg.max()) + 1)]})

    v = napari.Viewer()
    v.add_image(ct, name="CT", contrast_limits=(-1000, 400))
    lbl = v.add_labels(seg, name=args.seg.stem, features=feat)
    lbl.contour = 2        # outline only, 2 px thick (0 = back to filled)
    lbl.opacity = 1.0      # contours look best fully opaque
    napari.run()


if __name__ == "__main__":
    main()
