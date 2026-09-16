import nibabel as nib
import numpy as np
import napari
import pandas as pd
from totalsegmentator.map_to_binary import class_map

ct = nib.load("/Volumes/WD MB/MGH/ct/00982415.nii.gz").get_fdata()
seg = nib.load("/Volumes/WD MB/MGH/segs/00982415.nii.gz").get_fdata().astype(int)

names = class_map["total"]
feat = pd.DataFrame({"name": [names.get(i, "") for i in range(int(seg.max()) + 1)]})

v = napari.Viewer()
v.add_image(ct, name="CT", contrast_limits=(-1000, 400))
v.add_labels(seg, name="seg", features=feat)
lbl = v.add_labels(seg, name="seg", features=feat)
lbl.contour = 2        # outline only, 2 px thick (0 = back to filled)
lbl.opacity = 1.0      # contours look best fully opaque

napari.run()
