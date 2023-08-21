from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
import os
import numpy as np
from glob import glob
from pathlib import Path

dataset = "Napari-GT"  # "bioGT"/"Napari-GT"/"Slicer-GT"
path_train_data = Path("./train_data/")
path_train_data = path_train_data.joinpath(dataset)
path_train_data.mkdir(parents=True, exist_ok=True)

fns_raw = sorted(glob("../data/raw/dna/*.tiff"))[:20]
fns_gt = sorted(glob(os.path.join("../data/masks/", dataset, "*.tiff")))[:20]


for fn in fns_raw:
    reader_raw = AICSImage(fn)
    raw = reader_raw.get_image_data("ZYX")
    for i in range(raw.shape[0]):
        OmeTiffWriter.save(
            raw[i],
            path_train_data.joinpath(Path(fn).stem + "_" + str(i).zfill(3) + ".tiff"),
            dim_order="YX",
        )

for fn in fns_gt:
    reader_gt = AICSImage(fn)
    gt = reader_gt.get_image_data("ZYX")
    for i in range(gt.shape[0]):
        if np.sum(gt) == 1:  # we need != 1 annotated pixels for training
            gt = np.zeros(gt.shape)
        OmeTiffWriter.save(
            gt[i],
            path_train_data.joinpath(
                Path(fn).stem + "_" + str(i).zfill(3) + "_masks.tiff"
            ),
            dim_order="YX",
        )
