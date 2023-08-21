# THIS FILE IS ADAPTED FROM THE ORIGINIAL CELLPOSE REPO https://github.com/MouseLand/cellpose

from cellpose import core, io, models
from glob import glob
from aicsimageio.writers import OmeTiffWriter
from pathlib import Path


dataset = "Napari-GT"  # "bioGT"/"Napari-GT"

use_GPU = core.use_gpu()

# get image data
filenames = sorted(glob("../data/raw/dna/*.tiff"))[-18:]    # we use only the last 18 images for inference, see manuscript
path_preds = Path("../predictions/Cellpose/", dataset)
path_preds.mkdir(parents=True, exist_ok=True)


if dataset == "bioGT":
    diameter = 62.802
elif dataset == "Napari-GT":
    diameter = 102.3
# elif dataset == 'Slicer-GT':  # We don't provide a model for this dataset, but you can train your own using the provides masks
#     diameter = 98.140
else:
    raise ValueError("Dataset not recognized")

# declare model
model = models.CellposeModel(gpu=True, pretrained_model="../Models/Cellpose/" + dataset)

# run model on test images
for fn in filenames:
    image = io.imread(fn)
    masks, _, _ = model.eval([image], channels=[0, 0], diameter=diameter, do_3D=True)
    out_path = path_preds / Path(fn).name
    OmeTiffWriter.save(masks[0], out_path, dim_order="ZYX")
