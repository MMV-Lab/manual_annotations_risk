# THIS FILE IS ADAPTED FROM THE ORIGINIAL STARDIST REPO https://github.com/stardist/stardist

from __future__ import print_function, unicode_literals, absolute_import, division
from glob import glob
from tifffile import imread
from csbdeep.utils import Path, normalize
from csbdeep.io import save_tiff_imagej_compatible

from stardist.models import StarDist3D
from pathlib import Path

dataset = "Napari-GT"  # "bioGT"/"Napari-GT"
path_preds = Path("../predictions/StarDist-3D/")
path_preds = path_preds.joinpath(dataset)
path_preds.mkdir(parents=True, exist_ok=True)

# DATA
files = sorted(glob("../data/raw/dna/*.tiff"))[-18:]    # we use only the last 18 images for inference, see manuscript
X = list(map(imread, files))
axis_norm = (0, 1, 2)

# LOAD MODEL
model = StarDist3D(None, name="wts", basedir="../models/StarDist/")

# INFERENCE + SAVE PREDICTED MASKS
for i in range(len(X)):
    img = normalize(X[i], 1, 99.8, axis=axis_norm)
    labels, details = model.predict_instances(img)
    save_tiff_imagej_compatible(
        path_preds.joinpath(Path(files[i]).name), labels, axes="ZYX"
    )
