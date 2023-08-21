# THIS FILE IS ADAPTED FROM THE ORIGINIAL STARDIST REPO https://github.com/stardist/stardist

from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from glob import glob
import os
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize

from stardist import (
    fill_label_holes,
    random_label_cmap,
    calculate_extents,
    gputools_available,
)
from stardist import Rays_GoldenSpiral
from stardist.models import Config3D, StarDist3D

np.random.seed(42)
lbl_cmap = random_label_cmap()

dataset = "Napari-GT"  # "bioGT"/"Napari-GT"/"Slicer-GT"

# DATA
X = sorted(glob("../data/raw/dna/*.tiff"))[:20]
Y = sorted(glob(os.path.join("../data/masks/", dataset, "*.tiff")))[:20]
assert all(Path(x).name == Path(y).name for x, y in zip(X, Y))

X = list(map(imread, X))
Y = list(map(imread, Y))
n_channel = 1 if X[0].ndim == 3 else X[0].shape[-1]

# NORMALIZATION
axis_norm = (0, 1, 2)
# if n_channel > 1:
#     print(
#         "Normalizing image channels %s."
#         % ("jointly" if axis_norm is None or 3 in axis_norm else "independently")
#     )
#     sys.stdout.flush()
X = [normalize(x, 1, 99.8, axis=axis_norm) for x in tqdm(X)]
Y = [fill_label_holes(y) for y in tqdm(Y)]

# TRAIN + VALIDATION SPLIT
assert len(X) > 1, "not enough training data"
rng = np.random.RandomState(42)
ind = rng.permutation(len(X))
n_val = max(1, int(round(0.15 * len(ind))))
ind_train, ind_val = ind[:-n_val], ind[-n_val:]
X_val, Y_val = [X[i] for i in ind_val], [Y[i] for i in ind_val]
X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train]
print("number of images: %3d" % len(X))
print("- training:       %3d" % len(X_trn))
print("- validation:     %3d" % len(X_val))

# CONFIG
extents = calculate_extents(Y)
anisotropy = tuple(np.max(extents) / extents)
print("empirical anisotropy of labeled objects = %s" % str(anisotropy))

n_rays = 96
use_gpu = False and gputools_available()
# Predict on subsampled grid for increased efficiency and larger field of view
grid = tuple(1 if a > 1.5 else 2 for a in anisotropy)

# Use rays on a Fibonacci lattice adjusted for measured anisotropy of the training data
rays = Rays_GoldenSpiral(n_rays, anisotropy=anisotropy)

conf = Config3D(
    rays=rays,
    grid=grid,
    anisotropy=anisotropy,
    use_gpu=use_gpu,
    n_channel_in=n_channel,
    train_patch_size=(32, 256, 256),
    train_batch_size=2,
)
print(conf)
vars(conf)

if use_gpu:
    from csbdeep.utils.tf import limit_gpu_memory
    limit_gpu_memory(0.8)

model = StarDist3D(conf, name=dataset, basedir="models")

# AUGMENTATION
def random_fliprot(img, mask, axis=None):
    if axis is None:
        axis = tuple(range(mask.ndim))
    axis = tuple(axis)

    assert img.ndim >= mask.ndim
    perm = tuple(np.random.permutation(axis))
    transpose_axis = np.arange(mask.ndim)
    for a, p in zip(axis, perm):
        transpose_axis[a] = p
    transpose_axis = tuple(transpose_axis)
    img = img.transpose(transpose_axis + tuple(range(mask.ndim, img.ndim)))
    mask = mask.transpose(transpose_axis)
    for ax in axis:
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask


def random_intensity_change(img):
    img = img * np.random.uniform(0.6, 2) + np.random.uniform(-0.2, 0.2)
    return img


def augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    # Note that we only use fliprots along axis=(1,2), i.e. the yx axis
    # as 3D microscopy acquisitions are usually not axially symmetric
    x, y = random_fliprot(x, y, axis=(1, 2))
    x = random_intensity_change(x)
    return x, y


# TRAINING + THRESHOLD OPTIMIZATION
model.train(
    X_trn, Y_trn, validation_data=(X_val, Y_val), augmenter=augmenter, epochs=1
)
model.optimize_thresholds(X_val, Y_val)
