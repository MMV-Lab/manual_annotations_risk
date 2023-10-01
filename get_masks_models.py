import pooch
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
import matplotlib.pyplot as plt
import zipfile
from pathlib import Path
from random import random
import numpy as np


data_path = Path("./tmp")
p = data_path / Path("download")
p.mkdir(exist_ok=True, parents=True)
p = data_path / Path("train")
p.mkdir(exist_ok=True)
p = data_path / Path("test")
p.mkdir(exist_ok=True)




source_part1 = pooch.retrieve(
    url="https://doi.org/10.5281/zenodo.8247136",
    known_hash=None,
    fname="test.zip",
    path=data_path / Path("download")
)

