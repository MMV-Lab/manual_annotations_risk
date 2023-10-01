import pooch
import os
import zipfile
from pathlib import Path
from shutil import rmtree

tmp_path = Path("./tmp")
tmp_path.mkdir(exist_ok=True)

# download the data
source_part1 = pooch.retrieve(
    url="https://zenodo.org/record/8247136/files/masks_and_models.zip?download=1",
    known_hash="53329d2dad5546cf9d7dcda2587c3d46e2266bfd0af14eecf1a74ef3b0db34e0",
    fname="test.zip",
    path=tmp_path / Path("download")
)

# unzip the data
with zipfile.ZipFile(source_part1,"r") as zip_ref:
    zip_ref.extractall("./")

# removee temp path
rmtree(tmp_path, ignore_errors=True)

# move masks to data folder
Path('data').mkdir(exist_ok=True)
os.rename('masks', 'data/masks')