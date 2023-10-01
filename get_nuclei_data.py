from pathlib import Path
import quilt3
import shutil
import pandas as pd
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter

cell_line_name = "LMNB1"
id_list = [
    1521,
    1530,
    1538,
    1541,
    1547,
    1550,
    1565,
    1589,
    1590,
    1608,
    1609,
    1612,
    1617,
    1618,
    1619,
    1620,
    1623,
    1625,
    1639,
    1641,
    1647,
    1648,
    1664,
    1666,
    1669,
    1671,
    1684,
    1687,
    1698,
    1700,
    1702,
    1703,
    1704,
    1706,
    1709,
    1717,
    1722,
    1723,
    1725,
    1729,
    1746,
    1747,
    1749,
    1753,
    1757,
    1767,
    1771,
    1777,
    1781,
    1781,
    1791,
    1802,
    1807,
    1822,
]
parent_path = Path("./data")  # add your path here
parent_path.mkdir(exist_ok=True)

# prepare file path
tmp_path = parent_path / Path("tmp")
tmp_path.mkdir(exist_ok=True)
raw_path = parent_path / Path("raw")
raw_path.mkdir(exist_ok=True)
dna_path = raw_path / Path("dna")
dna_path.mkdir(exist_ok=True)
lamin_b1_path = raw_path / Path("lamin_b1")
lamin_b1_path.mkdir(exist_ok=True)

# connect to quilt and load meta table
pkg = quilt3.Package.browse(
    "aics/hipsc_single_cell_image_dataset", registry="s3://allencell"
)
meta_df_obj = pkg["metadata.csv"]
meta_df_obj.fetch(parent_path / "meta.csv")
meta_df = pd.read_csv(parent_path / "meta.csv")


# fetch the data of the specific cell line
meta_df_line = meta_df.query("structure_name==@cell_line_name")

# collapse the data table based on FOVId
meta_df_line.drop_duplicates(subset="FOVId", inplace=True)

# reset index
meta_df_line.reset_index(drop=True, inplace=True)

# get all data with specified FOVId
df_fov = meta_df_line.query("FOVId in @id_list")
for row in df_fov.itertuples():
    # fetch the raw image
    subdir_name = row.fov_path.split("/")[0]
    file_name = row.fov_path.split("/")[1]

    local_fn = tmp_path / f"{row.FOVId}_original.tiff"
    pkg[subdir_name][file_name].fetch(local_fn)

    # extract the structures channel (lamin B1) and the DNA channel
    reader = AICSImage(local_fn)
    str_img = reader.get_image_data("ZYX", C=row.ChannelNumberStruct, T=0)
    dna_img = reader.get_image_data("ZYX", C=row.ChannelNumber405, T=0)

    # save the images
    im_fn = lamin_b1_path / f"{row.FOVId}.tiff"
    OmeTiffWriter.save(str_img, im_fn, dim_order="ZYX")

    im_fn = dna_path / f"{row.FOVId}.tiff"
    OmeTiffWriter.save(dna_img, im_fn, dim_order="ZYX")

# remove temp path
shutil.rmtree(tmp_path)