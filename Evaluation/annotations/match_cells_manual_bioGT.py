import numpy as np
import pandas as pd
from aicsimageio import AICSImage 
from aicsimageio.writers import OmeTiffWriter 
from pathlib import Path
from skimage.transform import rescale
from utils import crop_cell, find_overlap_cell



bioGT_seg_path = Path("../../data/masks/bioGT/")
napari_gt_seg_path = Path("../../data/masks/Napari-GT/")
slicer_gt_seg_path = Path("../../data/masks/Slicer-GT/")
raw_path = Path("../../data/raw/dna/")
out_path = Path("./single_cells")


results = []
filenames = sorted(napari_gt_seg_path.glob("*.tiff"))
for fidx, fn in enumerate(filenames):
    fn_core = fn.name
    bioGT_seg = AICSImage(bioGT_seg_path / fn_core).get_image_data("ZYX")
    napari_gt_seg = AICSImage(napari_gt_seg_path / fn_core).get_image_data("ZYX")
    slicer_gt_seg = AICSImage(slicer_gt_seg_path / fn_core).get_image_data("ZYX")
    raw_img = AICSImage(fn).get_image_data("ZYX")

    boundary_template = np.zeros_like(bioGT_seg)
    boundary_template[:, :5, :] = 1
    boundary_template[:, -5:, :] = 1
    boundary_template[:, :, :5] = 1
    boundary_template[:, :, -5:] = 1

    bd_idx = list(np.unique(bioGT_seg[boundary_template > 0]))
    adjusted_mask = bioGT_seg.copy()
    for idx in bd_idx:
        adjusted_mask[bioGT_seg == idx] = 0

    bioGT_seg = np.copy(adjusted_mask)


    # get all cells
    valid_cells = np.unique(bioGT_seg[bioGT_seg > 0])

    # resize to images to isotropic
    bioGT_seg = rescale(bioGT_seg, (0.29 / 0.108, 1.0, 1.0), order=0)
    napari_gt_seg = rescale(napari_gt_seg, (0.29 / 0.108, 1.0, 1.0), order=0)
    slicer_gt_seg = rescale(slicer_gt_seg, (0.29 / 0.108, 1.0, 1.0), order=0)    
    raw_img = rescale(raw_img, (0.29 / 0.108, 1.0, 1.0), order=1)

    # loop through all cells in this image
    for oid, this_cell_index in enumerate(valid_cells):
        this_cell_index = int(this_cell_index)
        cell_id = f"cell_{fidx:02d}_{oid:05d}_{this_cell_index}"
        single_seg_bioGT = bioGT_seg == this_cell_index
        single_seg_size = np.count_nonzero(single_seg_bioGT)

        # no need to count if the cell is too small
        if single_seg_size < 79384: #500:
            continue

        # find the corresponding cell in napari_gt_seg
        single_seg_napari_gt = find_overlap_cell(napari_gt_seg, single_seg_bioGT)     
        if single_seg_napari_gt is None:
            print(f"skiping cell {this_cell_index}, no match found in Napari-GT")
            continue

        # find the corresponding cell in slicer_gt_seg
        single_seg_slicer_gt = find_overlap_cell(slicer_gt_seg, single_seg_bioGT)     
        if single_seg_slicer_gt is None:
            print(f"skiping cell {this_cell_index}, no match found in Napari-GT")
            continue

        # prepare the single cell path
        thiscell_path = out_path / Path(cell_id)
        thiscell_path.mkdir(exist_ok=False, parents=True)

        ################################
        # for bioGT
        ################################
        single_seg, crop_raw = crop_cell(single_seg_bioGT.copy(), raw_img)
        crop_seg_path_bioGT = thiscell_path / Path('cell_bioGT.tiff')
        out_cell = np.stack((single_seg, crop_raw), axis=0)
        writer = OmeTiffWriter.save(out_cell, crop_seg_path_bioGT, dim_orders="CZYX")

        #####################
        # for Napari-GT
        #####################
        #import pdb; pdb.set_trace()
        single_seg, crop_raw = crop_cell(single_seg_napari_gt.copy(), raw_img)
        crop_seg_path_napari_gt = thiscell_path / Path('seg_napari-gt.tiff')
        out_cell = np.stack((single_seg, crop_raw), axis=0)
        writer = OmeTiffWriter.save(out_cell, crop_seg_path_napari_gt, dim_orders="CZYX")


        #####################
        # for Slicer-GT
        #####################
        single_seg, crop_raw = crop_cell(single_seg_slicer_gt.copy(), raw_img)
        crop_seg_path_slicer_gt = thiscell_path / Path('seg_slicer-gt.tiff')
        out_cell = np.stack((single_seg, crop_raw), axis=0)
        writer = OmeTiffWriter.save(out_cell, crop_seg_path_slicer_gt, dim_orders="CZYX")

        # generate csv
        results.append({
            "CellId": cell_id,
            "seg_bioGT": crop_seg_path_bioGT,
            "seg_napari_gt": crop_seg_path_napari_gt,
            "seg_slicer_gt": crop_seg_path_slicer_gt,
        })

out = pd.DataFrame(results)
csv_path = Path('./csvs')
csv_path.mkdir()
#out.to_csv("./csvs/single_cell_man_vs_bioGT.csv", index=False)
out.to_csv(csv_path / Path("single_cell_man_vs_bioGT.csv"), index=False)