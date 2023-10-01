import numpy as np
import pandas as pd
from aicsimageio import AICSImage 
from aicsimageio.writers import OmeTiffWriter 
from pathlib import Path
from skimage.transform import rescale
from utils import crop_cell, find_overlap_cell

method = "StarDist-3D" # "EmbedSeg"/"Cellpose"/"StarDist-3D" 

#parent_path = Path("data/")
gt_path = Path("../../data/masks/bioGT")
wts_seg_path = Path("../../predictions", method, "bioGT")
man_seg_path = Path("../../predictions", method, "Napari-GT")
raw_path = Path("../../data/raw/dna")
#out_path = parent_path / Path("single_cells")
out_path = Path(method, "single_cells")

out_path.mkdir(parents=True, exist_ok=True)


results = []
filenames = sorted(wts_seg_path.glob("*.tiff"))
for fidx, fn in enumerate(filenames):
    fn_core = fn.name
    gt = AICSImage(gt_path / fn_core).get_image_data("ZYX")
    wts_seg = AICSImage(wts_seg_path / fn_core).get_image_data("ZYX")
    man_seg = AICSImage(man_seg_path / fn_core).get_image_data("ZYX")
    raw_img = AICSImage(fn).get_image_data("ZYX")

    # get all cells
    valid_cells = np.unique(gt[gt > 0])

    # resize to images to isotropic
    gt = rescale(gt, (0.29 / 0.108, 1.0, 1.0), order=0)
    wts_seg = rescale(wts_seg, (0.29 / 0.108, 1.0, 1.0), order=0)
    man_seg = rescale(man_seg, (0.29 / 0.108, 1.0, 1.0), order=0)
    raw_img = rescale(raw_img, (0.29 / 0.108, 1.0, 1.0), order=1)

    boundary_template = np.zeros_like(gt)
    boundary_template[:, :5, :] = 1
    boundary_template[:, -5:, :] = 1
    boundary_template[:, :, :5] = 1
    boundary_template[:, :, -5:] = 1

    bd_idx = list(np.unique(gt[boundary_template > 0]))
    adjusted_mask = gt.copy()
    for idx in bd_idx:
        adjusted_mask[gt == idx] = 0

    gt = np.copy(adjusted_mask)

    # loop through all cells in this image
    for oid, this_cell_index in enumerate(valid_cells):
        this_cell_index = int(this_cell_index)
        cell_id = f"cell_{fidx:02d}_{oid:05d}_{this_cell_index}"
        single_seg_gt = gt == this_cell_index
        single_seg_size = np.count_nonzero(single_seg_gt)

        #single_seg_wts = wts_seg == this_cell_index
        
        # no need to count if the cell is too small
        if single_seg_size < 79384:             ### 100 micron^3
            continue

        # find the corresponding cell in seg_man and seg_wts
        single_seg_man = find_overlap_cell(man_seg, single_seg_gt)    
        if single_seg_man is None:
            print(f"skiping cell {this_cell_index} in {fn_core}, no match found in man")
            continue

        single_seg_wts = find_overlap_cell(wts_seg, single_seg_gt) 
        if single_seg_wts is None:
            print(f"skiping cell {this_cell_index} in {fn_core}, no match found in wts")
            continue

        # prepare the single cell path
        thiscell_path = out_path / Path(cell_id)
        thiscell_path.mkdir(exist_ok=False)

        ################################################
        # for segmentation from watershed based model
        ################################################
        single_seg, crop_raw = crop_cell(single_seg_wts.copy(), raw_img)
        crop_seg_path_wts = thiscell_path / Path('cell_wts.tif')
        out_cell = np.stack((single_seg, crop_raw), axis=0)
        writer = OmeTiffWriter.save(out_cell, crop_seg_path_wts, dim_orders="CZYX")

        ########################################################
        # for segmentation from manual annotation based model
        ########################################################
        single_seg, crop_raw = crop_cell(single_seg_man.copy(), raw_img)
        crop_seg_path_man = thiscell_path / Path('seg_manual.tif')
        out_cell = np.stack((single_seg, crop_raw), axis=0)
        writer = OmeTiffWriter.save(out_cell, crop_seg_path_man, dim_orders="CZYX")

        ########################################################
        # for ground truth
        ########################################################
        single_seg, crop_raw = crop_cell(single_seg_gt.copy(), raw_img)
        crop_seg_path_gt = thiscell_path / Path('seg_gt.tif')
        out_cell = np.stack((single_seg, crop_raw), axis=0)
        writer = OmeTiffWriter.save(out_cell, crop_seg_path_gt, dim_orders="CZYX")

        # generate csv
        results.append({
            "CellId": cell_id,
            "seg_gt": crop_seg_path_gt,
            "seg_wts": crop_seg_path_wts,
            "seg_man": crop_seg_path_man,
        })

out = pd.DataFrame(results)
csv_path = Path(method)
csv_path.mkdir(exist_ok=True)
#out.to_csv(method + "/single_cell_compare_seg.csv", index=False)
out.to_csv(mcsv_path / Path("/single_cell_compare_seg.csv"), index=False)