import pandas as pd
from aicsshparam import shparam
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
# from sklearn import metrics as skmetrics
from aicsimageio import AICSImage
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


fn_csv = "./single_cell_man_vs_bioGT.csv"
sh_lmax = 16
df = pd.read_csv(fn_csv)
bioGT_feature = []
napari_gt_feature = []
slicer_gt_feature = []
for row in df.itertuples():
    bioGT_path = row.seg_bioGT
    napari_gt_path = row.seg_napari_gt
    slicer_gt_path = row.seg_slicer_gt
    CellId = row.CellId

    bioGT_seg = AICSImage(bioGT_path).get_image_data("ZYX", T=0, C=0)
    napari_gt_seg = AICSImage(napari_gt_path).get_image_data("ZYX", T=0, C=0)
    slicer_gt_seg = AICSImage(slicer_gt_path).get_image_data("ZYX", T=0, C=0)

    (bioGT_coeffs, _), _ = shparam.get_shcoeffs(image = bioGT_seg, lmax = sh_lmax)
    bioGT_coeffs.update({'volume':np.sum(bioGT_seg > 0)*np.product((0.108, 0.108, 0.108))})
    bioGT_coeffs.update({'CellId':CellId}) 
    bioGT_feature.append(bioGT_coeffs)

    (napari_gt_coeffs, _), _ = shparam.get_shcoeffs(image = napari_gt_seg, lmax = sh_lmax)
    napari_gt_coeffs.update({'volume':np.sum(napari_gt_seg > 0)*np.product((0.108, 0.108, 0.108))})
    napari_gt_coeffs.update({'CellId':CellId}) 
    napari_gt_feature.append(napari_gt_coeffs)

    (slicer_gt_coeffs, _), _ = shparam.get_shcoeffs(image = slicer_gt_seg, lmax = sh_lmax)
    slicer_gt_coeffs.update({'volume':np.sum(slicer_gt_seg > 0)*np.product((0.108, 0.108, 0.108))})
    slicer_gt_coeffs.update({'CellId':CellId}) 
    slicer_gt_feature.append(slicer_gt_coeffs)    

bioGT_df = pd.DataFrame(bioGT_feature)
napari_gt_df = pd.DataFrame(napari_gt_feature)
slicer_gt_df = pd.DataFrame(slicer_gt_feature)

bioGT_df.to_csv("./csvs/bioGT.csv", index=False)
napari_gt_df.to_csv("./csvs/napari_gt.csv", index=False)
slicer_gt_df.to_csv("./csvs/slicer_gt.csv", index=False)