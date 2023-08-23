import numpy as np
import pandas as pd
from aicsimageio import AICSImage 
from aicsimageio.writers import OmeTiffWriter 
from pathlib import Path
from skimage.transform import rescale
from utils import crop_cell, find_overlap_cell


import pandas as pd
from aicsshparam import shtools, shparam
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
# from sklearn import metrics as skmetrics
from aicsimageio import AICSImage
import numpy as np

method = "StarDist-3D" # "EmbedSeg"/"Cellpose"/"StarDist-3D"


fn_csv = method + "/single_cell_compare_seg.csv"
sh_lmax = 16
df = pd.read_csv(fn_csv)
wts_feature = []
man_feature = []
gt_feature = []
for row in df.itertuples():
    wts_path = row.seg_wts
    man_path = row.seg_man
    gt_path = row.seg_gt
    CellId = row.CellId
    wts_seg = AICSImage(wts_path).get_image_data("ZYX", T=0, C=0)
    man_seg = AICSImage(man_path).get_image_data("ZYX", T=0, C=0)
    gt_seg = AICSImage(gt_path).get_image_data("ZYX", T=0, C=0)
    (wts_coeffs, _), _ = shparam.get_shcoeffs(image = wts_seg, lmax = sh_lmax)
    wts_coeffs.update({'volume':np.sum(wts_seg > 0)*np.product((0.108, 0.108, 0.108))})
    wts_coeffs.update({'CellId':CellId}) 
    wts_feature.append(wts_coeffs)
    (man_coeffs, _), _ = shparam.get_shcoeffs(image = man_seg, lmax = sh_lmax)
    man_coeffs.update({'volume':np.sum(man_seg > 0)*np.product((0.108, 0.108, 0.108))})
    man_coeffs.update({'CellId':CellId}) 
    man_feature.append(man_coeffs)
    (gt_coeffs, _), _ = shparam.get_shcoeffs(image = gt_seg, lmax = sh_lmax)
    gt_coeffs.update({'volume':np.sum(gt_seg > 0)*np.product((0.108, 0.108, 0.108))})
    gt_coeffs.update({'CellId':CellId}) 
    gt_feature.append(gt_coeffs)
    wts_coeffs.update({'height': 0.108*(np.where(wts_seg>0)[0][-1]-np.where(wts_seg>0)[0][0])})
    man_coeffs.update({'height': 0.108*(np.where(man_seg>0)[0][-1]-np.where(man_seg>0)[0][0])})
    gt_coeffs.update({'height': 0.108*(np.where(gt_seg>0)[0][-1]-np.where(gt_seg>0)[0][0])})
wts_df = pd.DataFrame(wts_feature)
man_df = pd.DataFrame(man_feature)
gt_df = pd.DataFrame(gt_feature)
gt_df.to_csv(method + "/gt.csv", index=False)
wts_df.to_csv(method + "/wts.csv", index=False)
man_df.to_csv(method + "/man.csv", index=False)