import numpy as np

def crop_cell(single_seg, raw=None):
    # determine crop roi
    z_range = np.where(np.any(single_seg, axis=(1,2)))
    y_range = np.where(np.any(single_seg, axis=(0,2)))
    x_range = np.where(np.any(single_seg, axis=(0,1)))
    z_range = z_range[0]
    y_range = y_range[0]
    x_range = x_range[0]

    # extra +2 for z top
    roi = [max(z_range[0]-10, 0), min(z_range[-1]+12, single_seg.shape[0]), \
        max(y_range[0]-10, 0), min(y_range[-1]+10, single_seg.shape[1]) , \
        max(x_range[0]-10, 0), min(x_range[-1]+10, single_seg.shape[2]) ]

    # crop seg
    single_seg = single_seg.astype(np.uint8)
    single_seg = single_seg[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]]
    single_seg[single_seg>0]=255

    if raw is None:
        return single_seg
    else:
        crop_raw = raw[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]]
        return single_seg, crop_raw


def find_overlap_cell(seg, single_seg_gt, overlap_ratio=0.51):
    single_seg_size = np.count_nonzero(single_seg_gt)
    overlap_ids = np.unique(seg[single_seg_gt > 0])
    single_seg = None
    for candidate in overlap_ids:
        if candidate == 0:
            continue
        overlap_cell = seg == candidate
        overlap_size = np.count_nonzero(np.logical_and(overlap_cell > 0, single_seg_gt > 0))
        if overlap_size > overlap_ratio * single_seg_size:
            single_seg = overlap_cell
            break

    return single_seg
