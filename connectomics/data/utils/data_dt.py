from scipy import ndimage
import numpy as np
from .data_segmentation import fix_dup_ind, one_hot


def dt_3d(seg, res, bins):
    out = np.zeros(seg.shape, dtype=np.float32)
    for i in np.unique(seg):
        if i == 0:
            continue
        tmp = (seg == i).astype(np.uint8)
        tmp = ndimage.distance_transform_edt(tmp, sampling=res)
        tmp[tmp >= bins] = bins - 1
        out += tmp
    out = one_hot(out, bins=bins)
    return out


def dt_2d(volume, bins):
    # volume: instance map, D x H x W
    out = np.zeros(volume.shape, dtype=float)
    D = volume.shape[0]
    for d in range(D):
        s = volume[d, ...]
        s = fix_dup_ind(s)
        for i in np.unique(s):
            if i == 0:
                continue
            out[d, ...] += ndimage.distance_transform_edt((s == i))
    out = out.astype(np.uint8)
    out = one_hot(out, bins)
    return out
