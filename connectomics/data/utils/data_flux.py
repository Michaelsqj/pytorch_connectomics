"""
1. distance transform + flux
2. distance to center + flux
"""
import numpy as np
from skimage.morphology import skeletonize_3d
from scipy.ndimage import center_of_mass
from scipy.ndimage.morphology import distance_transform_edt
from .data_segmentation import seg_to_instance_bd, fix_dup_ind, one_hot


def flux_center(volume):
    volume = fix_dup_ind(volume)
    A, B, C = np.zeros(volume.shape, int), np.zeros(volume.shape, int), np.zeros(volume.shape, int)
    for i in np.unique(volume):
        if i == 0:
            continue
        v = (volume == i).astype(np.uint8)
        center = np.round(center_of_mass(v))
        a, b, c = np.meshgrid(range(v.shape[0]), range(v.shape[1]), range(v.shape[2]), indexing='ij')
        a, b, c = v * (a - center[0]), v * (b - center[1]), v * (c - center[2])
        A, B, C = A + a, B + b, C + c
    return np.stack((A, B, C), axis=0)


def flux_border_2d(volume, opt=0):
    # angle or neighborhood
    # opt 0: angle, regression   opt 1: neighborhood, classification
    assert opt in [0, 1]
    border = seg_to_instance_bd(volume, 2, True)
    D = volume.shape[0]
    # direction vector [-1,1]
    dega = np.zeros(volume.shape, dtype=float)
    degb = np.zeros(volume.shape, dtype=float)
    for d in range(D):
        s = fix_dup_ind(volume[d, ...])
        for i in np.unique(volume[d, ...]):
            if i == 0:
                continue
            a, b = np.meshgrid(range(volume.shape[1]), range(volume.shape[2]), indexing='ij')
            _, indices = distance_transform_edt(s == i, return_indices=True)
            indices = indices * (volume[d, ...] == i)
            # indices 2xHxW
            inda, indb = indices[0], indices[1]
            dega += (s == i) * (inda - a) / np.sqrt((inda - a) * (inda - a) + (indb - b) * (indb - b))
            degb += (s == i) * (indb - b) / np.sqrt((inda - a) * (inda - a) + (indb - b) * (indb - b))
    if opt == 0:
        return np.stack([dega, degb], axis=0)
    else:
        deg = np.stack([dega, degb], axis=0)
        direct = np.array([[1, 0], [0.707, 0.707], [0, 1], [-0.707, 0.707],
                           [-1, 0], [-0.707, -0.707], [0, -1], [0.707, -0.707]])
        m = np.stack([deg[0] * direct[i][0] for i in range(8)], axis=0)
        n = np.stack([deg[1] * direct[i][1] for i in range(8)], axis=0)
        return one_hot((volume > 0) * (np.argmax(m + n, axis=0) + 1), 9)


def flux_z(volume):
    # 0,1,2
    # 0: background, 1: below center 2: above center
    volume = fix_dup_ind(volume)
    out = np.zeros(volume.shape, dtype=np.uint8)
    z, _, _ = np.meshgrid(range(volume.shape[0]), range(volume.shape[1]), range(volume.shape[2]), indexing='ij')
    for i in np.unique(volume):
        if i == 0:
            continue
        center = np.round(center_of_mass((volume == i)))[0]
        temp = (volume == i) * (z - center)
        out += 1 * (temp < 0) + 2 * (temp > 0)
    out = one_hot(out, 3)
    return out
