"""
1. distance transform + flux
2. distance to center + flux
"""
import numpy as np
from skimage.morphology import skeletonize_3d
from scipy.ndimage import center_of_mass, measurements
from scipy.ndimage.morphology import distance_transform_edt


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
    D = volume.shape[0]
    volume = volume.astype(np.uint16)
    # direction vector [-1,1]
    dega = np.zeros(volume.shape, dtype=float)
    degb = np.zeros(volume.shape, dtype=float)
    for d in range(D):
        s = fix_dup_ind(volume[d, ...])
        for i in np.unique(s):
            if i == 0:
                continue
            a, b = np.meshgrid(range(volume.shape[1]), range(volume.shape[2]), indexing='ij')
            _, indices = distance_transform_edt(s == i, return_indices=True)
            # indices 2xHxW
            indices = indices * (s == i)
            inda, indb = indices[0], indices[1]
            if opt == 0:
                temp = np.sqrt((inda - a) ** 2 + (indb - b) ** 2)
                tempa = np.divide(inda - a, temp, where=temp > 0, out=np.zeros_like(inda, dtype=float))
                tempb = np.divide(indb - b, temp, where=temp > 0, out=np.zeros_like(indb, dtype=float))
                dega[d, ...] += (s == i) * tempa
                degb[d, ...] += (s == i) * tempb
            else:
                dega[d, ...] += (s == i) * (inda-a)
                degb[d, ...] += (s == i) * (indb-b)
    return np.stack([dega, degb], axis=0)


def flux_z(volume):
    # 0,1,2
    # 0: background, 1: below center 2: above center
    volume = volume.astype(np.uint16)
    volume = fix_dup_ind(volume)
    out = np.zeros(volume.shape, dtype=np.uint8)
    z, _, _ = np.meshgrid(range(volume.shape[0]), range(volume.shape[1]), range(volume.shape[2]), indexing='ij')
    for i in np.unique(volume):
        if i == 0:
            continue
        center = np.round(center_of_mass((volume == i)))[0]
        temp = (volume == i) * (z - center)
        out += 1 * (temp < 0).astype(np.uint8) + 2 * (temp > 0).astype(np.uint8)
    out = one_hot(out, 3, leave_bg=True)
    return out


def seg_to_instance_bd(seg, tsz_h=7, do_bg=False):
    tsz = tsz_h * 2 + 1
    mm = seg.max()
    sz = seg.shape
    bd = np.zeros(sz, np.uint8)
    for z in range(sz[0]):
        patch = im2col(np.pad(seg[z], ((tsz_h, tsz_h), (tsz_h, tsz_h)), 'reflect'), [tsz, tsz])
        p0 = patch.max(axis=1)
        if do_bg:  # at least one non-zero seg
            p1 = patch.min(axis=1)
            bd[z] = ((p0 > 0) * (p0 != p1)).reshape(sz[1:])
        else:  # between two non-zero seg
            patch[patch == 0] = mm + 1
            p1 = patch.min(axis=1)
            bd[z] = ((p0 != 0) * (p1 != 0) * (p0 != p1)).reshape(sz[1:])
    return bd


def im2col(A, BSZ, stepsize=1):
    # Parameters
    M, N = A.shape
    # Get Starting block indices
    start_idx = np.arange(0, M - BSZ[0] + 1, stepsize)[:, None] * N + np.arange(0, N - BSZ[1] + 1, stepsize)
    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(BSZ[0])[:, None] * N + np.arange(BSZ[1])
    # Get all actual indices & index into input array for final output
    return np.take(A, start_idx.ravel()[:, None] + offset_idx.ravel())


def fix_dup_ind(ann):
    """
    deal with duplicated instance
    """
    current_max_id = np.amax(ann)
    inst_list = list(np.unique(ann))
    inst_list.remove(0)  # 0 is background
    for inst_id in inst_list:
        inst_map = np.array(ann == inst_id, np.uint8)
        remapped_ids = measurements.label(inst_map)[0]
        remapped_ids[remapped_ids > 1] += current_max_id
        ann[remapped_ids > 1] = remapped_ids[remapped_ids > 1]
        current_max_id = np.amax(ann)
    return ann


def one_hot(a, bins: int, leave_bg=False):
    shape = a.shape
    a = a.reshape(-1, 1).squeeze()
    out = np.zeros((a.size, bins), dtype=a.dtype)
    out[np.arange(a.size), a] = 1
    out = out.reshape(list(shape) + [bins])
    out = np.transpose(out, (3, 0, 1, 2))
    # CDHW
    # leave out the channel 0 background
    if leave_bg:
        out = out[1:]
    return out
