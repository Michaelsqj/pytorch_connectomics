from scipy.ndimage import measurements, distance_transform_edt
import numpy as np


def dt_3d(seg, res, bins):
    out = np.zeros(seg.shape, dtype=np.float32)
    for i in np.unique(seg):
        if i == 0:
            continue
        tmp = (seg == i).astype(np.uint8)
        tmp = distance_transform_edt(tmp, sampling=res)
        tmp[tmp >= bins] = bins - 1
        out += tmp
    out = one_hot(out, bins=bins)
    return out


def dt_2d(volume, bins):
    # volume: instance map, D x H x W, dtype uint16
    volume = volume.astype(np.uint16)
    out = np.zeros(volume.shape, dtype=float)
    D = volume.shape[0]
    for d in range(D):
        s = volume[d, ...]
        s = fix_dup_ind(s)
        for i in np.unique(s):
            if i == 0:
                continue
            out[d, ...] += distance_transform_edt((s == i))
    out = out.astype(np.uint8)
    out = one_hot(out, bins)
    return out


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


def one_hot(a, bins: int):
    shape=a.shape
    a = a.reshape(-1, 1).squeeze()
    out = np.zeros((a.size, bins), dtype=a.dtype)
    out[np.arange(a.size), a] = 1
    out = out.reshape(list(shape) + [bins])
    out = np.transpose(out, (3, 0, 1, 2))
    # CDHW
    return out
