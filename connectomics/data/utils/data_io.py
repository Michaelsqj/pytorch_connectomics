import h5py
import os, sys
import glob
import numpy as np
import imageio
from scipy.ndimage import zoom

def readh5(filename, dataset=''):
    fid = h5py.File(filename,'r')
    if dataset=='':
        dataset = list(fid)[0]
    return np.array(fid[dataset])

def readvol(filename, dataset=''):
    img_suf = filename[filename.rfind('.')+1:]
    if img_suf == 'h5':
        data = readh5(filename, dataset)
    elif 'tif' in img_suf:
        data = imageio.volread(filename).squeeze()
    elif 'png' in img_suf:
        data = readimgs(filename)
    else:
        raise ValueError('unrecognizable file format for %s'%(filename))
    return data

def savevol(filename, vol, dataset='main', format='h5'):
    if format == 'h5':
        writeh5(filename, vol, dataset='main')
    if format == 'png':
        currentDirectory = os.getcwd()
        img_save_path = os.path.join(currentDirectory, filename)
        if not os.path.exists(img_save_path):
            os.makedirs(img_save_path)
        for i in range(vol.shape[0]):
            imageio.imsave('%s/%04d.png' % (img_save_path, i), vol[i])

def readim(filename, do_channel=False):
    # x,y,c
    if not os.path.exists(filename):
        im = None
    else:# note: cv2 do "bgr" channel order
        im = imageio.imread(filename)
        if do_channel and im.ndim==2:
            im=im[:,:,None]
    return im

def readimgs(filename):
    filelist = sorted(glob.glob(filename))
    num_imgs = len(filelist)

    # decide numpy array shape:
    img = imageio.imread(filelist[0])
    data = np.zeros((num_imgs, img.shape[0], img.shape[1]), dtype=np.uint8)
    data[0] = img

    # load all images
    if num_imgs > 1:
        for i in range(1, num_imgs):
            data[i] = imageio.imread(filelist[i])

    return data

def writeh5(filename, dtarray, dataset='main'):
    fid=h5py.File(filename,'w')
    if isinstance(dataset, (list,)):
        for i,dd in enumerate(dataset):
            ds = fid.create_dataset(dd, dtarray[i].shape, compression="gzip", dtype=dtarray[i].dtype)
            ds[:] = dtarray[i]
    else:
        ds = fid.create_dataset(dataset, dtarray.shape, compression="gzip", dtype=dtarray.dtype)
        ds[:] = dtarray
    fid.close()

def act(pred, topt):
    # pred: torch tensor / numpy array
    is_tensor = (type(pred)==torch.Tensor)
    act_fn={'0': torch.sigmoid,'1':torch.sigmoid,'2':torch.sigmoid,'3':torch.sigmoid,'4':torch.sigmoid,
            '5':torch.sigmoid,'6':lambda x: x,'7':torch.sigmoid,'8':torch.sigmoid, '9':torch.tanh}
    act_fn_np={'0': sigmoid,'1':sigmoid,'2':sigmoid,'3':sigmoid,'4':sigmoid,
            '5':sigmoid,'6':lambda x: x,'7':sigmoid,'8':sigmoid, '9':tanh}
    num_channels_dict = {'0': 1, '1': 3, '2': 3, '3': 1, '4': 1, '5': 10, '6': 2, '7': 3, '8': 1, '9': 2}
    cid = 0
    for i in range(len(topt)):
        # for each target
        numC = num_channels_dict[topt[i]]
        #  add activation function
        if is_tensor:
            pred[:, cid:cid+numC] = act_fn[topt[i]](pred[:, cid:cid+numC])
        else:
            pred[:, cid:cid+numC] = act_fn_np[topt[i]](pred[:, cid:cid+numC])
    return pred

def squeeze(pred, topt):
    # pred: numpy array
    cid=0
    pred_channels = {'0': 1, '1': 3, '2': 3, '3': 1, '4': 1, '5': 10, '6': 2, '7': 3, '8': 1, '9': 2}
    out_channels = {'0': 1, '1': 3, '2': 3, '3': 1, '4': 1, '5': 1, '6': 2, '7': 1, '8': 1, '9': 2}
    for t in topt:
        numC=pred_channels[topt[i]]
        v = pred[:, cid: cid+numC]
        if pred_channels[topt[i]]>out_channels[topt[i]]:
            v=np.argmax(v, axis=1)
            result.append(np.expand_dims(v, axis=1))
        cid+=numC[t]
    return np.concatenate(result, axis=1)

####################################################################
## tile to volume
####################################################################
def vast2Seg(seg):
    # convert to 24 bits
    if seg.ndim==2:
        return seg
    else: #vast: rgb
        return seg[:,:,0].astype(np.uint32)*65536+seg[:,:,1].astype(np.uint32)*256+seg[:,:,2].astype(np.uint32)

def tileToVolume(tiles, coord, coord_m, tile_sz, dt=np.uint8, tile_st=[0,0], tile_ratio=1, do_im=True, ndim=1, black=128):
    # x: column
    # y: row
    # no padding at the boundary
    # st: starting index 0 or 1
    z0o, z1o, y0o, y1o, x0o, x1o = coord # region to crop
    z0m, z1m, y0m, y1m, x0m, x1m = coord_m # tile boundary

    bd = [max(-z0o,z0m), max(0,z1o-z1m), max(-y0o,y0m), max(0,y1o-y1m), max(-x0o,x0m), max(0,x1o-x1m)]
    z0, y0, x0 = max(z0o,z0m), max(y0o,y0m), max(x0o,x0m)
    z1, y1, x1 = min(z1o,z1m), min(y1o,y1m), min(x1o,x1m)

    result = black*np.ones((z1-z0, y1-y0, x1-x0), dt)
    c0 = x0 // tile_sz # floor
    c1 = (x1 + tile_sz-1) // tile_sz # ceil
    r0 = y0 // tile_sz
    r1 = (y1 + tile_sz-1) // tile_sz
    for z in range(z0, z1):
        pattern = tiles[z]
        for row in range(r0, r1):
            for column in range(c0, c1):
                if '{' in pattern:
                    path = pattern.format(row=row+tile_st[0], column=column+tile_st[1])
                else:
                    path = pattern
                patch = readim(path, do_channel=True)
                if patch is not None:
                    if tile_ratio != 1: # im ->1, label->0
                        patch = zoom(patch, [tile_ratio,tile_ratio,1], order=int(do_im))

                    # last tile may not be of the same size
                    xp0 = column * tile_sz
                    xp1 = xp0 + patch.shape[1]
                    yp0 = row * tile_sz
                    yp1 = yp0 + patch.shape[0]
                    x0a = max(x0, xp0)
                    x1a = min(x1, xp1)
                    y0a = max(y0, yp0)
                    y1a = min(y1, yp1)
                    if do_im:# image
                        result[z-z0, y0a-y0:y1a-y0, x0a-x0:x1a-x0] = patch[y0a-yp0:y1a-yp0, x0a-xp0:x1a-xp0,0]
                    else: # label
                        result[z-z0, y0a-y0:y1a-y0, x0a-x0:x1a-x0] = vast2Seg(patch[y0a-yp0:y1a-yp0, x0a-xp0:x1a-xp0])
    if max(bd)>0:
        result = np.pad(result,((bd[0], bd[1]),(bd[2], bd[3]),(bd[4], bd[5])),'reflect')
    return result
