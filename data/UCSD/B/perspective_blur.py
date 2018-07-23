import numpy as np
import string
from skimage import io
import scipy.io as sio
from glob import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter

image_list = list(glob('./*.png'))
sigmadots = 8

pmap = sio.loadmat('vidf1_33_dmap3.mat')
pmap = pmap['dmap'][0][0]
pmap = pmap['pmapxy']

for fname in image_list:
    im = io.imread(fname)
    count = np.count_nonzero(im[:,:,0])
    dot = np.zeros(im[:,:,0].shape, dtype='float32')
    nonzero = np.nonzero(im)
    nonzero = zip(nonzero[0], nonzero[1])
    for (x,y) in nonzero:
        curr_dot = np.zeros(dot.shape, dtype='float32')
        curr_dot[x,y] = 1.0
        dot += gaussian_filter(curr_dot, sigmadots/(np.sqrt(pmap[x,y])))
    countdot = np.sum(dot)
    print('{0}: {1} (8-bit) {2} (summed)'.format(fname,count,countdot))
    fname_out = fname.replace('.png','.npy')
    np.save(fname_out, dot)
