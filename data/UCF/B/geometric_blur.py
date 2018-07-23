import numpy as np
import string
from skimage import io
from glob import glob
from scipy.ndimage.filters import gaussian_filter

image_list = list(glob('./*.png'))
sigmadots = 15

for fname in image_list:
    im = io.imread(fname)
    count = np.count_nonzero(im[:,:,0])
    im32 = (im[:,:,0]/255).astype('float32')
    count32 = np.count_nonzero(im32)
    dot = gaussian_filter(im32, sigmadots)
    countdot = np.sum(dot)
    print('%s: %d (8-bit) %d (32-bit) %f (summed)'%(fname,count,count32,countdot))
    fname_out = string.replace(fname,'.png','.npy')
    np.save(fname_out, dot)
