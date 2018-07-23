import h5py
import numpy as np
import scipy.io as sio
from random import random
from scipy import misc
from glob import glob
from tqdm import tqdm
from skimage import io

# maybe make these arguments later
stride = 50
patch_size = 100
height = 1024
width = 1024

test_f = h5py.File('full_image_test.hdf5', 'w')

print('[i] reading images')

testimlist = [io.imread(f, as_grey=False) for f in sorted(glob('./A_0_testing/*.jpg'))]
testlabellist = [np.load(f) for f in sorted(glob('./B_0_testing/*.npy'))]
testimlist += [io.imread(f, as_grey=False) for f in sorted(glob('./A_1_testing/*.jpg'))]
testlabellist += [np.load(f) for f in sorted(glob('./B_1_testing/*.npy'))]
testimlist += [io.imread(f, as_grey=False) for f in sorted(glob('./A_2_testing/*.jpg'))]
testlabellist += [np.load(f) for f in sorted(glob('./B_2_testing/*.npy'))]
testimlist += [io.imread(f, as_grey=False) for f in sorted(glob('./A_3_testing/*.jpg'))]
testlabellist += [np.load(f) for f in sorted(glob('./B_3_testing/*.npy'))]
testimlist += [io.imread(f, as_grey=False) for f in sorted(glob('./A_4_testing/*.jpg'))]
testlabellist += [np.load(f) for f in sorted(glob('./B_4_testing/*.npy'))]

print('[i] creating full patched testing set')

test_ims = []
test_labels = []

for i in tqdm(range(len(testimlist))):
    im = np.zeros((height, width))
    temp_im = testimlist[i]
    im[0:temp_im.shape[0], 0:temp_im.shape[1]] = temp_im
    label = np.zeros((height, width))
    temp_label = testlabellist[i]
    label[0:temp_label.shape[0], 0:temp_label.shape[1]] = temp_label
    test_ims.append(np.reshape(im, (1, height, width, 1)).astype('float32')/255.)
    test_labels.append(np.reshape(label, (1, height, width)))

print('[i] adding patched testing set to hdf5 file')
num_test_ims = len(test_ims)
test_data = test_f.create_dataset('data', (num_test_ims, height, width, 1), dtype='float32')
test_label = test_f.create_dataset('label', (num_test_ims, height, width), dtype='float32')
for i in tqdm(range(num_test_ims)):
    test_data[i,:,:,:] = test_ims[i]
    test_label[i,:,:] = test_labels[i]

test_f.close()
