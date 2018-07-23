import h5py
import numpy as np
import scipy.io as sio
from scipy import misc
from glob import glob
from tqdm import tqdm
from skimage import io

# maybe make these arguments later
pad = 33
patch_height = 79
patch_width = 119
height = 158
width = 238

train_f = h5py.File('train.hdf5', 'w')
test_f = h5py.File('test.hdf5', 'w')
train_data = train_f.create_dataset('data', (3200, patch_height, patch_width, 1), dtype='float32')
train_label = train_f.create_dataset('label', (3200, patch_height, patch_width), dtype='float32')


print('[i] reading images')
imlist = [io.imread(f, as_grey=True) for f in sorted(glob('./A_downscale/*.png'))]
labellist = [np.load(f) for f in sorted(glob('./B_downscale/*.npy'))]
testimlist = [io.imread(f, as_grey=True) for f in sorted(glob('./A_downscale_testing/*.png'))]
testlabellist = [np.load(f) for f in sorted(glob('./B_downscale_testing/*.npy'))]

num_test = len(testimlist)
test_data = test_f.create_dataset('data', (4 * num_test, patch_height, patch_width, 1), dtype='float32')
test_label = test_f.create_dataset('label', (4 * num_test, patch_height, patch_width), dtype='float32')

im = None
label = None
mask = np.loadtxt('ucsd_mask.txt')

print('[i] creating patched training set')

counter = 0
with tqdm(total = 1600) as pbar:
    while counter < 1600:
        sum = 0
        while sum <= 3.0: # don't train on *very* empty images
            i = np.random.randint(0, len(imlist))
            im = imlist[i]
            im[mask == 0] = 0
            x = np.random.randint(0, height - (patch_height))
            y = np.random.randint(0, width - (patch_width))
            im = im[x:x+patch_height,y:y+patch_width]
            label = labellist[i]
            label = label[x:x+patch_height,y:y+patch_width]
            sum = np.sum(label)
        train_data[counter,:,:,:] = np.reshape(im, (1, patch_height, patch_width, 1)).astype('float32')/255
        train_label[counter,:,:] = np.reshape(label, (1, patch_height, patch_width))
        counter += 1
        pbar.update(1)

print('[i] adding flips to patched training set')

for i in tqdm(range(1600)):
    train_data[i+1600,:,:,:] = np.array(np.fliplr(train_data[i,:,:,:]))
    train_label[i+1600,:,:] = np.array(np.fliplr(train_label[i,:,:]))


print('[i] creating patched testing set')

counter = 0
for i in tqdm(range(num_test)):
    im = testimlist[i]
    im[mask == 0] = 0
    label = testlabellist[i]
    for x in range(0, height, patch_height):
        for y in range(0, width, patch_width):
            im_patch = im[x:x+patch_height, y:y+patch_width]
            label_patch = label[x:x+patch_height, y:y+patch_width]
            test_data[counter,:,:,:] = np.reshape(im_patch, (1, patch_height, patch_width, 1)).astype('float32')/255.
            test_label[counter,:,:] = np.reshape(label_patch, (1, patch_height, patch_width))
            counter += 1

train_f.close()
test_f.close()
