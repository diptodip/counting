import h5py
import numpy as np
import scipy.io as sio
from scipy import misc
from glob import glob
from tqdm import tqdm
from skimage import io

# maybe make these arguments later
pad = 65
patch_size = 256
height = 1024
width = 1024

train_f = h5py.File('train.hdf5', 'w')
test_f = h5py.File('test.hdf5', 'w')
test_image_f = h5py.File('test_image.hdf5', 'w')
train_data = train_f.create_dataset('data', (3200, pad*2+patch_size, pad*2+patch_size, 1), dtype='float32')
train_label = train_f.create_dataset('label', (3200, patch_size, patch_size), dtype='float32')
test_data = test_f.create_dataset('data', (16 * 10, patch_size, patch_size, 1), dtype='float32')
test_label = test_f.create_dataset('label', (16 * 10, patch_size, patch_size), dtype='float32')
test_image_data = test_image_f.create_dataset('data', (10, height, width, 1), dtype='float32')
test_image_label = test_image_f.create_dataset('label', (10, height, width), dtype='float32')

print('[i] reading images')
range = xrange
imlist = [io.imread(f, as_grey=False) for f in sorted(glob('./A_0/*.jpg'))]
labellist = [np.load(f) for f in sorted(glob('./B_0/*.npy'))]
testimlist = [io.imread(f, as_grey=False) for f in sorted(glob('./A_0_testing/*.jpg'))]
testlabellist = [np.load(f) for f in sorted(glob('./B_0_testing/*.npy'))]

im = None
label = None

print('[i] creating patched training set')

counter = 0
with tqdm(total = 1600) as pbar:
    while counter < 1600:
        sum = 0
        while sum <= 200.0: # don't train on *very* empty images
            i = np.random.randint(0, len(imlist))
            im = np.zeros((height, width))
            temp_im = imlist[i]
            im[0:temp_im.shape[0], 0:temp_im.shape[1]] = temp_im
            x = np.random.randint(0, height - (pad*2+patch_size))
            y = np.random.randint(0, width - (pad*2+patch_size))
            im = im[x:x+pad*2+patch_size,y:y+pad*2+patch_size]
            label = np.zeros((height, width))
            temp_label = labellist[i]
            label[0:temp_label.shape[0], 0:temp_label.shape[1]] = temp_label
            label = label[x+pad:x+pad+patch_size,y+pad:y+pad+patch_size]
            sum = np.sum(label)
        train_data[counter,:,:,:] = np.reshape(im, (1, pad*2+patch_size, pad*2+patch_size, 1)).astype('float32')/255.
        train_label[counter,:,:] = np.reshape(label, (1, patch_size, patch_size))
        counter += 1
        pbar.update(1)

print('[i] adding flips to patched training set')

for i in tqdm(range(1600)):
    train_data[i+1600,:,:,:] = np.array(np.flipud(train_data[i,:,:,:]))
    train_label[i+1600,:,:] = np.array(np.flipud(train_label[i,:,:]))


print('[i] creating patched testing set')

counter = 0
for i in tqdm(range(len(testimlist))):
    im = np.zeros((height, width))
    temp_im = testimlist[i]
    im[0:temp_im.shape[0], 0:temp_im.shape[1]] = temp_im
    label = np.zeros((height, width))
    temp_label = testlabellist[i]
    label[0:temp_label.shape[0], 0:temp_label.shape[1]] = temp_label
    for x in range(0, height, patch_size):
        for y in range(0, width, patch_size):
            im_patch = im[x:x+patch_size, y:y+patch_size]
            label_patch = label[x:x+patch_size, y:y+patch_size]
            test_data[counter,:,:,:] = np.reshape(im_patch, (1, patch_size, patch_size, 1)).astype('float32')/255.
            test_label[counter,:,:] = np.reshape(label_patch, (1, patch_size, patch_size))
            counter += 1

print('[i] creating whole image testing set')

for i in tqdm(range(len(testimlist))):
    im = np.zeros((height, width))
    temp_im = testimlist[i]
    im[0:temp_im.shape[0], 0:temp_im.shape[1]] = temp_im
    label = np.zeros((height, width))
    temp_label = testlabellist[i]
    label[0:temp_label.shape[0], 0:temp_label.shape[1]] = temp_label
    test_image_data[i,:,:,:] = np.reshape(im, (1, height, width, 1)).astype('float32')/255.
    test_image_label[i,:,:] = np.reshape(label, (1, height, width))

train_f.close()
test_f.close()
test_image_f.close()
