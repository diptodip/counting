import h5py
import numpy as np
from scipy import misc
from glob import glob
from tqdm import tqdm
from skimage import io

train_f = h5py.File('train.hdf5', 'w')
test_f = h5py.File('test.hdf5', 'w')
train_data = train_f.create_dataset('data', (1600, 79, 119, 1), dtype='float32')
train_label = train_f.create_dataset('label', (1600, 79, 119), dtype='float32')
test_data = test_f.create_dataset('data', (4*1200, 79, 119, 1), dtype='float32')
test_label = test_f.create_dataset('label', (4*1200, 79, 119), dtype='float32')
test_image_f = h5py.File('test_image.hdf5', 'w')
test_image_data = test_image_f.create_dataset('data', (1200, 158, 238, 1), dtype='uint8')
test_image_label = test_image_f.create_dataset('label', (1200, 158, 238), dtype='float32')

range = xrange
imlist = [io.imread(f, as_grey=True) for f in sorted(glob('./A_shanghai/*.png'))]
#labellist = [io.imread(f, as_grey=True) for f in sorted(glob('./B_shanghai/*.png'))]
labellist = [np.load(f) for f in sorted(glob('./B_shanghai/*.npy'))]
testimlist = [io.imread(f, as_grey=True) for f in sorted(glob('./A_shanghai_testing/*.png'))]
#testlabellist = [io.imread(f, as_grey=True) for f in sorted(glob('./B_shanghai_testing/*.png'))]
testlabellist = [np.load(f) for f in sorted(glob('./B_shanghai_testing/*.npy'))]

mask = np.loadtxt('ucsd_mask.txt')

print('[i] creating patched training set')

for j in tqdm(range(1600)):
    i = np.random.randint(0, 60)
    x = np.random.randint(0, 79)
    y = np.random.randint(0, 119)
    im = imlist[i]
    im[mask == 0] = 0
    im = im[x:x+79,y:y+119]
    label = labellist[i]
    label = label[x:x+79,y:y+119]
    train_data[j,:,:,:] = np.reshape(im, (1, 79, 119, 1)).astype('float32')/255
    train_label[j,:,:] = np.reshape(label, (1, 79, 119))

print('[i] creating patched testing set')

counter = 0
for j in tqdm(range(len(testimlist))):
    im = testimlist[j]
    im[mask == 0] = 0
    label = testlabellist[j]
    for (x, y) in [(0, 0), (0, 118), (78, 0), (78, 118)]: # loop through quadrants
        window = im[x:x+79,y:y+119]
        window_label = label[x:x+79,y:y+119]
        test_data[counter,:,:,:] = np.reshape(window, (1, 79, 119, 1)).astype('float32')/255
        test_label[counter,:,:] = np.reshape(window_label, (1, 79, 119))
        counter += 1

print('[i] creating full image test set')

for i in tqdm(range(240)):
    im = testimlist[i]
    im[mask == 0] = 0
    label = testlabellist[i]
    test_image_data[i,:,:,:] = np.reshape(im, (1, 158, 238, 1))
    test_image_label[i,:,:] = np.reshape(label, (1, 158, 238))
    
train_f.close()
test_f.close()
test_image_f.close()
