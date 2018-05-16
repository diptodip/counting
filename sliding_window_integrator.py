import h5py
import numpy as np
import scipy.io as sio
from scipy import misc
from glob import glob
from tqdm import tqdm
from skimage import io
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Examine all 8-bit density maps in the current folder to obtain an object count estimate')
# "Integration" in this context essentially means
# summing the density values over each pixel
# (because raster images are discretized).

parser.add_argument('--input_path', default='./',
                    help='Folder of images to test integrate over.')
parser.add_argument('--num_images', type=int, default=599,
                    help='Number of test images.')
parser.add_argument('--height', type=int, default=576,
                    help='Height of test images.')
parser.add_argument('--width', type=int, default=720,
                    help='Width of test images.')
parser.add_argument('--patch_size', type=int, default=150,
                    help='Patch size used in testing.')
parser.add_argument('--stride', type=int, default=100,
                    help='Stride used to densely scan the image.')
parser.add_argument('--output_path', default='results.csv',
                    help='Output path for the file containing the counts.')

args = parser.parse_args()

input_path = args.input_path
if input_path[-1] != '/':
    input_path += '/'

output_path = args.output_path
num_images = args.num_images
height = args.height
width = args.width
patch_size = args.patch_size
stride = args.stride
lines = ['image,ground_truth,prediction\n']

print('[r] reading image names')
predict_fname_list = list(sorted(glob('{}*predict.npy'.format(input_path))))
label_fname_list = list(sorted(glob('{}*label.npy'.format(input_path))))

print('[i] testing integration over all images')
counter = 0
for i in tqdm(range(num_images)):
    predict = np.zeros((height, width), dtype=np.float32)
    label = np.zeros((height, width), dtype=np.float32)
    overlap_counter = np.zeros((height, width), dtype=np.float32)
    x = 0
    while x <= (height - patch_size):
        y = 0
        while y <= (width - patch_size):
            predict_patch = np.reshape(np.load(predict_fname_list[counter]), (patch_size, patch_size))
            label_patch = np.reshape(np.load(label_fname_list[counter]), (patch_size, patch_size))
            label[x:x+patch_size, y:y+patch_size] = label_patch
            predict[x:x+patch_size, y:y+patch_size] += predict_patch
            overlap_counter[x:x+patch_size, y:y+patch_size] += np.ones((patch_size, patch_size), dtype=np.float32)
            counter += 1
            if y != (width - patch_size):
                y += min(stride, width - (y + patch_size))
            else:
                y = width
        if x != (height - patch_size):
            x += min(stride, height - (x + patch_size))
        else:
            x = height
    predict = predict/overlap_counter # element wise division
    label_sum = np.sum(label)
    predict_sum = np.sum(predict)
    lines.append('{},{},{}\n'.format(i, label_sum, predict_sum))

print('[o] writing .csv output file')
with open(output_path, 'w') as f:
    for line in lines: f.write(line)
