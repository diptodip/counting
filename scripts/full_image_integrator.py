import numpy as np
from skimage import io
from skimage.feature import blob_doh
from skimage import transform
from tqdm import tqdm
from glob import glob
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Examine all 8-bit density maps in the current folder to obtain an object count estimate')
# "Integration" in this context essentially means
# summing the density values over each pixel
# (because raster images are discretized).

parser.add_argument('--input_path', default='./',
                    help='Folder of images to test integrate over.')
parser.add_argument('--num_images', type=int, default=50,
                    help='Number of test images.')
parser.add_argument('--height', type=int, default=1024,
                    help='Height of test images.')
parser.add_argument('--width', type=int, default=1024,
                    help='Width of test images.')
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
lines = ['image,ground_truth,prediction\n']

print('[info] reading images')
original_image_list = [io.imread(f, as_grey=True) for f in sorted(glob('./data/UCF/A_0_testing/*.jpg'))]
original_image_list += [io.imread(f, as_grey=True) for f in sorted(glob('./data/UCF/A_1_testing/*.jpg'))]
original_image_list += [io.imread(f, as_grey=True) for f in sorted(glob('./data/UCF/A_2_testing/*.jpg'))]
original_image_list += [io.imread(f, as_grey=True) for f in sorted(glob('./data/UCF/A_3_testing/*.jpg'))]
original_image_list += [io.imread(f, as_grey=True) for f in sorted(glob('./data/UCF/A_4_testing/*.jpg'))]

predict_image_list = [np.load(f) for f in sorted(glob('{}*predict.npy'.format(input_path)))]
predict_fname_list = list(sorted(glob('{}*predict.npy'.format(input_path))))
label_image_list = [np.load(f) for f in sorted(glob('{}*label.npy'.format(input_path)))]
label_fname_list = list(sorted(glob('{}*label.npy'.format(input_path))))

print('[info] testing integration over all images')
counter = 0
for i in tqdm(range(num_images)):
    predict_sum = 0
    label_sum = 0
    predict_f = predict_fname_list[i]
    original = original_image_list[i]
    im_unclipped = predict_image_list[i].reshape((1024, 1024))
    im = np.zeros((1024, 1024))
    im[0:original.shape[0],0:original.shape[1]] = im_unclipped[0:original.shape[0],0:original.shape[1]]
    label = label_image_list[i]
    predict_sum = np.sum(im)
    label_sum = np.sum(label)
    lines.append('{},{},{}\n'.format(predict_f, label_sum, predict_sum))

print('[info] writing .csv output file')
with open(output_path, 'w') as f:
    for line in lines: f.write(line)
