import numpy as np
from skimage import io
from skimage.feature import blob_doh
from skimage import transform
from tqdm import tqdm
from glob import glob
import argparse

parser = argparse.ArgumentParser(description='Examine all 8-bit density maps in the current folder to obtain an object count estimate')
# "Integration" in this context essentially means
# summing the density values over each pixel
# (because raster images are discretized).

parser.add_argument('--input_path', default='./',
                    help='Folder of images to test integrate over.')
parser.add_argument('--output_path', default='results.csv',
                    help='Output path for the file containing the counts.')

args = parser.parse_args()

input_path = args.input_path
if input_path[-1] != '/':
    input_path += '/'

output_path = args.output_path
lines = ['image,ground_truth,prediction\n']

print('[info] reading images')
predict_image_list = [np.load(f) for f in sorted(glob('{}*predict.npy'.format(input_path)))]
predict_fname_list = list(sorted(glob('{}*predict.npy'.format(input_path))))
label_image_list = [np.load(f) for f in sorted(glob('{}*label.npy'.format(input_path)))]
label_fname_list = list(sorted(glob('{}*label.npy'.format(input_path))))

num_images = len(predict_image_list)/4
print(num_images)
num_images = int(num_images)

mask = np.loadtxt('data/UCSD/ucsd_mask.txt')

print(len(predict_image_list))

print('[i] testing integration over patches while reconstructing full image')

counter = 0
for j in tqdm(range(num_images)):
    predict = np.zeros((158, 238, 1))
    label = np.zeros((158, 238))
    predict_sum = 0
    label_sum = 0
    predict_f = j
    for (x, y) in [(0, 0), (0, 118), (78, 0), (78, 118)]: # loop through quadrants
        predict_patch = predict_image_list[counter]
        label_patch = label_image_list[counter]
        predict[x:x+79,y:y+119,:] = predict_patch
        label[x:x+79,y:y+119] = label_patch
        counter += 1
    predict[mask == 0] = 0
    label[mask == 0] = 0
    predict_sum = np.sum(predict)
    label_sum = np.sum(label)
    lines.append('{},{},{}\n'.format(predict_f,label_sum,predict_sum))

print('[info] writing .csv output file')
with open(output_path, 'w') as f:
    for line in lines: f.write(line)
