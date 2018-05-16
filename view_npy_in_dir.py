import numpy as np
import argparse
from glob import glob
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Preview images from a dataset.')

# paths
parser.add_argument('--input_path', default='output',
                    help='Input path for data and labels')
parser.add_argument('--index', type=int, default=0,
                    help='Image index to view')

args = parser.parse_args()

print("[i] reading data names")
fnames = [f for f in sorted(glob(args.input_path + '/*.npy'))]

i = args.index

while i < len(fnames):
    print('[i] viewing {}'.format(fnames[i]))
    image = np.load(fnames[i])
    print(np.shape(image))
    plt.imshow(np.reshape(image, (image.shape[0], image.shape[1])))
    plt.show()
    i += 1

