from common import categorical_to_image, binary_to_image, ushort_to_image
import cv2
import h5py
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Produce images from a dataset.')

# paths
parser.add_argument('--input_path', default='data.hdf5',
                    help='Input path for data and labels')
parser.add_argument('--index', type=int, default=0,
                    help='Image index to view')

args = parser.parse_args()

print("[r] reading data")
f_in = h5py.File(args.input_path, "r")
X_in = f_in["data"]
y_in = None
if 'label' in f_in.keys():
    y_in = f_in["label"]
ignore_in = None
if 'ignore' in f_in.keys():
    ignore_in = f_in['ignore']

for i in range(X_in.shape[3]):
    print((np.amin(X_in[args.index,:,:,i]),np.amax(X_in[args.index,:,:,i])))
    cv2.imwrite('output/%05d_image%d.png'%(args.index,i),ushort_to_image(X_in[args.index,:,:,i]))
if y_in is not None:
    print((np.amin(y_in[args.index,:,:]),np.amax(y_in[args.index,:,:])))
    cv2.imwrite('output/%05d_label.png'%(args.index),ushort_to_image(y_in[args.index,:,:]))
if ignore_in is not None:
    cv2.imwrite('output/%05d_ignore.png'%(args.index),binary_to_image(ignore_in[args.index,:,:]))

f_in.close()

