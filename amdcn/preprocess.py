from common import mirror_pad_images, constant_pad_images, remove_mean
import cv2
import h5py
import numpy as np
import argparse
import models
from tqdm import trange

parser = argparse.ArgumentParser(description='Pre-process a dataset.')

# paths
parser.add_argument('--input_path', default='data.hdf5',
                    help='Input path for data and labels')
parser.add_argument('--output_path', default='preprocessed.hdf5',
                    help='Output path for pre-processed data and labels')

# model specification
parser = models.add_arguments(parser)

# preprocessing
parser.add_argument('--tile_size', type=int, default='-1',
                    help='Tile size: -1 for no tiling, 1 for patches, n>1 for nxn tiles')
parser.add_argument('--mirror_pad', action='store_true',
                    help='Use mirror padding instead of zero padding')
parser.add_argument('--remove_mean', action='store_true',
                    help='Remove per-channel mean from each image')
parser.add_argument('--do_not_pad', action='store_true',
                    help='Do not pad input (i.e. if already padded).')


args = parser.parse_args()

# Get model as specified
m = models.get_model(args)
pad = m.get_padding()

# Open input file
print("[r] reading data")
f_in = h5py.File(args.input_path, "r")
X_in = f_in["data"]
print(X_in.dtype)
y_in = None
if 'label' in f_in.keys():
    y_in = f_in["label"]
ignore_in = None
if 'ignore' in f_in.keys():
    ignore_in = f_in['ignore']

# Calculate output data shape
output_shape = None
if args.do_not_pad:
    output_shape = X_in.shape
else:
    output_shape = (X_in.shape[0], X_in.shape[1]+pad*2, X_in.shape[2]+pad*2, X_in.shape[3])

# Open output file
print("[o] creating output file")
f_out = h5py.File(args.output_path, "w")
X_out = f_out.create_dataset("data",output_shape,dtype='float32')
if y_in is not None:
    y_out = f_out.create_dataset("label",data=y_in)
if ignore_in is not None:
    ignore_out = f_out.create_dataset("ignore",data=ignore_in)

if args.do_not_pad:
    print("[i] skip padding images")
    for i in trange(X_in.shape[0]):
        X_out[i,:,:,:] = np.reshape(np.expand_dims(X_in[i,:,:,:], axis=-1), (1, X_in.shape[1], X_in.shape[2], X_in.shape[3]))
else:
    print("[i] padding images")
    if ignore_in is not None:
        padded_ignore = np.zeros(output_shape[0:3],dtype='bool')
        for i in trange(X_in.shape[0]):
            padded_ignore[i,:,:] = np.expand_dims(constant_pad_images(ignore_in[i,:,:].astype('uint8'),pad,1).astype('bool'), axis=-1)
    if args.mirror_pad:
        for i in trange(X_in.shape[0]):
            X_out[i,:,:,:] = np.expand_dims(mirror_pad_images(X_in[i,:,:,:],pad).astype('float32'), axis=-1)
    else:
        for i in trange(X_in.shape[0]):
            X_out[i,:,:,:] = np.reshape(np.expand_dims(constant_pad_images(X_in[i,:,:,:],pad).astype('float32'), axis=-1), (1, X_in.shape[1]+2*pad, X_in.shape[2]+2*pad, X_in.shape[3]))

if args.remove_mean:
    print("[i] removing mean from images")
    for i in trange(X_in.shape[0]):
        if ignore_in is not None:
            X_out[i,:,:,:] = remove_mean(X_out[i,:,:,:],np.logical_not(padded_ignore[i,:,:]))
        else:
            X_out[i,:,:,:] = remove_mean(X_out[i,:,:,:])

f_in.close()
f_out.close()
