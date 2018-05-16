import numpy as np
np.random.seed(seed=24)
import keras
from keras.models import Model
from keras.optimizers import Adam, SGD, Adagrad, Adadelta
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.losses import mean_absolute_error
from losses import density_loss, scaler_loss
from keras import metrics
from keras import backend as K
import tensorflow as tf
import cv2
import h5py
import argparse
import models
from generators import PatchGenerator, ImageGenerator
from common import set_gpu_memory_fraction

parser = argparse.ArgumentParser(description='Train a model on a dataset.')

# paths
parser.add_argument('--train_path', default='train.hdf5',
                    help='Input path for pre-processed training data and labels')
parser.add_argument('--val_path', default='val.hdf5',
                    help='Input path for pre-processed validation data and labels')
parser.add_argument('--output_path', default='model.hdf5',
                    help='Output path for model weights')
parser.add_argument('--log_path', default='logs/UCSD/upscale/',
                    help='Output path for TensorFlow logs')

# model specification
parser = models.add_arguments(parser)

# training
parser.add_argument('--gpu_frac', type=float, default=0.,
                    help='Fraction of GPU memory to allocate (TensorFlow only)')
parser.add_argument('--tile_size', type=int, default='-1',
                    help='Tile size: -1 for no tiling, 1 for patches, n>1 for nxn tiles')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size')
parser.add_argument('--batches_per_epoch', type=int, default=100,
                    help='Number of batches in an epoch')
parser.add_argument('--num_epochs', type=int, default=10,
                    help='Number of epochs')
parser.add_argument('--learning_rate', type=float, default=0.01,
                    help='Use balanced sampling of patches')


args = parser.parse_args()

# Set GPU usage
if args.gpu_frac > 0:
    set_gpu_memory_fraction(args.gpu_frac)

# Get model according to specification
m = models.get_model(args)
pad = m.get_padding()

# Open data files
print("[r] reading training data")
f_train = h5py.File(args.train_path, "r")
X_train = f_train["data"]
y_train = f_train["label"]
ignore_train = None
if 'ignore' in f_train.keys():
    ignore_train = f_train["ignore"]

print("[r] reading validation data")
f_val = h5py.File(args.val_path, "r")
X_val = f_val["data"]
y_val = f_val["label"]
ignore_val = None
if 'ignore' in f_val.keys():
    ignore_val = f_val["ignore"]

# Create model
print("[i] creating model")
if args.tile_size < 0:
    print(X_train.shape[1:4])
    model = m.get_model(X_train.shape[1:4])
else:
    print((pad*2+args.tile_size,pad*2+args.tile_size,X_train.shape[3]))
    model = m.get_model((pad*2+args.tile_size,pad*2+args.tile_size,X_train.shape[3]))

# Compile model
learning_rate = args.learning_rate
opt = Adam(lr=learning_rate)
model.compile(optimizer=opt, loss=density_loss, metrics=[scaler_loss])

if args.tile_size < 0:
    gen = ImageGenerator()
else:
    gen = PatchGenerator()

batch_size = args.batch_size

validation_steps = (args.batches_per_epoch*1)/5

train_gen = gen.flow(X_train,y_train,batch_size,pad,ignore=ignore_train)
val_gen = gen.flow(X_val,y_val,batch_size,pad,ignore=ignore_val)

log_path = args.log_path
print("[i] training model")
model_checkpoint = ModelCheckpoint(args.output_path, monitor='val_loss', save_best_only=True)
tensorboard = TensorBoard(log_dir=log_path, histogram_freq=1)
model.fit_generator(train_gen, args.batches_per_epoch, epochs=args.num_epochs, verbose=1,
          validation_data=val_gen, validation_steps=validation_steps,
          callbacks=[model_checkpoint, tensorboard])

# Close data file
f_train.close()
f_val.close()

