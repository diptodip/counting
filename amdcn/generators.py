import numpy as np
from skimage import transform
import cv2

class ImageGenerator:
    def flow(self, X, y, batch_size, pad, ignore=None):
        """Generate images in sequential order (currently limited to batch size of 1)"""

        # Sample images
        batch_num = 0
        while True:
            # Sample image
            X_batch = X[[batch_num],:,:,:].astype('float32')
            y_batch = y[[batch_num],:,:]
            y_batch = np.expand_dims(y_batch, axis=3)

            yield X_batch,y_batch

            batch_num = (batch_num+1)%X.shape[0]

def sample_patch(X, y, inds, pad, tile_size):
    image_ind = inds[0]
    y_ind1 = inds[1]
    y_ind2 = inds[2]
    
    # uncomment to see patch sampling
    # print('[i] sampling patch from image %d at (%d,%d)'%(image_ind,y_ind1,y_ind2))

    # Adjust for padding in X
    X_ind1 = y_ind1
    X_ind2 = y_ind2

    # Crop out patch
    X_patch = X[image_ind,(X_ind1):(X_ind1+(pad*2+tile_size)),(X_ind2):(X_ind2+(pad*2+tile_size)),:]
    
    # Get label
    y_patch  = y[image_ind,(y_ind1):(y_ind1+(tile_size)),(y_ind2):(y_ind2+(tile_size))]
       
    return X_patch,y_patch

class PatchGenerator:
    def flow(self, X, y, batch_size, pad, tile_size=1, ignore=None):
        """Generate batches of patches with random sampling over image"""

        # Allocate space for batch
        num_images = X.shape[0]
        X_batch = np.ndarray((batch_size,pad*2+tile_size,pad*2+tile_size,X.shape[3]),dtype='float32')
        y_batch = np.ndarray((batch_size,tile_size,tile_size,1),dtype=y.dtype)

        # Sample batches of patches
        batch_num = 0
        while True:
            # Sample patch locations
            for n in xrange(batch_size):
                # Sample patch
                image_ind = np.random.randint(0,num_images)
                y_ind1 = np.random.randint(0,y.shape[1] - (tile_size))
                y_ind2 = np.random.randint(0,y.shape[2] - (tile_size))
                inds = (image_ind, y_ind1, y_ind2)
                sum = 0
                shape = (0, 0, 0)
                X_patch = None
                y_patch = None
                while (sum <= 1.0 and shape != (pad*2+tile_size, pad*2+tile_size, X.shape[3])):
                    X_patch, y_patch = sample_patch(X,y,(inds[0],inds[1],inds[2]),pad,tile_size)
                    y_patch = transform.resize(y_patch, (tile_size, tile_size), order=1)
                    sum = np.sum(y_patch)
                    shape = X_patch.shape
                X_batch[n,:,:,:] = X_patch.astype('float32')
                y_batch[n,:,:,:] = np.expand_dims(y_patch, axis=-1)
                
                # uncomment to save patches
                # cv2.imwrite('output/patch%05d_%05d.png'%(batch_num,n),X_patch)
            yield X_batch,y_batch
            batch_num = batch_num+1
