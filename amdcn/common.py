import numpy as np
import cv2
from tqdm import trange

def set_gpu_memory_fraction(frac):
    """Sets fraction of GPU memory to allocate when using TensorFlow"""
    import keras.backend as K
    if K.backend() == 'tensorflow':
        import tensorflow as tf
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction=frac
        sess = tf.Session(config=config)
        K.set_session(sess)

def binary_to_image(labelim):
    """Convert binary (height,width) to image (height,width)"""
    return labelim.astype('uint8')*255

def categorical_to_image(labelim,ignore=None):
    """Convert categorical labeling (height,width,num_classes) to image (height,width)"""
    num_classes = labelim.shape[2]
    maxind = np.argmax(labelim,axis=2)
    if ignore is not None:
        maxind = maxind + 1
        maxind[ignore] = 0
        num_classes = num_classes+1
    im = maxind.astype('float32')/(num_classes-1)
    return (im*255).astype('uint8')

def ushort_to_image(im):
    """Convert unsigned 16-bit image to unsigned 8-bit image"""
    im_out = im.astype('float32')
    im_out = im_out - np.amin(im_out)
    im_out = im_out / np.amax(im_out)
    return (im_out*255).astype('uint8')

def balanced_binary_crossentropy(y_true, y_pred):
    """Binary crossentropy loss function with automatic class balancing"""
    from keras import backend as K
    y_true_f = K.cast(y_true,K.floatx())
    num_pos = K.sum(y_true_f)+1
    num_neg = K.sum(1-y_true_f)+1
    #weights = 1/(2*num_pos)*y_true + 1/(2*num_neg)*(1-y_true)
    weights = (num_neg)/(num_pos)*y_true_f + (1-y_true_f)
    score_array = K.binary_crossentropy(y_pred,y_true_f)
    score_array *= weights
    score_array /= K.mean(K.cast(K.not_equal(weights, 0), K.floatx()))
    return K.mean(score_array)

def balanced_categorical_crossentropy(y_true, y_pred):
    """Categorical crossentropy loss function with automatic class balancing"""
    y_true_f = K.cast(y_true,K.floatx())

    counts = K.sum(y_true_f,axis=0)
    print(K.shape(counts))
    counts = K.sum(counts,axis=0)
    counts = K.sum(counts,axis=0)

    max_count = K.max(counts)

    #weights = (num_neg)/(num_pos)*y_true_f + (1-y_true_f)
    score_array = K.categorical_crossentropy(y_pred,y_true_f)
    #score_array *= weights
    #score_array /= K.mean(K.cast(K.not_equal(weights, 0), K.floatx()))
    return K.mean(score_array)

def constant_pad_images(imgs,pad,val=0):
    """Pad images with constant value"""
    if len(imgs.shape)==4:
        imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1]+pad*2, imgs.shape[2]+pad*2, imgs.shape[3]), dtype=imgs.dtype)
        for i in range(imgs.shape[0]):
            padded_img = cv2.copyMakeBorder(imgs[i,:,:,:],pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=val)
            imgs_p[i,:,:,:] = np.reshape(padded_img,(1,imgs_p.shape[1],imgs_p.shape[2],imgs_p.shape[3]))
    else:
        imgs_p = cv2.copyMakeBorder(imgs,pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=val)
    return imgs_p

def mirror_pad_images(imgs,pad):
    """Pad images with mirror (reflection) padding"""
    if len(imgs.shape)==4:
        imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1]+pad*2, imgs.shape[2]+pad*2, imgs.shape[3]), dtype=imgs.dtype)
        for i in range(imgs.shape[0]):
            padded_img = cv2.copyMakeBorder(imgs[i,:,:,:],pad,pad,pad,pad,cv2.BORDER_REFLECT)
            imgs_p[i,:,:,:] = np.reshape(padded_img,(1,imgs_p.shape[1],imgs_p.shape[2],imgs_p.shape[3]))
    else:
        imgs_p = cv2.copyMakeBorder(imgs,pad,pad,pad,pad,cv2.BORDER_REFLECT)
    return imgs_p

def tile_images(imgs,tile_sz,pad):
    """Tile images to size tile_sz"""
    sz = imgs.shape[1:3]
    unpad_sz = (sz[0]-pad*2,sz[1]-pad*2)
    if unpad_sz[0] % tile_sz[0] != 0 or unpad_sz[1] % tile_sz[1] != 0:
        raise ValueError
    nx = unpad_sz[0]/tile_sz[0]
    ny = unpad_sz[1]/tile_sz[1]
    tiles = np.ndarray((imgs.shape[0]*nx*ny, tile_sz[0]+pad*2, tile_sz[1]+pad*2, imgs.shape[3]), dtype=imgs.dtype)
    print('Extracting %d tiles of size (%d,%d,%d)'%(tiles.shape[0], tiles.shape[1], tiles.shape[2], tiles.shape[3]))
    n = 0
    for i in range(imgs.shape[0]):
        for x in trange(pad,imgs.shape[1]-pad,tile_sz[0]):
            for y in range(pad,imgs.shape[2]-pad,tile_sz[1]):
                tiles[n,:,:,:] = imgs[i,(x-pad):(x+tile_sz[0]+pad),(y-pad):(y+tile_sz[1]+pad),:]
                n = n+1
    return tiles

def remove_mean(imgs,mask=None):
    """Subtract per-channel mean from each image"""
    if len(imgs.shape)==4:
        for i in range(imgs.shape[0]):
            for j in range(imgs.shape[3]):
                imgs[i,:,:,j] -= np.mean(imgs[i,:,:,j])
    else:
        if mask is not None:
            if np.count_nonzero(mask)>0:
                for j in range(imgs.shape[2]):
                    imgs[:,:,j] -= np.mean(imgs[mask,j])
        else:
            for j in range(imgs.shape[2]):
                imgs[:,:,j] -= np.mean(imgs[:,:,j])
    return imgs

def remove_local_mean(imgs,ksize):
    """Subtract per-channel mean filtered image from each image"""
    for i in range(imgs.shape[0]):
        for j in range(imgs.shape[3]):
            imgs[i,:,:,j] -= cv2.blur(imgs[i,:,:,j],(ksize,ksize))
    return imgs

