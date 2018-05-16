from keras import backend as K

def density_loss(y_true, y_pred):
    density_scale = K.constant(255.0)
    return K.mean(K.abs(y_pred - density_scale*y_true), axis=-1)

def scaler_loss(y_true, y_pred):
    density_scale = K.constant(255.0)
    pred_count = K.sum(K.batch_flatten(y_pred),axis=1)/density_scale
    true_count = K.sum(K.batch_flatten(y_true),axis=1)
    return K.mean(K.abs(pred_count-true_count))
