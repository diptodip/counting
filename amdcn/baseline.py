
class BaselineModel:
    """Baseline model of convolutional layers for regression"""

    def __init__(self,num_layers,num_channels):
        self.num_layers = num_layers
        self.num_channels = num_channels

    def get_model(self,input_shape):
        import keras
        from keras.models import Model
        from keras.layers import Input, Conv2D, Activation
        from keras.layers.normalization import BatchNormalization

        inputs = Input(input_shape)
        x = inputs

        # 3x3 convolution layers
        for i in xrange(self.num_layers):
            x = Conv2D(self.num_channels, (3, 3), padding='valid', kernel_initializer='he_normal')(x)
            x = BatchNormalization(axis=3)(x)
            x = Activation('relu')(x)

        # 1x1 convolution layer
        # This is essentially a per-pixel fully connected layer,
        # which makes this a "fully convolutional network"
        x = Conv2D(1, (1, 1), kernel_initializer='he_normal')(x)

        # Final ReLU activation for regression
        x = Activation('relu')(x)

        return Model(inputs=inputs, outputs=x)

    def get_padding(self):
        return self.num_layers

