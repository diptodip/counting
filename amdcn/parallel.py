class ParallelDilationModel:
    """
    Multicolumn regression model that uses dilations like Koltun et al.
    """

    def __init__(self, num_channels):
        self.num_channels = num_channels

    def get_model(self, input_shape):
        import keras
        from keras.models import Model
        from keras.layers import Input, Conv2D, Activation
        
        inputs = Input(input_shape)
        x = inputs
        print(x._keras_shape)

        # col 1

        # 3x3 convolution layer
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', )(x)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)
        
        # 3x3 convolution layer
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', )(col1)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)
        
        # 3x3 convolution layer with dilation 2
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(col1)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)
        
        # 3x3 convolution layer with dilation 4
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(col1)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)

        # col 2

        # 3x3 convolution layer with dilation 2
        col2 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(x)
        col2 = Activation('relu')(col2)
        print(col2._keras_shape)
        
        # 3x3 convolution layer with dilation 2
        col2 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(col2)
        col2 = Activation('relu')(col2)
        print(col2._keras_shape)
        
        # 3x3 convolution layer with dilation 4
        col2 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(col2)
        col2 = Activation('relu')(col2)
        print(col2._keras_shape)
        
        # 3x3 convolution layer with dilation 8
        col2 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(col2)
        col2 = Activation('relu')(col2)
        print(col2._keras_shape)
        
        # col 3

        # 3x3 convolution layer with dilation 4
        col3 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(x)
        col3 = Activation('relu')(col3)
        print(col3._keras_shape)
        
        # 3x3 convolution layer with dilation 4
        col3 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(col3)
        col3 = Activation('relu')(col3)
        print(col3._keras_shape)
        
        # 3x3 convolution layer with dilation 8
        col3 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(8,8))(col3)
        col3 = Activation('relu')(col3)
        print(col3._keras_shape)
        
        # 3x3 convolution layer with dilation 16
        col3 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(16,16))(col2)
        col3 = Activation('relu')(col3)
        print(col3._keras_shape)

        # col 4

        # 5x5 convolution layer with dilation 2
        col4 = Conv2D(self.num_channels, (5, 5), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(x)
        col4 = Activation('relu')(col4)
        print(col4._keras_shape)
        
        # 5x5 convolution layer with dilation 2
        col4 = Conv2D(self.num_channels, (5, 5), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(col4)
        col4 = Activation('relu')(col4)
        print(col4._keras_shape)
        
        # 5x5 convolution layer with dilation 4
        col4 = Conv2D(self.num_channels, (5, 5), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(col4)
        col4 = Activation('relu')(col4)
        print(col4._keras_shape)
        
        # 5x5 convolution layer with dilation 8
        col4 = Conv2D(self.num_channels, (5, 5), padding='same', kernel_initializer='he_normal', dilation_rate=(8,8))(col4)
        col4 = Activation('relu')(col4)
        print(col4._keras_shape)
        
        # col 5

        # 5x5 convolution layer with dilation 4
        col5 = Conv2D(self.num_channels, (5, 5), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(x)
        col5 = Activation('relu')(col5)
        print(col5._keras_shape)
        
        # 5x5 convolution layer with dilation 4
        col5 = Conv2D(self.num_channels, (5, 5), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(col5)
        col5 = Activation('relu')(col5)
        print(col5._keras_shape)
        
        # 5x5 convolution layer with dilation 8
        col5 = Conv2D(self.num_channels, (5, 5), padding='same', kernel_initializer='he_normal', dilation_rate=(8,8))(col5)
        col5 = Activation('relu')(col5)
        print(col5._keras_shape)
        
        # 5x5 convolution layer with dilation 16
        col5 = Conv2D(self.num_channels, (5, 5), padding='same', kernel_initializer='he_normal', dilation_rate=(16,16))(col5)
        col5 = Activation('relu')(col5)
        print(col5._keras_shape)

        # merge columns

        out = keras.layers.concatenate([col1, col2, col3, col4, col5])

        # 1x1 convolution layer
        # This is essentially a per-pixel fully connected layer,
        # which makes this a "fully convolutional network"
        out = Conv2D(1, (1, 1), kernel_initializer='he_normal')(out)

        # Final ReLU activation (for regression)
        out = Activation('relu')(out)
        print(out._keras_shape)
        
        return Model(inputs=inputs, outputs=out)

    def get_padding(self):
        return 0

class ParallelAggregationModel:
    """
    Multicolumn regression model that uses dilations and an aggregation module.
    """

    def __init__(self, num_channels):
        self.num_channels = num_channels

    def get_model(self, input_shape):
        import keras
        from keras.models import Model
        from keras.layers import Input, Conv2D, Activation
        
        inputs = Input(input_shape)
        x = inputs
        print(x._keras_shape)

        # col 1

        # 3x3 convolution layer
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', )(x)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)
        
        # 3x3 convolution layer
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', )(col1)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)
        
        # 3x3 convolution layer with dilation 2
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(col1)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)
        
        # 3x3 convolution layer with dilation 4
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(col1)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)

        # col 2

        # 3x3 convolution layer with dilation 2
        col2 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(x)
        col2 = Activation('relu')(col2)
        print(col2._keras_shape)
        
        # 3x3 convolution layer with dilation 2
        col2 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(col2)
        col2 = Activation('relu')(col2)
        print(col2._keras_shape)
        
        # 3x3 convolution layer with dilation 4
        col2 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(col2)
        col2 = Activation('relu')(col2)
        print(col2._keras_shape)
        
        # 3x3 convolution layer with dilation 8
        col2 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(8,8))(col2)
        col2 = Activation('relu')(col2)
        print(col2._keras_shape)
        
        # col 3

        # 3x3 convolution layer with dilation 4
        col3 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(x)
        col3 = Activation('relu')(col3)
        print(col3._keras_shape)
        
        # 3x3 convolution layer with dilation 4
        col3 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(col3)
        col3 = Activation('relu')(col3)
        print(col3._keras_shape)
        
        # 3x3 convolution layer with dilation 8
        col3 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(8,8))(col3)
        col3 = Activation('relu')(col3)
        print(col3._keras_shape)
        
        # 3x3 convolution layer with dilation 16
        col3 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(16,16))(col2)
        col3 = Activation('relu')(col3)
        print(col3._keras_shape)

        # col 4

        # 5x5 convolution layer with dilation 2
        col4 = Conv2D(self.num_channels, (5, 5), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(x)
        col4 = Activation('relu')(col4)
        print(col4._keras_shape)
        
        # 5x5 convolution layer with dilation 2
        col4 = Conv2D(self.num_channels, (5, 5), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(col4)
        col4 = Activation('relu')(col4)
        print(col4._keras_shape)
        
        # 5x5 convolution layer with dilation 4
        col4 = Conv2D(self.num_channels, (5, 5), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(col4)
        col4 = Activation('relu')(col4)
        print(col4._keras_shape)
        
        # 5x5 convolution layer with dilation 8
        col4 = Conv2D(self.num_channels, (5, 5), padding='same', kernel_initializer='he_normal', dilation_rate=(8,8))(col4)
        col4 = Activation('relu')(col4)
        print(col4._keras_shape)
        
        # col 5

        # 5x5 convolution layer with dilation 4
        col5 = Conv2D(self.num_channels, (5, 5), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(x)
        col5 = Activation('relu')(col5)
        print(col5._keras_shape)
        
        # 5x5 convolution layer with dilation 4
        col5 = Conv2D(self.num_channels, (5, 5), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(col5)
        col5 = Activation('relu')(col5)
        print(col5._keras_shape)
        
        # 5x5 convolution layer with dilation 8
        col5 = Conv2D(self.num_channels, (5, 5), padding='same', kernel_initializer='he_normal', dilation_rate=(8,8))(col5)
        col5 = Activation('relu')(col5)
        print(col5._keras_shape)
        
        # 5x5 convolution layer with dilation 16
        col5 = Conv2D(self.num_channels, (5, 5), padding='same', kernel_initializer='he_normal', dilation_rate=(16,16))(col5)
        col5 = Activation('relu')(col5)
        print(col5._keras_shape)

        # merge columns

        out = keras.layers.concatenate([col1, col2, col3, col4, col5])

        # 3x3 convolution layer with dilation 2
        out = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(out)
        out = Activation('relu')(out)
        print(out._keras_shape)

        # 3x3 convolution layer with dilation 4
        out = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(out)
        out = Activation('relu')(out)
        print(out._keras_shape)
        
        # 3x3 convolution layer with dilation 8
        out = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(8,8))(out)
        out = Activation('relu')(out)
        print(out._keras_shape)
        
        # 3x3 convolution layer with dilation 16
        out = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(16,16))(out)
        out = Activation('relu')(out)
        print(out._keras_shape)

        # 1x1 convolution layer
        # This is essentially a per-pixel fully connected layer,
        # which makes this a "fully convolutional network"
        out = Conv2D(1, (1, 1), kernel_initializer='he_normal')(out)

        # Final ReLU activation (for regression)
        out = Activation('relu')(out)
        print(out._keras_shape)
        
        return Model(inputs=inputs, outputs=out)

    def get_padding(self):
        return 0

class ParallelAggregationModelTest:

    def __init__(self, num_channels):
        self.num_channels = num_channels

    def get_model(self, input_shape):
        import keras
        from keras.models import Model
        from keras.layers import Input, Conv2D, Activation
        
        inputs = Input(input_shape)
        x = inputs
        print(x._keras_shape)

        # col 1

        # 3x3 convolution layer
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', )(x)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)
        
        # 3x3 convolution layer
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', )(col1)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)
        
        # 3x3 convolution layer with dilation 2
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(col1)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)
        
        # 3x3 convolution layer with dilation 4
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(col1)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)

        # col 2

        # 3x3 convolution layer with dilation 2
        col2 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(x)
        col2 = Activation('relu')(col2)
        print(col2._keras_shape)
        
        # 3x3 convolution layer with dilation 2
        col2 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(col2)
        col2 = Activation('relu')(col2)
        print(col2._keras_shape)
        
        # 3x3 convolution layer with dilation 4
        col2 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(col2)
        col2 = Activation('relu')(col2)
        print(col2._keras_shape)
        
        # 3x3 convolution layer with dilation 8
        col2 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(8,8))(col2)
        col2 = Activation('relu')(col2)
        print(col2._keras_shape)
        
        # col 3

        # 3x3 convolution layer with dilation 4
        col3 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(x)
        col3 = Activation('relu')(col3)
        print(col3._keras_shape)
        
        # 3x3 convolution layer with dilation 4
        col3 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(col3)
        col3 = Activation('relu')(col3)
        print(col3._keras_shape)
        
        # 3x3 convolution layer with dilation 8
        col3 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(8,8))(col3)
        col3 = Activation('relu')(col3)
        print(col3._keras_shape)
        
        # 3x3 convolution layer with dilation 16
        col3 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(16,16))(col2)
        col3 = Activation('relu')(col3)
        print(col3._keras_shape)

        # col 4

        # 5x5 convolution layer with dilation 2
        col4 = Conv2D(self.num_channels, (5, 5), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(x)
        col4 = Activation('relu')(col4)
        print(col4._keras_shape)
        
        # 5x5 convolution layer with dilation 2
        col4 = Conv2D(self.num_channels, (5, 5), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(col4)
        col4 = Activation('relu')(col4)
        print(col4._keras_shape)
        
        # 5x5 convolution layer with dilation 4
        col4 = Conv2D(self.num_channels, (5, 5), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(col4)
        col4 = Activation('relu')(col4)
        print(col4._keras_shape)
        
        # 5x5 convolution layer with dilation 8
        col4 = Conv2D(self.num_channels, (5, 5), padding='same', kernel_initializer='he_normal', dilation_rate=(8,8))(col4)
        col4 = Activation('relu')(col4)
        print(col4._keras_shape)
        
        # col 5

        # 5x5 convolution layer with dilation 4
        col5 = Conv2D(self.num_channels, (5, 5), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(x)
        col5 = Activation('relu')(col5)
        print(col5._keras_shape)
        
        # 5x5 convolution layer with dilation 4
        col5 = Conv2D(self.num_channels, (5, 5), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(col5)
        col5 = Activation('relu')(col5)
        print(col5._keras_shape)
        
        # 5x5 convolution layer with dilation 8
        col5 = Conv2D(self.num_channels, (5, 5), padding='same', kernel_initializer='he_normal', dilation_rate=(8,8))(col5)
        col5 = Activation('relu')(col5)
        print(col5._keras_shape)
        
        # 5x5 convolution layer with dilation 16
        col5 = Conv2D(self.num_channels, (5, 5), padding='same', kernel_initializer='he_normal', dilation_rate=(16,16))(col5)
        col5 = Activation('relu')(col5)
        print(col5._keras_shape)

        # merge columns

        out = keras.layers.concatenate([col1, col2, col3, col4, col5])

        # 3x3 convolution layer with dilation 2
        out = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(out)
        out = Activation('relu')(out)
        print(out._keras_shape)

        # 3x3 convolution layer with dilation 4
        out = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(out)
        out = Activation('relu')(out)
        print(out._keras_shape)
        
        # 3x3 convolution layer with dilation 8
        out = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(8,8))(out)
        out = Activation('relu')(out)
        print(out._keras_shape)
        
        # 3x3 convolution layer with dilation 16
        out = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(16,16))(out)
        out = Activation('relu')(out)
        print(out._keras_shape)

        # 1x1 convolution layer
        # This is essentially a per-pixel fully connected layer,
        # which makes this a "fully convolutional network"
        out = Conv2D(1, (1, 1), kernel_initializer='he_normal')(out)

        # Final ReLU activation (for regression)
        out = Activation('relu')(out)
        print(out._keras_shape)
        
        return Model(inputs=inputs, outputs=out)

    def get_padding(self):
        return 0

class ParallelAggregationModel4ColTest:

    def __init__(self, num_channels):
        self.num_channels = num_channels

    def get_model(self, input_shape):
        import keras
        from keras.models import Model
        from keras.layers import Input, Conv2D, Activation
        
        inputs = Input(input_shape)
        x = inputs
        print(x._keras_shape)

        # col 1

        # 3x3 convolution layer
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', )(x)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)
        
        # 3x3 convolution layer
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', )(col1)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)
        
        # 3x3 convolution layer with dilation 2
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(col1)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)
        
        # 3x3 convolution layer with dilation 4
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(col1)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)

        # col 2

        # 3x3 convolution layer with dilation 2
        col2 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(x)
        col2 = Activation('relu')(col2)
        print(col2._keras_shape)
        
        # 3x3 convolution layer with dilation 2
        col2 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(col2)
        col2 = Activation('relu')(col2)
        print(col2._keras_shape)
        
        # 3x3 convolution layer with dilation 4
        col2 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(col2)
        col2 = Activation('relu')(col2)
        print(col2._keras_shape)
        
        # 3x3 convolution layer with dilation 8
        col2 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(8,8))(col2)
        col2 = Activation('relu')(col2)
        print(col2._keras_shape)
        
        # col 3

        # 3x3 convolution layer with dilation 4
        col3 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(x)
        col3 = Activation('relu')(col3)
        print(col3._keras_shape)
        
        # 3x3 convolution layer with dilation 4
        col3 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(col3)
        col3 = Activation('relu')(col3)
        print(col3._keras_shape)
        
        # 3x3 convolution layer with dilation 8
        col3 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(8,8))(col3)
        col3 = Activation('relu')(col3)
        print(col3._keras_shape)
        
        # 3x3 convolution layer with dilation 16
        col3 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(16,16))(col2)
        col3 = Activation('relu')(col3)
        print(col3._keras_shape)

        # col 4

        # 5x5 convolution layer with dilation 2
        col4 = Conv2D(self.num_channels, (5, 5), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(x)
        col4 = Activation('relu')(col4)
        print(col4._keras_shape)
        
        # 5x5 convolution layer with dilation 2
        col4 = Conv2D(self.num_channels, (5, 5), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(col4)
        col4 = Activation('relu')(col4)
        print(col4._keras_shape)
        
        # 5x5 convolution layer with dilation 4
        col4 = Conv2D(self.num_channels, (5, 5), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(col4)
        col4 = Activation('relu')(col4)
        print(col4._keras_shape)
        
        # 5x5 convolution layer with dilation 8
        col4 = Conv2D(self.num_channels, (5, 5), padding='same', kernel_initializer='he_normal', dilation_rate=(8,8))(col4)
        col4 = Activation('relu')(col4)
        print(col4._keras_shape)
        
        # merge columns

        out = keras.layers.concatenate([col1, col2, col3, col4])

        # 3x3 convolution layer with dilation 2
        out = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(out)
        out = Activation('relu')(out)
        print(out._keras_shape)

        # 3x3 convolution layer with dilation 4
        out = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(out)
        out = Activation('relu')(out)
        print(out._keras_shape)
        
        # 3x3 convolution layer with dilation 8
        out = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(8,8))(out)
        out = Activation('relu')(out)
        print(out._keras_shape)
        
        # 3x3 convolution layer with dilation 16
        out = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(16,16))(out)
        out = Activation('relu')(out)
        print(out._keras_shape)

        # 1x1 convolution layer
        # This is essentially a per-pixel fully connected layer,
        # which makes this a "fully convolutional network"
        out = Conv2D(1, (1, 1), kernel_initializer='he_normal')(out)

        # Final ReLU activation (for regression)
        out = Activation('relu')(out)
        print(out._keras_shape)
        
        return Model(inputs=inputs, outputs=out)

    def get_padding(self):
        return 0

class ParallelAggregationModel3ColTest:

    def __init__(self, num_channels):
        self.num_channels = num_channels

    def get_model(self, input_shape):
        import keras
        from keras.models import Model
        from keras.layers import Input, Conv2D, Activation
        
        inputs = Input(input_shape)
        x = inputs
        print(x._keras_shape)

        # col 1

        # 3x3 convolution layer
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', )(x)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)
        
        # 3x3 convolution layer
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', )(col1)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)
        
        # 3x3 convolution layer with dilation 2
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(col1)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)
        
        # 3x3 convolution layer with dilation 4
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(col1)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)

        # col 2

        # 3x3 convolution layer with dilation 2
        col2 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(x)
        col2 = Activation('relu')(col2)
        print(col2._keras_shape)
        
        # 3x3 convolution layer with dilation 2
        col2 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(col2)
        col2 = Activation('relu')(col2)
        print(col2._keras_shape)
        
        # 3x3 convolution layer with dilation 4
        col2 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(col2)
        col2 = Activation('relu')(col2)
        print(col2._keras_shape)
        
        # 3x3 convolution layer with dilation 8
        col2 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(8,8))(col2)
        col2 = Activation('relu')(col2)
        print(col2._keras_shape)
        
        # col 3

        # 3x3 convolution layer with dilation 4
        col3 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(x)
        col3 = Activation('relu')(col3)
        print(col3._keras_shape)
        
        # 3x3 convolution layer with dilation 4
        col3 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(col3)
        col3 = Activation('relu')(col3)
        print(col3._keras_shape)
        
        # 3x3 convolution layer with dilation 8
        col3 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(8,8))(col3)
        col3 = Activation('relu')(col3)
        print(col3._keras_shape)
        
        # 3x3 convolution layer with dilation 16
        col3 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(16,16))(col2)
        col3 = Activation('relu')(col3)
        print(col3._keras_shape)
        
        # merge columns

        out = keras.layers.concatenate([col1, col2, col3])

        # 3x3 convolution layer with dilation 2
        out = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(out)
        out = Activation('relu')(out)
        print(out._keras_shape)

        # 3x3 convolution layer with dilation 4
        out = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(out)
        out = Activation('relu')(out)
        print(out._keras_shape)
        
        # 3x3 convolution layer with dilation 8
        out = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(8,8))(out)
        out = Activation('relu')(out)
        print(out._keras_shape)
        
        # 3x3 convolution layer with dilation 16
        out = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(16,16))(out)
        out = Activation('relu')(out)
        print(out._keras_shape)

        # 1x1 convolution layer
        # This is essentially a per-pixel fully connected layer,
        # which makes this a "fully convolutional network"
        out = Conv2D(1, (1, 1), kernel_initializer='he_normal')(out)

        # Final ReLU activation (for regression)
        out = Activation('relu')(out)
        print(out._keras_shape)
        
        return Model(inputs=inputs, outputs=out)

    def get_padding(self):
        return 0

class ParallelAggregationModel2ColTest:

    def __init__(self, num_channels):
        self.num_channels = num_channels

    def get_model(self, input_shape):
        import keras
        from keras.models import Model
        from keras.layers import Input, Conv2D, Activation
        
        inputs = Input(input_shape)
        x = inputs
        print(x._keras_shape)

        # col 1

        # 3x3 convolution layer
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', )(x)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)
        
        # 3x3 convolution layer
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', )(col1)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)
        
        # 3x3 convolution layer with dilation 2
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(col1)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)
        
        # 3x3 convolution layer with dilation 4
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(col1)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)

        # col 2

        # 3x3 convolution layer with dilation 2
        col2 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(x)
        col2 = Activation('relu')(col2)
        print(col2._keras_shape)
        
        # 3x3 convolution layer with dilation 2
        col2 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(col2)
        col2 = Activation('relu')(col2)
        print(col2._keras_shape)
        
        # 3x3 convolution layer with dilation 4
        col2 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(col2)
        col2 = Activation('relu')(col2)
        print(col2._keras_shape)
        
        # 3x3 convolution layer with dilation 8
        col2 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(8,8))(col2)
        col2 = Activation('relu')(col2)
        print(col2._keras_shape)
        
        # merge columns

        out = keras.layers.concatenate([col1, col2])

        # 3x3 convolution layer with dilation 2
        out = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(out)
        out = Activation('relu')(out)
        print(out._keras_shape)

        # 3x3 convolution layer with dilation 4
        out = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(out)
        out = Activation('relu')(out)
        print(out._keras_shape)
        
        # 3x3 convolution layer with dilation 8
        out = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(8,8))(out)
        out = Activation('relu')(out)
        print(out._keras_shape)
        
        # 3x3 convolution layer with dilation 16
        out = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(16,16))(out)
        out = Activation('relu')(out)
        print(out._keras_shape)

        # 1x1 convolution layer
        # This is essentially a per-pixel fully connected layer,
        # which makes this a "fully convolutional network"
        out = Conv2D(1, (1, 1), kernel_initializer='he_normal')(out)

        # Final ReLU activation (for regression)
        out = Activation('relu')(out)
        print(out._keras_shape)
        
        return Model(inputs=inputs, outputs=out)

    def get_padding(self):
        return 0

class ParallelAggregationModel1ColTest:

    def __init__(self, num_channels):
        self.num_channels = num_channels

    def get_model(self, input_shape):
        import keras
        from keras.models import Model
        from keras.layers import Input, Conv2D, Activation
        
        inputs = Input(input_shape)
        x = inputs
        print(x._keras_shape)

        # col 1

        # 3x3 convolution layer
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', )(x)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)
        
        # 3x3 convolution layer
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', )(col1)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)
        
        # 3x3 convolution layer with dilation 2
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(col1)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)
        
        # 3x3 convolution layer with dilation 4
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(col1)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)

        # merge columns

        # out = keras.layers.concatenate([col1])
        out = col1

        # 3x3 convolution layer with dilation 2
        out = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(out)
        out = Activation('relu')(out)
        print(out._keras_shape)

        # 3x3 convolution layer with dilation 4
        out = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(out)
        out = Activation('relu')(out)
        print(out._keras_shape)
        
        # 3x3 convolution layer with dilation 8
        out = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(8,8))(out)
        out = Activation('relu')(out)
        print(out._keras_shape)
        
        # 3x3 convolution layer with dilation 16
        out = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(16,16))(out)
        out = Activation('relu')(out)
        print(out._keras_shape)

        # 1x1 convolution layer
        # This is essentially a per-pixel fully connected layer,
        # which makes this a "fully convolutional network"
        out = Conv2D(1, (1, 1), kernel_initializer='he_normal')(out)

        # Final ReLU activation (for regression)
        out = Activation('relu')(out)
        print(out._keras_shape)
        
        return Model(inputs=inputs, outputs=out)

    def get_padding(self):
        return 0

class ParallelNoAggregationModelTest:

    def __init__(self, num_channels):
        self.num_channels = num_channels

    def get_model(self, input_shape):
        import keras
        from keras.models import Model
        from keras.layers import Input, Conv2D, Activation
        
        inputs = Input(input_shape)
        x = inputs
        print(x._keras_shape)

        # col 1

        # 3x3 convolution layer
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', )(x)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)
        
        # 3x3 convolution layer
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', )(col1)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)
        
        # 3x3 convolution layer with dilation 2
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(col1)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)
        
        # 3x3 convolution layer with dilation 4
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(col1)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)

        # col 2

        # 3x3 convolution layer with dilation 2
        col2 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(x)
        col2 = Activation('relu')(col2)
        print(col2._keras_shape)
        
        # 3x3 convolution layer with dilation 2
        col2 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(col2)
        col2 = Activation('relu')(col2)
        print(col2._keras_shape)
        
        # 3x3 convolution layer with dilation 4
        col2 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(col2)
        col2 = Activation('relu')(col2)
        print(col2._keras_shape)
        
        # 3x3 convolution layer with dilation 8
        col2 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(8,8))(col2)
        col2 = Activation('relu')(col2)
        print(col2._keras_shape)
        
        # col 3

        # 3x3 convolution layer with dilation 4
        col3 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(x)
        col3 = Activation('relu')(col3)
        print(col3._keras_shape)
        
        # 3x3 convolution layer with dilation 4
        col3 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(col3)
        col3 = Activation('relu')(col3)
        print(col3._keras_shape)
        
        # 3x3 convolution layer with dilation 8
        col3 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(8,8))(col3)
        col3 = Activation('relu')(col3)
        print(col3._keras_shape)
        
        # 3x3 convolution layer with dilation 16
        col3 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(16,16))(col2)
        col3 = Activation('relu')(col3)
        print(col3._keras_shape)

        # col 4

        # 5x5 convolution layer with dilation 2
        col4 = Conv2D(self.num_channels, (5, 5), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(x)
        col4 = Activation('relu')(col4)
        print(col4._keras_shape)
        
        # 5x5 convolution layer with dilation 2
        col4 = Conv2D(self.num_channels, (5, 5), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(col4)
        col4 = Activation('relu')(col4)
        print(col4._keras_shape)
        
        # 5x5 convolution layer with dilation 4
        col4 = Conv2D(self.num_channels, (5, 5), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(col4)
        col4 = Activation('relu')(col4)
        print(col4._keras_shape)
        
        # 5x5 convolution layer with dilation 8
        col4 = Conv2D(self.num_channels, (5, 5), padding='same', kernel_initializer='he_normal', dilation_rate=(8,8))(col4)
        col4 = Activation('relu')(col4)
        print(col4._keras_shape)
        
        # col 5

        # 5x5 convolution layer with dilation 4
        col5 = Conv2D(self.num_channels, (5, 5), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(x)
        col5 = Activation('relu')(col5)
        print(col5._keras_shape)
        
        # 5x5 convolution layer with dilation 4
        col5 = Conv2D(self.num_channels, (5, 5), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(col5)
        col5 = Activation('relu')(col5)
        print(col5._keras_shape)
        
        # 5x5 convolution layer with dilation 8
        col5 = Conv2D(self.num_channels, (5, 5), padding='same', kernel_initializer='he_normal', dilation_rate=(8,8))(col5)
        col5 = Activation('relu')(col5)
        print(col5._keras_shape)
        
        # 5x5 convolution layer with dilation 16
        col5 = Conv2D(self.num_channels, (5, 5), padding='same', kernel_initializer='he_normal', dilation_rate=(16,16))(col5)
        col5 = Activation('relu')(col5)
        print(col5._keras_shape)

        # merge columns

        out = keras.layers.concatenate([col1, col2, col3, col4, col5])

        # 1x1 convolution layer
        # This is essentially a per-pixel fully connected layer,
        # which makes this a "fully convolutional network"
        out = Conv2D(1, (1, 1), kernel_initializer='he_normal')(out)

        # Final ReLU activation (for regression)
        out = Activation('relu')(out)
        print(out._keras_shape)
        
        return Model(inputs=inputs, outputs=out)

    def get_padding(self):
        return 0

class ParallelNoAggregationModel4ColTest:

    def __init__(self, num_channels):
        self.num_channels = num_channels

    def get_model(self, input_shape):
        import keras
        from keras.models import Model
        from keras.layers import Input, Conv2D, Activation
        
        inputs = Input(input_shape)
        x = inputs
        print(x._keras_shape)

        # col 1

        # 3x3 convolution layer
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', )(x)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)
        
        # 3x3 convolution layer
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', )(col1)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)
        
        # 3x3 convolution layer with dilation 2
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(col1)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)
        
        # 3x3 convolution layer with dilation 4
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(col1)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)

        # col 2

        # 3x3 convolution layer with dilation 2
        col2 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(x)
        col2 = Activation('relu')(col2)
        print(col2._keras_shape)
        
        # 3x3 convolution layer with dilation 2
        col2 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(col2)
        col2 = Activation('relu')(col2)
        print(col2._keras_shape)
        
        # 3x3 convolution layer with dilation 4
        col2 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(col2)
        col2 = Activation('relu')(col2)
        print(col2._keras_shape)
        
        # 3x3 convolution layer with dilation 8
        col2 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(8,8))(col2)
        col2 = Activation('relu')(col2)
        print(col2._keras_shape)
        
        # col 3

        # 3x3 convolution layer with dilation 4
        col3 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(x)
        col3 = Activation('relu')(col3)
        print(col3._keras_shape)
        
        # 3x3 convolution layer with dilation 4
        col3 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(col3)
        col3 = Activation('relu')(col3)
        print(col3._keras_shape)
        
        # 3x3 convolution layer with dilation 8
        col3 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(8,8))(col3)
        col3 = Activation('relu')(col3)
        print(col3._keras_shape)
        
        # 3x3 convolution layer with dilation 16
        col3 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(16,16))(col2)
        col3 = Activation('relu')(col3)
        print(col3._keras_shape)

        # col 4

        # 5x5 convolution layer with dilation 2
        col4 = Conv2D(self.num_channels, (5, 5), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(x)
        col4 = Activation('relu')(col4)
        print(col4._keras_shape)
        
        # 5x5 convolution layer with dilation 2
        col4 = Conv2D(self.num_channels, (5, 5), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(col4)
        col4 = Activation('relu')(col4)
        print(col4._keras_shape)
        
        # 5x5 convolution layer with dilation 4
        col4 = Conv2D(self.num_channels, (5, 5), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(col4)
        col4 = Activation('relu')(col4)
        print(col4._keras_shape)
        
        # 5x5 convolution layer with dilation 8
        col4 = Conv2D(self.num_channels, (5, 5), padding='same', kernel_initializer='he_normal', dilation_rate=(8,8))(col4)
        col4 = Activation('relu')(col4)
        print(col4._keras_shape)
        
        # merge columns

        out = keras.layers.concatenate([col1, col2, col3, col4])

        # 1x1 convolution layer
        # This is essentially a per-pixel fully connected layer,
        # which makes this a "fully convolutional network"
        out = Conv2D(1, (1, 1), kernel_initializer='he_normal')(out)

        # Final ReLU activation (for regression)
        out = Activation('relu')(out)
        print(out._keras_shape)
        
        return Model(inputs=inputs, outputs=out)

    def get_padding(self):
        return 0

class ParallelNoAggregationModel3ColTest:

    def __init__(self, num_channels):
        self.num_channels = num_channels

    def get_model(self, input_shape):
        import keras
        from keras.models import Model
        from keras.layers import Input, Conv2D, Activation
        
        inputs = Input(input_shape)
        x = inputs
        print(x._keras_shape)

        # col 1

        # 3x3 convolution layer
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', )(x)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)
        
        # 3x3 convolution layer
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', )(col1)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)
        
        # 3x3 convolution layer with dilation 2
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(col1)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)
        
        # 3x3 convolution layer with dilation 4
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(col1)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)

        # col 2

        # 3x3 convolution layer with dilation 2
        col2 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(x)
        col2 = Activation('relu')(col2)
        print(col2._keras_shape)
        
        # 3x3 convolution layer with dilation 2
        col2 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(col2)
        col2 = Activation('relu')(col2)
        print(col2._keras_shape)
        
        # 3x3 convolution layer with dilation 4
        col2 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(col2)
        col2 = Activation('relu')(col2)
        print(col2._keras_shape)
        
        # 3x3 convolution layer with dilation 8
        col2 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(8,8))(col2)
        col2 = Activation('relu')(col2)
        print(col2._keras_shape)
        
        # col 3

        # 3x3 convolution layer with dilation 4
        col3 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(x)
        col3 = Activation('relu')(col3)
        print(col3._keras_shape)
        
        # 3x3 convolution layer with dilation 4
        col3 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(col3)
        col3 = Activation('relu')(col3)
        print(col3._keras_shape)
        
        # 3x3 convolution layer with dilation 8
        col3 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(8,8))(col3)
        col3 = Activation('relu')(col3)
        print(col3._keras_shape)
        
        # 3x3 convolution layer with dilation 16
        col3 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(16,16))(col2)
        col3 = Activation('relu')(col3)
        print(col3._keras_shape)
        
        # merge columns

        out = keras.layers.concatenate([col1, col2, col3])

        # 1x1 convolution layer
        # This is essentially a per-pixel fully connected layer,
        # which makes this a "fully convolutional network"
        out = Conv2D(1, (1, 1), kernel_initializer='he_normal')(out)

        # Final ReLU activation (for regression)
        out = Activation('relu')(out)
        print(out._keras_shape)
        
        return Model(inputs=inputs, outputs=out)

    def get_padding(self):
        return 0

class ParallelNoAggregationModel2ColTest:

    def __init__(self, num_channels):
        self.num_channels = num_channels

    def get_model(self, input_shape):
        import keras
        from keras.models import Model
        from keras.layers import Input, Conv2D, Activation
        
        inputs = Input(input_shape)
        x = inputs
        print(x._keras_shape)

        # col 1

        # 3x3 convolution layer
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', )(x)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)
        
        # 3x3 convolution layer
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', )(col1)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)
        
        # 3x3 convolution layer with dilation 2
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(col1)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)
        
        # 3x3 convolution layer with dilation 4
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(col1)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)

        # col 2

        # 3x3 convolution layer with dilation 2
        col2 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(x)
        col2 = Activation('relu')(col2)
        print(col2._keras_shape)
        
        # 3x3 convolution layer with dilation 2
        col2 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(col2)
        col2 = Activation('relu')(col2)
        print(col2._keras_shape)
        
        # 3x3 convolution layer with dilation 4
        col2 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(col2)
        col2 = Activation('relu')(col2)
        print(col2._keras_shape)
        
        # 3x3 convolution layer with dilation 8
        col2 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(8,8))(col2)
        col2 = Activation('relu')(col2)
        print(col2._keras_shape)
        
        # merge columns

        out = keras.layers.concatenate([col1, col2])

        # 1x1 convolution layer
        # This is essentially a per-pixel fully connected layer,
        # which makes this a "fully convolutional network"
        out = Conv2D(1, (1, 1), kernel_initializer='he_normal')(out)

        # Final ReLU activation (for regression)
        out = Activation('relu')(out)
        print(out._keras_shape)
        
        return Model(inputs=inputs, outputs=out)

    def get_padding(self):
        return 0

class ParallelNoAggregationModel1ColTest:

    def __init__(self, num_channels):
        self.num_channels = num_channels

    def get_model(self, input_shape):
        import keras
        from keras.models import Model
        from keras.layers import Input, Conv2D, Activation
        
        inputs = Input(input_shape)
        x = inputs
        print(x._keras_shape)

        # col 1

        # 3x3 convolution layer
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', )(x)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)
        
        # 3x3 convolution layer
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', )(col1)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)
        
        # 3x3 convolution layer with dilation 2
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(2,2))(col1)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)
        
        # 3x3 convolution layer with dilation 4
        col1 = Conv2D(self.num_channels, (3, 3), padding='same', kernel_initializer='he_normal', dilation_rate=(4,4))(col1)
        col1 = Activation('relu')(col1)
        print(col1._keras_shape)

        # merge columns

        # out = keras.layers.concatenate([col1])
        out = col1

        # 1x1 convolution layer
        # This is essentially a per-pixel fully connected layer,
        # which makes this a "fully convolutional network"
        out = Conv2D(1, (1, 1), kernel_initializer='he_normal')(out)

        # Final ReLU activation (for regression)
        out = Activation('relu')(out)
        print(out._keras_shape)
        
        return Model(inputs=inputs, outputs=out)

    def get_padding(self):
        return 0
