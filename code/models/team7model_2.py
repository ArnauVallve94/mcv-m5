# Keras imports
from keras.utils.vis_utils import plot_model as plot
from keras.models import Model, Sequential, Input
from keras.layers import Dense, Dropout, Activation, Flatten, AveragePooling2D, BatchNormalization, merge, Add
from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
                                        ZeroPadding2D)


def build_vgg(img_shape=(3, 224, 224), n_classes=1000, l2_reg=0.):
    img_input = Input(shape=img_shape)

    x = Convolution2D(64, 7, 7, strides=(2, 2), activation='relu', border_mode='same', name='block1_conv1')(img_input)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='MaxPool_1')(x)
    x = BatchNormalization(name='Batch_1')(x)

    x = Convolution2D(64, 1, 1, activation='relu', border_mode='same', name='block2_conv1')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block2_conv2')(x)
    x = BatchNormalization(name='Batch_2')(x)

    steem_output = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='MaxPool_2')(x)

    inc_1 = Convolution2D(64, 1, 1, border_mode='same', activation='relu', name='inc_1')(steem_output)

    inc_2_a = Convolution2D(96, 1, 1, border_mode='same', activation='relu', name='inc_2_a')(steem_output)
    inc_2_b = Convolution2D(128, 3, 3, border_mode='same', activation='relu', name='inc_2_b')(inc_2_a)

    inc_3_a = Convolution2D(16, 1, 1, border_mode='same', activation='relu', name='inc_3_a')(steem_output)
    inc_3_b = Convolution2D(32, 5, 5, border_mode='same', activation='relu', name='inc_3_b')(inc_3_a)

    inc_4_a = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='inc_4_a')(steem_output)
    inc_4_b = Convolution2D(32, 1, 1, border_mode='same', activation='relu', name='inc_4_b')(inc_4_a)

    out_inc_1 = merge([inc_1, inc_2_b, inc_3_b, inc_4_b], mode='concat', concat_axis=1, name='out_inc_1')

    add_1 = Add()([steem_output, out_inc_1])

    add_1 = Convolution2D(480, 1, 1, border_mode='same', activation='relu', name='add_1')(add_1)

    inc_2_1 = Convolution2D(128, 1, 1, border_mode='same', activation='relu', name='inc_2_1')(add_1)

    inc_2_2_a = Convolution2D(128, 1, 1, border_mode='same', activation='relu', name='inc_2_2_a')(add_1)
    inc_2_2_b = Convolution2D(192, 3, 3, border_mode='same', activation='relu', name='inc_2_2_b')(inc_2_2_a)

    inc_2_3_a = Convolution2D(32, 1, 1, border_mode='same', activation='relu', name='inc_2_2_a')(add_1)
    inc_2_3_b = Convolution2D(96, 5, 5, border_mode='same', activation='relu', name='inc_2_2_b')(inc_2_3_a)

    inc_2_4_a = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same', name='inc_2_4_a')(add_1)
    inc_2_4_b = Convolution2D(64, 1, 1, border_mode='same', activation='relu', name='inc_2_4_b')(inc_2_4_a)

    out_inc_2 = merge([inc_2_1, inc_2_2_b, inc_2_3_b, inc_2_4_b], mode='concat', concat_axis=1, name='out_inc_2')

    add_2 = Add()([add_1, out_inc_2])

    out_inc = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='MaxPool_3')(add_2)

    conv_1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1')(out_inc)

    x = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), name='AvgPool')(out_inc)

    x = Dense(4096, activation='relu', name='lyr1_dense')(x)
    x = Dropout(rate=0.5)(x)
    x = Dense(1000, activation='relu', name='lyr2_dense')(x)
    x = Dropout(rate=0.5)(x)
    model = Dense(n_classes, activation='softmax', name='predictions')(x)

    plot(model, to_file='team7_model_2.png', show_shapes=True, show_layer_names=True)

    return model
