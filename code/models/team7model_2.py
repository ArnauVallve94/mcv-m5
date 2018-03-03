# Keras imports
from keras.utils.vis_utils import plot_model as plot
from keras.models import Model, Input
from keras.layers import Dense, Dropout, Flatten, AveragePooling2D, BatchNormalization, merge, Add
from keras.layers.convolutional import (Convolution2D, MaxPooling2D)


def build_own(img_shape=(3, 224, 224), n_classes=1000):
    img_input = Input(shape=img_shape)

    # Steem
    x = Convolution2D(64, 7, 7, subsample=(2, 2), activation='relu', border_mode='same', name='block1_conv1')(img_input)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='MaxPool_1')(x)
    x = BatchNormalization(name='Batch_1')(x)

    x = Convolution2D(64, 1, 1, activation='relu', border_mode='same', name='block2_conv1')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block2_conv2')(x)
    x = BatchNormalization(name='Batch_2')(x)

    steem_output = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='steem_output')(x)

    # Inception 1
    inc_1_1 = Convolution2D(64, 1, 1, border_mode='same', activation='relu', name='inc_1_1')(steem_output)

    inc_1_2_a = Convolution2D(96, 1, 1, border_mode='same', activation='relu', name='inc_1_2_a')(steem_output)
    inc_1_2_b = Convolution2D(128, 3, 3, border_mode='same', activation='relu', name='inc_1_2_b')(inc_1_2_a)

    inc_1_3_a = Convolution2D(16, 1, 1, border_mode='same', activation='relu', name='inc_1_3_a')(steem_output)
    inc_1_3_b = Convolution2D(32, 5, 5, border_mode='same', activation='relu', name='inc_1_3_b')(inc_1_3_a)

    inc_1_4_a = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='inc_1_4_a')(steem_output)
    inc_1_4_b = Convolution2D(32, 1, 1, border_mode='same', activation='relu', name='inc_1_4_b')(inc_1_4_a)

    out_inc_1 = merge([inc_1_1, inc_1_2_b, inc_1_3_b, inc_1_4_b], mode='concat', concat_axis=1, name='out_inc_1')

    # Residual 1
    add_1 = Add()([steem_output, out_inc_1])
    add_1 = Convolution2D(480, 1, 1, border_mode='same', activation='relu', name='add_1')(add_1)

    # Inception 2
    inc_2_1 = Convolution2D(128, 1, 1, border_mode='same', activation='relu', name='inc_2_1')(add_1)

    inc_2_2_a = Convolution2D(128, 1, 1, border_mode='same', activation='relu', name='inc_2_2_a')(add_1)
    inc_2_2_b = Convolution2D(192, 3, 3, border_mode='same', activation='relu', name='inc_2_2_b')(inc_2_2_a)

    inc_2_3_a = Convolution2D(32, 1, 1, border_mode='same', activation='relu', name='inc_2_2_a')(add_1)
    inc_2_3_b = Convolution2D(96, 5, 5, border_mode='same', activation='relu', name='inc_2_2_b')(inc_2_3_a)

    inc_2_4_a = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same', name='inc_2_4_a')(add_1)
    inc_2_4_b = Convolution2D(64, 1, 1, border_mode='same', activation='relu', name='inc_2_4_b')(inc_2_4_a)

    out_inc_2 = merge([inc_2_1, inc_2_2_b, inc_2_3_b, inc_2_4_b], mode='concat', concat_axis=1, name='out_inc_2')

    # Residual 2
    add_2 = Add()([add_1, out_inc_2])

    # CNN block
    out_inc = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='out_inc')(add_2)
    conv_3 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block3_conv1')(out_inc)
    mp_4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='MaxPool_4')(conv_3)
    conv_4 = Convolution2D(1024, 3, 3, activation='relu', border_mode='same', name='block4_conv1')(mp_4)

    avg_pool = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), name='AvgPool')(conv_4)
    fl = Flatten()(avg_pool)

    # Fully connected
    fc1000 = Dense(1000, activation='relu', name='lyr2_dense')(fl)
    dr_out = Dropout(rate=0.3)(fc1000)
    predictions = Dense(n_classes, activation='softmax', name='predictions')(dr_out)

    model = Model(input=img_input.input, output=predictions)

    plot(model, to_file='team7_model_2.png', show_shapes=True, show_layer_names=True)

    return model
