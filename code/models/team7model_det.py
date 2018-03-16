# Keras imports
from keras.utils.vis_utils import plot_model as plot
from keras.models import Model, Input
from keras.layers import BatchNormalization, concatenate, Add, merge, LeakyReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D

from layers.yolo_layers import YOLOConvolution2D, Reorg


def build_own_det(img_shape=(3, 224, 224), n_classes=1000, num_priors=5):
    img_input = Input(shape=img_shape)

    # Steem
    x = YOLOConvolution2D(64, 7, 7, border_mode='same', subsample=(1,1), epsilon=0.000001, name='block1_conv1')(img_input)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(4, 4), padding='same', name='MaxPool_1')(x)
    x = BatchNormalization(name='Batch_1')(x)

    x = YOLOConvolution2D(64, 1, 1, border_mode='same', subsample=(1,1), epsilon=0.000001, name='block2_conv1')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = YOLOConvolution2D(256, 3, 3, border_mode='same', subsample=(1,1), epsilon=0.000001, name='block2_conv2')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization(name='Batch_2')(x)

    steem_output = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='steem_output')(x)

    # Inception 1
    inc_1_1 = YOLOConvolution2D(64, 1, 1, border_mode='same', subsample=(1,1), epsilon=0.000001, name='inc_1_1')(steem_output)
    inc_1_1 = LeakyReLU(alpha=0.1)(inc_1_1)

    inc_1_2_a = YOLOConvolution2D(96, 1, 1, border_mode='same', subsample=(1,1), epsilon=0.000001, name='inc_1_2_a')(steem_output)
    inc_1_2_a = LeakyReLU(alpha=0.1)(inc_1_2_a)

    inc_1_2_b = YOLOConvolution2D(128, 3, 3, border_mode='same', subsample=(1,1), epsilon=0.000001, name='inc_1_2_b')(inc_1_2_a)
    inc_1_2_b = LeakyReLU(alpha=0.1)(inc_1_2_b)

    inc_1_3_a = YOLOConvolution2D(16, 1, 1, border_mode='same', subsample=(1,1), epsilon=0.000001, name='inc_1_3_a')(steem_output)
    inc_1_3_a = LeakyReLU(alpha=0.1)(inc_1_3_a)

    inc_1_3_b = YOLOConvolution2D(32, 5, 5, border_mode='same', subsample=(1,1), epsilon=0.000001, name='inc_1_3_b')(inc_1_3_a)
    inc_1_3_b = LeakyReLU(alpha=0.1)(inc_1_3_b)

    inc_1_4_a = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='inc_1_4_a')(steem_output)
    inc_1_4_b = YOLOConvolution2D(32, 1, 1, border_mode='same', subsample=(1,1), epsilon=0.000001, name='inc_1_4_b')(inc_1_4_a)
    inc_1_4_b = LeakyReLU(alpha=0.1)(inc_1_4_b)

    out_inc_1 = merge([inc_1_1, inc_1_2_b, inc_1_3_b, inc_1_4_b], mode='concat', concat_axis=1, name='out_inc_1')

    # Residual 1
    add_1 = Add()([steem_output, out_inc_1])
    add_1 = YOLOConvolution2D(480, 1, 1, border_mode='same', subsample=(1,1), epsilon=0.000001, name='add_1')(add_1)
    add_1 = LeakyReLU(alpha=0.1)(add_1)

    # Inception 2
    inc_2_1 = YOLOConvolution2D(128, 1, 1, border_mode='same', subsample=(1,1), epsilon=0.000001, name='inc_2_1')(add_1)
    inc_2_1 = LeakyReLU(alpha=0.1)(inc_2_1)

    inc_2_2_a = YOLOConvolution2D(128, 1, 1, border_mode='same', subsample=(1,1), epsilon=0.000001, name='inc_2_2_a')(add_1)
    inc_2_2_a = LeakyReLU(alpha=0.1)(inc_2_2_a)

    inc_2_2_b = YOLOConvolution2D(192, 3, 3, border_mode='same', subsample=(1,1), epsilon=0.000001, name='inc_2_2_b')(inc_2_2_a)
    inc_2_2_b = LeakyReLU(alpha=0.1)(inc_2_2_b)

    inc_2_3_a = YOLOConvolution2D(32, 1, 1, border_mode='same', subsample=(1,1), epsilon=0.000001, name='inc_2_2_a')(add_1)
    inc_2_3_a = LeakyReLU(alpha=0.1)(inc_2_3_a)

    inc_2_3_b = YOLOConvolution2D(96, 5, 5, border_mode='same', subsample=(1,1), epsilon=0.000001, name='inc_2_2_b')(inc_2_3_a)
    inc_2_3_b = LeakyReLU(alpha=0.1)(inc_2_3_b)

    inc_2_4_a = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same', name='inc_2_4_a')(add_1)
    inc_2_4_b = YOLOConvolution2D(64, 1, 1, border_mode='same', subsample=(1,1), epsilon=0.000001, name='inc_2_4_b')(inc_2_4_a)
    inc_2_4_b = LeakyReLU(alpha=0.1)(inc_2_4_b)

    out_inc_2 = merge([inc_2_1, inc_2_2_b, inc_2_3_b, inc_2_4_b], mode='concat', concat_axis=1, name='out_inc_2')

    # Residual 2
    add_2 = Add()([add_1, out_inc_2])

    # Parallel block
    out_inc = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='out_inc')(add_2)
    conv_3 = YOLOConvolution2D(512, 3, 3, border_mode='same', subsample=(1,1), epsilon=0.000001, name='block3_conv1')(out_inc)
    conv_3 = LeakyReLU(alpha=0.1)(conv_3)

    conv_4 = YOLOConvolution2D(1024, 3, 3, border_mode='same', subsample=(1,1), epsilon=0.000001, name='block4_conv1')(conv_3)
    conv_4 = LeakyReLU(alpha=0.1)(conv_4)

    conv_5 = YOLOConvolution2D(512, 3, 3, border_mode='same', subsample=(1,1), epsilon=0.000001, name='block5_conv1')(conv_4)
    conv_5 = LeakyReLU(alpha=0.1)(conv_5)

    conv_6 = YOLOConvolution2D(1024, 3, 3, border_mode='same', subsample=(1,1), epsilon=0.000001, name='block6_conv1')(conv_5)
    conv_6 = LeakyReLU(alpha=0.1)(conv_6)

    conv_7 = YOLOConvolution2D(512, 3, 3, border_mode='same', subsample=(1,1), epsilon=0.000001, name='block7_conv1')(conv_6)
    conv_7 = LeakyReLU(alpha=0.1)(conv_7)

    conv_8 = YOLOConvolution2D(1024, 3, 3, border_mode='same', subsample=(1,1), epsilon=0.000001, name='block8_conv1')(conv_7)
    conv_8 = LeakyReLU(alpha=0.1)(conv_8)

    reorg = (Reorg())(add_2)

    concat = concatenate(conv_8, reorg, axis=1)

    conv_9 = YOLOConvolution2D(1024, 3, 3, border_mode='same', subsample=(1,1), epsilon=0.000001, name='block9_conv1')(concat)
    conv_9 = LeakyReLU(alpha=0.1)(conv_9)

    last_conv = Convolution2D(num_priors * (4 + n_classes + 1), (1, 1), padding='same', strides=(1, 1))(conv_9)

    model = Model(input=img_input.input, output=last_conv)

    plot(model, to_file='team7_model_det.png', show_shapes=True, show_layer_names=True)

    return model
