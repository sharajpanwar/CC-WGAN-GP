from keras.models import Model
from keras.layers.core import Dense, Activation, Permute, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.regularizers import l1_l2
from keras.layers import Input, Flatten, AveragePooling2D, DepthwiseConv2D, SpatialDropout2D
from keras.layers.convolutional import SeparableConv2D
import os
os.environ["CUDA_VISIBLE_DEVICES"]= '0'

# EEGNet is state of the art model to predict cognitive events proposed as:
# Acknowledgement: codes used in this script are modified from https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py
#                  to fit data dimensions such as time samples, and channels. The EEGNet paper is cited in the article as:
#                  Lawhern, Vernon J., et al. "EEGNet: a compact convolutional neural network for EEG-based braincomputer interfaces."
#                  Journal of neural engineering 15.5 (2018): 056013.

def EEGNet():
    # Keras implementation of EEGNet (arXiv 1611.08024)
    input_conv        = Input((64, 64, 1), name = 'input')
    #The EEG data (X_train, X_test needs to be in the format: (eeg_epochs)x(channels)x(time_steps)
    conv_block1       = Conv2D(16, (64, 1), input_shape=(64, 64, 1),
                                 kernel_regularizer = l1_l2(l1=0.0001, l2=0.0001),
                                 name = 'conv_layer1')(input_conv)
    conv_block1       = BatchNormalization(axis=-1, name = 'bn_1')(conv_block1)
    conv_block1       = ELU(name = 'elu_1')(conv_block1)
    conv_block1       = Dropout(0.25, name = 'dropout_1')(conv_block1)
    # permute_block changes dimension ordering to (time_steps)x(Channels)X(1)

    permute_block     = Permute((2, 1, 3) , name = 'permute_1')(conv_block1)
    conv_block2       = Conv2D(4, (2, 16), padding = 'same',
                            kernel_regularizer=l1_l2(l1=0.0, l2=0.0001),
                            strides = (2, 4), name = 'conv_layer2')(permute_block)

    conv_block2       = BatchNormalization(axis=-1, name = 'bn_2')(conv_block2)
    conv_block2       = ELU(name = 'elu_2')(conv_block2)
    conv_block2       = Dropout(0.25, name = 'dropout_2')(conv_block2)
    conv_block3       = Conv2D(4, (8, 2), padding = 'same',
                            kernel_regularizer=l1_l2(l1=0.0, l2=0.0001),
                            strides = (2, 4), name = 'conv_layer3')(conv_block2)
    conv_block3       = BatchNormalization(axis=-1, name = 'bn_3')(conv_block3)
    conv_block3       = ELU(name = 'elu_3')(conv_block3)
    conv_block3       = Dropout(0.25, name = 'dropout_3')(conv_block3)
    
    flatten_layer     = Flatten(name = 'flatten')(conv_block3)
    dense_layer       = Dense(2, name = 'dense')(flatten_layer)
    out_put           = Activation('softmax', name = 'softmax')(dense_layer)
    
    return Model(input_conv, out_put)


def EEGNet2(nb_classes=2, Chans=64, Samples=64, regRate=0.001,
            dropoutRate=0.25, kernLength=64, numFilters=8):
    """ EEGNet variant that does band-pass filtering first, implemented here
    as a temporal convolution, prior to learning spatial filters. Here, we
    use a Depthwise Convolution to learn the spatial filters as opposed to
    regular Convolution as depthwise allows us to learn a spatial filter per
    temporal filter, without it being fully-connected to all feature maps
    from the previous layer. This helps primarily to reduce the number of
    parameters to fit... it also more closely represents standard BCI
    algorithms such as filter-bank CSP.

    """

    input_conv = Input(shape=(1, Chans, Samples))

    conv_block1 = Conv2D(numFilters, (1, kernLength), padding='same',
                    kernel_regularizer=l1_l2(l1=0.0, l2=0.0),
                    input_shape=(1, Chans, Samples),
                    use_bias=False)(input_conv)
    conv_block1 = BatchNormalization(axis=1)(conv_block1)
    conv_block1 = DepthwiseConv2D((Chans, 1),
                             depthwise_regularizer=l1_l2(l1=regRate, l2=regRate),
                             use_bias=False)(conv_block1)
    conv_block1 = BatchNormalization(axis=1)(conv_block1)
    conv_block1 = Activation('elu')(conv_block1)
    conv_block1 = SpatialDropout2D(dropoutRate)(conv_block1)

    conv_block2 = SeparableConv2D(numFilters, (1, 8),
                             depthwise_regularizer=l1_l2(l1=0.0, l2=regRate),
                             use_bias=False, padding='same')(conv_block1)
    conv_block2 = BatchNormalization(axis=1)(conv_block2)
    conv_block2 = Activation('elu', name='elu_2')(conv_block2)
    conv_block2 = AveragePooling2D((1, 4))(conv_block2)
    conv_block2 = SpatialDropout2D(dropoutRate, name='drop_2')(conv_block2)

    conv_block3 = SeparableConv2D(numFilters * 2, (1, 8), depth_multiplier=2,
                             depthwise_regularizer=l1_l2(l1=0.0, l2=regRate),
                             use_bias=False, padding='same')(conv_block2)
    conv_block3 = BatchNormalization(axis=1)(conv_block3)
    conv_block3 = Activation('elu', name='elu_3')(conv_block3)
    conv_block3 = AveragePooling2D((1, 4))(conv_block3)
    conv_block3 = SpatialDropout2D(dropoutRate, name='drop_3')(conv_block3)

    flatten_layer = Flatten(name='flatten')(conv_block3)

    dense_layer = Dense(nb_classes, name='dense')(flatten_layer)
    out_put = Activation('softmax', name='softmax')(dense_layer)

    return Model(inputs=input_conv, outputs=out_put)

