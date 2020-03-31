from __future__ import print_function, division
import keras
import tensorflow as tf
import os
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten, GaussianNoise, multiply, Embedding,Dropout,ZeroPadding2D
from keras.layers.convolutional import Convolution2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
import seaborn as sns;
from keras import backend as k
from custom_keras_layers import linear_kernel, clip_layer, UpSampling_2Dcubical

sns.set()
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# TensorFlow wizardry
config = tf.ConfigProto()
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.5
# Create a session with the above options specified.
k.tensorflow_backend.set_session(tf.Session(config=config))

Channels, time_step= 64, 64
weights = linear_kernel(2)

def eeg_generator():

    model = Sequential()

    model.add(Dense(1024, input_dim=120, name='linear'))
    model.add(LeakyReLU())

    model.add(Dense(128 * 18 * 18))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    if K.image_data_format() == 'channels_first':
        model.add(Reshape((128, 18, 18), input_shape=(128 * 18 * 18,)))
        bn_axis = 1
    else:
        model.add(Reshape((18, 18, 128), input_shape=(128 * 18 * 18,)))
        bn_axis = -1

    model.add(UpSampling_2Dcubical(2))
    model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())

    model.add(Convolution2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2),
                              kernel_initializer=keras.initializers.Constant(weights),padding='same'))

    model.add(clip_layer())
    model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())

    model.add(Convolution2D(1, (3, 3), padding='same', activation='tanh'))

    model.summary()

    return model
    
def eeg_discriminator():

    model = Sequential()

    model.add(GaussianNoise(0.05, input_shape=(64, 64, 1)))  # Add this layer to prevent D from overfitting!

    if K.image_data_format() == 'channels_first':
        model.add(Convolution2D(64, (3, 3), padding='same', input_shape=(1, 64, 64)))
    else:
        model.add(Convolution2D(64, (3, 3), padding='same', input_shape=(64, 64, 1)))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Convolution2D(128, (3, 3), kernel_initializer='he_normal', padding='same', strides=[2, 2]))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Convolution2D(128, (3, 3), kernel_initializer='he_normal', padding='same', strides=[2, 2]))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Flatten())
    
    model.add(Dense(1024, kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Dense(1, kernel_initializer='he_normal'))

    model.summary()

    return model
