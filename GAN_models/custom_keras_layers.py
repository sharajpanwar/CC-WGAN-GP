# Author: Sharaj Panwar, MSEE; Research Fellow at Brain Computer Interface Lab./& Open Cloud Institute at UTSA, San Antonio, Texas
# keras custom layers created for the experiments performed in https://arxiv.org/ftp/arxiv/papers/1911/1911.04379.pdffrom keras import Sequential
import numpy as np
from keras.layers import Lambda
import tensorflow as tf
from keras import backend as K

Channels, time_step = 64, 64

# custom keras layer to perform 1D upsampling in time dimension using cubical interpolation
# this layer will be used in one channel GAN model  
def UpSampling_2Dcubical(stride, **kwargs):
    def layer(x):
        input_shape = K.int_shape(x)
        output_shape = (stride * input_shape[1], stride * input_shape[2])
        # return tf.image.resize_linear(x, output_shape, align_corners=True)
        # return tf.image.resize_nearest_neighbor(x, output_shape, align_corners=True)
        return tf.image.resize_bicubic(x, output_shape, align_corners=True)
    return Lambda(layer, **kwargs)
    
# keras layer to perform 2D upsampling in both Channel and time dimensions using cubical interpolation
# this layer will be used in 64 channel GAN model 
def UpSampling_1Dcubical(stride, **kwargs):
    def layer(x):
        input_shape = K.int_shape(x)
        output_shape = (input_shape[1], stride * input_shape[2])
        # return tf.image.resize_linear(x, output_shape, align_corners=True)
        # return tf.image.resize_nearest_neighbor(x, output_shape, align_corners=True)
        return tf.image.resize_bicubic(x, output_shape, align_corners=True)
    return Lambda(layer, **kwargs)

# Weights for Deconvolution kernel with Bilinear interpolation for one channel GAN model
num_channels=1
def Bilinear_kernel(stride): # stride = 2; num_channels = 1
    filter_size = (2 * stride - stride % 2)
    # Create Bilinear weights in numpy array
    Bilinear_kernel = np.zeros([1, filter_size], dtype=np.float32)
    scale_factor = (filter_size + 1) // 2
    if filter_size % 2 == 1:
        center = scale_factor - 1
    else:
        center = scale_factor - 0.5
    for x in range(1):
        for y in range(filter_size):
            Bilinear_kernel[x, y] = (1 - abs(x - center) / scale_factor) * \
                                    (1 - abs(y - center) / scale_factor)
    weights = np.zeros((1, filter_size, num_channels, num_channels))
    for i in range(num_channels):
        weights[:, :, i, i] = Bilinear_kernel
    weights = weights.astype('float32')
    return weights
# These weights will be used to initialize the deconvolution Bilinear_kernel to imitate Bilinear interpolation
# kernel_initializer=keras.initializers.Constant(weights)

# Weights for Deconvolution kernel with linear interpolation for 64 channel GAN model
num_channels=1
def linear_kernel(stride): # stride = 2; num_channels = 1
    filter_size = (2 * stride - stride % 2)
    # Create linear weights in numpy array
    linear_kernel = np.zeros([filter_size, filter_size], dtype=np.float32)
    scale_factor = (filter_size + 1) // 2
    if filter_size % 2 == 1:
        center = scale_factor - 1
    else:
        center = scale_factor - 0.5
    for x in range(filter_size):
        for y in range(filter_size):
            linear_kernel[x, y] = (1 - abs(x - center) / scale_factor) * \
                                    (1 - abs(y - center) / scale_factor)
    weights = np.zeros((filter_size, filter_size, num_channels, num_channels))
    for i in range(num_channels):
        weights[:, :, i, i] = linear_kernel
    weights = weights.astype('float32')
    return weights
# These weights will be used to initialize the deconvolution linear_kernel to imitate linear interpolation
# kernel_initializer=keras.initializers.Constant(weights)

# Clipping with outer cropping
def clip_layer(**kwargs):
    def layer_c(x):
        return tf.image.resize_image_with_crop_or_pad(x, Channels, time_step)
    return Lambda(layer_c, **kwargs)

# normalizing over mean
def mean_zero_layer(**kwargs):
    def layer_mean(x):
        mu = tf.reduce_mean(tf.reduce_mean(x, axis=2), axis=0)
        x =x-tf.expand_dims(mu, -1)
        return x
    return Lambda(layer_mean, **kwargs)
