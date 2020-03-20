# generate target and nonTarget from CC_GAN
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model, Model, Sequential
import scipy.io
import os
import seaborn as sns; sns.set()
import tensorflow as tf
from keras import backend as k
from plots_python import create_heat_map, create_one_channel_plot
os.environ["CUDA_VISIBLE_DEVICES"]= '0'

## set GPU options
# TensorFlow wizardry
config = tf.ConfigProto()
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.5
# Create a session with the above options specified.
k.tensorflow_backend.set_session(tf.Session(config=config))

# upload saved model from directory
g_model = load_model("./saved_models/generator_model.h5", custom_objects={'tf':tf})

# directory to store generated data
out_put_dir= './cc_gan_data'
# creating random input for generator model
noise1 = np.random.normal(0, 1, (784, 120)).astype(np.float32)
l1=np.random.randint(1, 2, 784).reshape(-1, 1)

# generating target samples
target_gen_cc_gan = g_model.predict([noise1, l1.astype('float32')])
print (target_gen_cc_gan.shape)

# save target data as numpy file
np.save(out_put_dir+'/target_gen_cc_gan.npy', target_gen_cc_gan )

# save target data as matlab file
y = np.cos(target_gen_cc_gan)
scipy.io.savemat(out_put_dir+'/targe_gen_cc_gant.mat', dict(target_gen_cc_gan=target_gen_cc_gan, y=y))

# creating random input for generator model
noise2 = np.random.normal(0, 1, (784, 120)).astype(np.float32)
l2=np.random.randint(0, 1, 784).reshape(-1, 1)

# generating nontarget samples
nontarget_gen_cc_gan = g_model.predict([noise2, l2.astype('float32')])
print (nontarget_gen_cc_gan.shape)

# save nontarget data as numpy file
np.save(out_put_dir+'/nontarget_gen_cc_gan.npy', nontarget_gen_cc_gan )

# save nontarget data as matlab file
y = np.cos(nontarget_gen_cc_gan)
scipy.io.savemat(out_put_dir+'/nontarget_gen_cc_gan.mat', dict(nontarget_gen_cc_gan=nontarget_gen_cc_gan, y=y))

# Create Heat Maps for visualization
create_heat_map(target_gen_cc_gan, out_put_dir, 'Target')
create_heat_map(nontarget_gen_cc_gan, out_put_dir, 'NonTarget')

# Plot POz channel
create_one_channel_plot(target_gen_cc_gan, 29, out_put_dir, 'Target_POz')
create_one_channel_plot(nontarget_gen_cc_gan, 29, out_put_dir, 'NonTarget_POz')
