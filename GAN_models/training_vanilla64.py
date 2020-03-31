# Author: Sharaj Panwar, MSEE; Research Fellow at Brain Computer Interface Lab./& Open Cloud Institute at UTSA, San Antonio, Texas
# Acknowledgement: this code takes basic WGAN framework from keras implementation "https://github.com/keras-team/
# keras-contrib/blob/master/examples/improved_wgan.py" of the improved WGAN described in https://arxiv.org/abs/1704.00028

from __future__ import print_function, division
import tensorflow as tf
import keras
import os
import numpy as np
from keras.models import Model
from keras.layers import Input
from keras.layers.merge import _Merge
from keras.optimizers import Adam
from keras import backend as K
from functools import partial
import pandas as pd
import seaborn as sns;
# importing custom modules created for GAN training
from out_put_module import generate_eeg, plot_losses
from wgan_gp_loss import wasserstein_loss, gradient_penalty_loss
from arch_vanilla64 import eeg_generator, eeg_discriminator

sns.set()
os.environ["CUDA_VISIBLE_DEVICES"]= '1'
# TensorFlow wizardry
config = tf.ConfigProto()
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.5
K.tensorflow_backend.set_session(tf.Session(config=config))

batch_size = 64
training_ratio = 2  # The training ratio is the number of discriminator updates per generator update.
GP_Weight = 10  # As per the paper
Number_epochs = 1000
sample_freq = 10 # frequency at which you want to generate image or save generator models
Channels, time_step = 64, 64 #eeg samples' dimension

# we train for target and nontarget separtely
# let's train for target first
X_train = np.load('/home-new/ijz121/PycharmProjects/gan_projects/gan_rsvp_project/'
                 'rsvp_data/data_rsvp_preprocessed/rsvp_64_ch_balanced/T_ch64_s1_r2.npy').astype('float32')
# reshape the data to fit in model
if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, Channels, time_step)
    input_shape = (1, Channels, time_step)
else:
    X_train = X_train.reshape(X_train.shape[0], Channels, time_step, 1)
    input_shape = (Channels, time_step, 1)

# directories to store the results
output_dir = '/home-new/ijz121/PycharmProjects/guest_sharaj_works/op_data/'
# Run tag will dynamically create the sub-directories to store results
Run_tag = 'vanilla64_target_' + str(Number_epochs) + '_epoch'
print (Run_tag)
#  sub-directories to store results
dir = os.path.join(output_dir, Run_tag)
if not os.path.exists(dir):
    os.mkdir(dir)
    os.mkdir(dir + '/eeg_generated')
    os.mkdir(dir + '/evaluation')
    os.mkdir(dir + '/saved_model')
else:
    raise Exception('Excuse me Boss!! Please change the Run Tag to avoid rewriting files ;)')

class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        weights = K.random_uniform((batch_size, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])
    
# create generator and discriminator models imported from arch_vanilla64
generator = eeg_generator()
discriminator = eeg_discriminator()

for layer in discriminator.layers:
    layer.trainable = False
discriminator.trainable = False

generator_input = Input(shape=(120,))
generator_layers = generator(generator_input)

discriminator_layers_for_generator = discriminator(generator_layers)
generator_model = Model(inputs=(generator_input), outputs=[discriminator_layers_for_generator])
generator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9), loss=wasserstein_loss)

# Now that the generator_model is compiled, we can make the discriminator layers trainable.
for layer in discriminator.layers:
    layer.trainable = True
for layer in generator.layers:
    layer.trainable = False
discriminator.trainable = True
generator.trainable = False

real_samples = Input(shape=X_train.shape[1:])
generator_input_for_discriminator = Input(shape=(120,))
generated_samples_for_discriminator = generator(generator_input_for_discriminator)
discriminator_output_from_generator = discriminator(generated_samples_for_discriminator)
discriminator_output_from_real_samples = discriminator(real_samples)

averaged_samples = RandomWeightedAverage()([real_samples, generated_samples_for_discriminator])
averaged_samples_out = discriminator(averaged_samples)

partial_gp_loss = partial(gradient_penalty_loss,
                          averaged_samples=averaged_samples,
                          gradient_penalty_weight=GP_Weight)
partial_gp_loss.__name__ = 'gradient_penalty'  # Functions need names or Keras will throw an error


discriminator_model = Model(inputs=[real_samples, generator_input_for_discriminator],
                            outputs=[discriminator_output_from_real_samples,
                                     discriminator_output_from_generator,
                                     averaged_samples_out])

discriminator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
                            loss=[wasserstein_loss,
                                  wasserstein_loss,
                                  partial_gp_loss])

positive_y = np.ones((batch_size, 1), dtype=np.float32)
negative_y = -positive_y
dummy_y = np.zeros((batch_size, 1), dtype=np.float32)

epochs = [];
wgan_d_itr =[];wgan_g_itr=[]; #discriminator and generator losses for each iteration in training ratio
wgan_d_ep=[];wgan_g_ep=[]; #discriminator and generator losses for each epoch averaging over training ratio

for epoch in range(Number_epochs):
    np.random.shuffle(X_train)
    print("Epoch: ", epoch)
    print("Number of batches: ", int(X_train.shape[0] // batch_size))
    discriminator_loss = []
    generator_loss = []

    minibatches_size = batch_size * training_ratio
    for i in range(int(X_train.shape[0] // (batch_size * training_ratio))):
        discriminator_minibatches = X_train[i * minibatches_size:(i + 1) * minibatches_size]

        for j in range(training_ratio):
            eeg_batch = discriminator_minibatches[j * batch_size:(j + 1) * batch_size]
            noise = np.random.normal(0, 1, (batch_size, 120)).astype(np.float32)
            discriminator_loss.append(discriminator_model.train_on_batch([eeg_batch, noise],
                                                                         [positive_y, negative_y, dummy_y]))

        generator_loss.append(generator_model.train_on_batch(np.random.normal(0, 1, (batch_size, 120)), positive_y))

    epochs.append(epoch)
    wgan_d_itr.append(np.array(discriminator_loss)[:, 0])
    wgan_g_itr.append(np.array(generator_loss))

    wgan_d_ep.append(np.mean(discriminator_loss, axis=0))
    wgan_g_ep.append(np.mean(np.array(generator_loss), axis=0))

    generate_eeg(generator, dir, epoch, sample_freq)

# saving 'losses, average over training ratio' to csv file
validation_matrix = pd.DataFrame({'epoch': epochs, 'Discriminator_loss': wgan_d_ep, 'Generator_loss': wgan_g_ep})
validation_matrix.to_csv(str(dir) + '/evaluation/'+ 'validation_matrix.csv')

# plotting losses per iteration
wgan_d_itr = np.array(wgan_d_itr).flatten()
wgan_g_itr = np.array(wgan_g_itr).flatten()
plot_losses(wgan_d_itr, wgan_g_itr, dir)