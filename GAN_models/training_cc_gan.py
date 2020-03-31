# Author: Sharaj Panwar, MSEE; Research Fellow at Brain Computer Interface Lab./& Open Cloud Institute at UTSA, San Antonio, Texas
# Acknowledgement: this code takes basic WGAN framework from keras implementation "https://github.com/keras-team/
# keras-contrib/blob/master/examples/improved_wgan.py" of the improved WGAN described in https://arxiv.org/abs/1704.00028

from __future__ import print_function, division
import keras
import tensorflow as tf
import os
import argparse
import numpy as np
from keras.models import Model
from keras.layers import Input
from keras.layers.merge import _Merge
from keras.optimizers import Adam
from keras import backend as K
from functools import partial
import pandas as pd
from random import shuffle
from sklearn.metrics import roc_auc_score
import seaborn as sns;
from keras import backend as k
# importing custom modules created for GAN training
from wgan_gp_loss import wasserstein_loss, gradient_penalty_loss
from arch_cc_gan import eeg_generator, eeg_discriminator
from data_loader import data_import
from out_put_module import save_CC_GAN, plot_AUC

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

batch_size = 64
training_ratio = 2  # The training ratio is the number of discriminator updates per generator update. The paper uses 5.
GP_Weight = 10  # gradient_penalty As per Improved WGAN paper

# For training CC_GAN across sessions within subjects=> four sessions out of five will be treated as training data 
# and remaining ones as test data for each subject. This script requires three positional arguments first is subject  
# to train (out of 8), second is test session (out of 5), and third argument is number of training epochs;
# For example "./$ python training_cc_gan 3 4 100" means we are creating the model on subject 3; 
# Training on -: session 1,2,3,5, test on session 4, and training will occur for 100 epochs.

parser = argparse.ArgumentParser()
parser.add_argument("sub", help="enter the subject") # subject number
parser.add_argument("sess", help="enter the test session") # test session
parser.add_argument("tr_epoch", help="enter the test session") # test session
args = parser.parse_args()
print ("CC-GAN training on subject {}, Test session {}".format(args.sub, args.sess))


input_dir = '/home/guest/PycharmProjects/sharaj_works/input_data/rsvp_session_wise/'
# directory where all the data is kept in sub-directories named with subjects
# e.g.# dir/S1/T_s1_r1, T1_s1_r1 stands for target subject 1 session 1

sub = args.sub # parsing the subject No.
sess = args.sess # parsing the test session No.
Number_epochs =args.tr_epoch # parsing the numbers of training epochs.

# uploading target sessions data for the given subject
a1 = np.load(input_dir + 's' + str(sub) + '/T_s' + str(sub) + '_r1.npy')
a2 = np.load(input_dir + 's' + str(sub) + '/T_s' + str(sub) + '_r2.npy')
a3 = np.load(input_dir + 's' + str(sub) + '/T_s' + str(sub) + '_r3.npy')
a4 = np.load(input_dir + 's' + str(sub) + '/T_s' + str(sub) + '_r4.npy')
a5 = np.load(input_dir + 's' + str(sub) + '/T_s' + str(sub) + '_r5.npy')
# uploading nonTarget sessions data for the given subject
b1 = np.load(input_dir + 's' + str(sub) + '/nT_s' + str(sub) + '_r1.npy')
b2 = np.load(input_dir + 's' + str(sub) + '/nT_s' + str(sub) + '_r2.npy')
b3 = np.load(input_dir + 's' + str(sub) + '/nT_s' + str(sub) + '_r3.npy')
b4 = np.load(input_dir + 's' + str(sub) + '/nT_s' + str(sub) + '_r4.npy')
b5 = np.load(input_dir + 's' + str(sub) + '/nT_s' + str(sub) + '_r5.npy')

session_list = [1,2,3,4,5];
session_list.remove(int(sess))

# creating training data
T_tr = np.concatenate((eval('a'+str(session_list[0])), eval('a'+str(session_list[1])),
                       eval('a'+str(session_list[2])), eval('a'+str(session_list[3]))));
nT_tr = np.concatenate((eval('b'+str(session_list[0])), eval('b'+str(session_list[1])),
                        eval('b'+str(session_list[2])), eval('b'+str(session_list[3]))));
y_T_tr = np.ones(T_tr.shape[0]); y_nT_tr = np.zeros(nT_tr.shape[0])
X_train = np.concatenate((T_tr, nT_tr), axis=0); y_train = np.concatenate((y_T_tr, y_nT_tr), axis=0)

# creating test data
T_te = eval('a'+str(sess)); nT_te = eval('b'+str(sess))
y_T_te = np.ones(T_te.shape[0]); y_nT_te = np.zeros(nT_te.shape[0]);
X_test = np.concatenate((T_te, nT_te), axis=0); y_test = np.concatenate((y_T_te, y_nT_te), axis=0)

# Shuffle and reshape the data to fit in model
X_train, y_train = data_import(X_train, y_train)
X_test, y_test = data_import(X_test, y_test)

y_test = keras.utils.to_categorical(y_test, 2)

# _____________________________________________________________________________________________________

# Now we have uploaded the data, let's create directories to store the results
output_dir = '/home-new/ijz121/PycharmProjects/guest_sharaj_works/op_data/'
# output_dir is the main output directory to store results

# sub-directories will be created for each subject 
sub_dir = output_dir + 's_' + str(sub) + '/'
if not os.path.exists(sub_dir):
    os.mkdir(sub_dir)
# Run tag will dynamically create the sub-directories to store results
Run_tag = 'cc_gan_subj_'+ str(sub)+'_te_sess_' + str(sess)+ '_epoch'+ str(Number_epochs) 
print(Run_tag)

dir = os.path.join(sub_dir, Run_tag)
if not os.path.exists(dir):
    os.mkdir(dir)
    os.mkdir(dir + '/evaluation')
    os.mkdir(dir + '/saved_model')
else:
    raise Exception ('Excuse me Boss!! Please change the Run Tag to avoid rewriting files ;)')

class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        weights = K.random_uniform((batch_size, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

# create generator and discriminator models imported from arch_cc_gan
generator = eeg_generator()
discriminator = eeg_discriminator()

for layer in discriminator.layers:
    layer.trainable = False
discriminator.trainable = False
noise = Input(shape=(120,))
input_label = Input(shape=(1,), dtype='int32')
generator_input = ([noise, input_label])
generator_layers = generator(generator_input)

discriminator_layers_for_generator, label_discriminator_layers_for_generator= discriminator(generator_layers)
generator_model = Model(inputs=[noise, input_label], 
                        outputs=[discriminator_layers_for_generator,label_discriminator_layers_for_generator])

generator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9), 
                        loss=[wasserstein_loss, 'sparse_categorical_crossentropy'])

# Now that the generator_model is compiled, we can make the discriminator layers trainable.
for layer in discriminator.layers:
    layer.trainable = True
for layer in generator.layers:
    layer.trainable = False
discriminator.trainable = True
generator.trainable = False

real_samples = Input(shape=X_train.shape[1:])
generator_input_for_discriminator = Input(shape=(120,))
input_label = Input(shape=(1,), dtype='int32')

generated_samples_for_discriminator = generator([generator_input_for_discriminator,input_label])
discriminator_output_from_generator, label_discriminator_output_from_generator = discriminator(
    generated_samples_for_discriminator)
discriminator_output_from_real_samples, label_discriminator_output_from_real_samples = discriminator(real_samples)

averaged_samples = RandomWeightedAverage()([real_samples, generated_samples_for_discriminator])
averaged_samples_out, _= discriminator(averaged_samples)

partial_gp_loss = partial(gradient_penalty_loss,
                          averaged_samples=averaged_samples,
                          gradient_penalty_weight=GP_Weight)
partial_gp_loss.__name__ = 'gradient_penalty'  # Functions need names or Keras will throw an error

discriminator_model = Model(inputs=[real_samples, generator_input_for_discriminator,input_label],
                            outputs=[discriminator_output_from_real_samples,
                                     label_discriminator_output_from_real_samples,
                                     discriminator_output_from_generator, label_discriminator_output_from_generator,
                                     averaged_samples_out])

discriminator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
                            loss=[wasserstein_loss, 'sparse_categorical_crossentropy', wasserstein_loss,
                                  'sparse_categorical_crossentropy',
                                  partial_gp_loss], loss_weights=[0.5, 0.5, 0.5, 0.5, 1.0])  
positive_y = np.ones((batch_size, 1), dtype=np.float32)
negative_y = -positive_y
dummy_y = np.zeros((batch_size, 1), dtype=np.float32)

epochs = []; AUC_ROC = [0.2];
# discriminator_loss = []; generator_loss = [];
# for CC-GAN,we will use classifier AUC on test data to guide the training.
# So, we will check AUC after each training epoch and save the Generator and Discriminator models for Max AUC
# hence, avoiding discriminator ans classifier losses
for epoch in range(Number_epochs):
    ind_list = [i for i in range(X_train.shape[0])];
    shuffle(ind_list)
    X_train = X_train[ind_list, :];
    y_train = y_train[ind_list]

    print("Epoch: ", epoch)
    print("Number of batches: ", int(X_train.shape[0] // batch_size))
    cw1 = {-1: 1, 1: 1}
    minibatches_size = batch_size * training_ratio

    for i in range(int(X_train.shape[0] // (batch_size * training_ratio))):
        discriminator_minibatches = X_train[i * minibatches_size:(i + 1) * minibatches_size]
        y_value_minibatches = y_train[i * minibatches_size:(i + 1) * minibatches_size]

        for j in range(training_ratio):
            eeg_batch = discriminator_minibatches[j * batch_size:(j + 1) * batch_size]

            y_value2 = y_value_minibatches[j * batch_size:(j + 1) * batch_size].reshape(-1, 1)
            cw2 = {i: 2 / y_value2.shape[0] for i in range(2)}
            cw3 = {0: 1}

            noise = np.random.normal(0, 1, (batch_size, 120)).astype(np.float32)
            sampled_labels = np.random.randint(0, 2, (batch_size,1))

            discriminator_model.train_on_batch([eeg_batch, noise, sampled_labels], 
                    [positive_y, y_value2, negative_y, sampled_labels, dummy_y], class_weight=[cw1, cw2, cw1, cw2, cw3])

        sampled_labels1 = np.random.randint(0, 2, (batch_size, 1))
        valid, label1 = discriminator.predict(X_test, batch_size=32)
        auc = roc_auc_score(y_test, label1[:, :2])

        if auc > max(AUC_ROC):
            discriminator1 = discriminator
            generator1 = generator
            print("AUC", auc)
        else:
            print('No Improvement')
        AUC_ROC.append(auc)
        generator_model.train_on_batch([np.random.normal(0, 1, (batch_size, 120)), 
                                        sampled_labels1], [positive_y, sampled_labels1])

# save classifier test AUC during training
AUC_ROC_Classifier = pd.DataFrame({ 'AUC_ROC': AUC_ROC[1:]})
AUC_ROC_Classifier.to_csv(str(dir) + '/evaluation/' + 'Classifier_AUC' + '.csv')
# Save no of training and test samples, and Best AUC in a text file
text_file = open(str(dir) + '/evaluation/'+ 'Performance_AUC', "w")
text_file.writelines(["Training_samples: ", str(X_train.shape[0]), 
                      "\nTest_samples: ", str(X_test.shape[0]), "\nBest_AUC: ", str(max(AUC_ROC))])
text_file.close()
# save trained CC-GAN models, generated target & nontarget for best classifier test AUC during training
save_CC_GAN(generator1, discriminator1, dir)
# plot & save classifier training AUC on test data curve
plot_AUC(AUC_ROC, dir)