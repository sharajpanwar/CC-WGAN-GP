import keras
from keras.callbacks import ModelCheckpoint
from EEGModels import EEGNet
import numpy as np
from sklearn.metrics import roc_auc_score
from plot_learning_curve import learning_curve
from data_import import data_import
import os
import argparse
from numpy import *
import seaborn as sns;
import tensorflow as tf
from keras import backend as k

sns.set()
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# TensorFlow wizardry
config = tf.ConfigProto()
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.5
# Create a session with the above options specified.
k.tensorflow_backend.set_session(tf.Session(config=config))

# For cross session performance evaluation=> four sessions out of five will be treated as training data and remaining ones as test data for each subject
# This script requires two positional arguments one is subject under evaluation (out of 8) and second is test session (out of 5);
# For example "./$ python training_EEGNet_classifier 3 4" means we are creating the model on subject 3; Training on -: session 1,2,3,5 and test on session 4

parser = argparse.ArgumentParser()
parser.add_argument("sub", help="enter the subject") # subject number
parser.add_argument("sess", help="enter the test session") # test session
args = parser.parse_args()
print ("Evaluation on subject {}, Test session {}".format(args.sub, args.sess))

dir = '/home/guest/PycharmProjects/sharaj_works/input_data/rsvp_session_wise/'
# Directory where all the data is kept in folders named with subjects
# e.g. dir/S1/T_s1_r1, T1_s1_r1 stands for target subject 1 session 1

# dir2 = '/home/guest/PycharmProjects/sharaj_works/NSRE/rsvp_gen_results/subject_wise/'
# # dir2 is the main output directory to store results
sub = args.sub # parsing the subject No.
sess = args.sess # parsing the test session No.

# uploading target & nonTarget session data for the given subject
a1 = np.load(dir + 's' + str(sub) + '/T_s' + str(sub) + '_r1.npy')
a2 = np.load(dir + 's' + str(sub) + '/T_s' + str(sub) + '_r2.npy')
a3 = np.load(dir + 's' + str(sub) + '/T_s' + str(sub) + '_r3.npy')
a4 = np.load(dir + 's' + str(sub) + '/T_s' + str(sub) + '_r4.npy')
a5 = np.load(dir + 's' + str(sub) + '/T_s' + str(sub) + '_r5.npy')

b1 = np.load(dir + 's' + str(sub) + '/nT_s' + str(sub) + '_r1.npy')
b2 = np.load(dir + 's' + str(sub) + '/nT_s' + str(sub) + '_r2.npy')
b3 = np.load(dir + 's' + str(sub) + '/nT_s' + str(sub) + '_r3.npy')
b4 = np.load(dir + 's' + str(sub) + '/nT_s' + str(sub) + '_r4.npy')
b5 = np.load(dir + 's' + str(sub) + '/nT_s' + str(sub) + '_r5.npy')

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

X_train = X_train.astype('float32'); y_train = y_train.astype('int32')
X_test = X_test.astype('float32'); y_test = y_test.astype('int32')

# Now since we have uploaded the data, let's create directories to store the results
dir2 = '/home/guest/PycharmProjects/sharaj_works/NSRE/rsvp_gen_results/subject_wise/'
# dir2 is the main output directory to store results

# Run tag will dynamically create the sub-directories to store results
Run_tag = 'EEGNet_subject_'+ str(sub)+'_te_session_' + str(sess)
print(Run_tag)

# output_dir & output_dir1 will be created for each subject and test session results
output_dir = dir2 + 's_try_' + str(sub) + '/'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
output_dir1 = os.path.join(output_dir, Run_tag)

if not os.path.exists(output_dir1):
    os.mkdir(output_dir1)
  
# construct the model
img_rows, img_cols = 64, 64
model = EEGNet() #obtained from the EEGNet models module
print (model.summary())

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

# let's fit the model
history = model.fit(X_train, y_train, batch_size=32, epochs=2, verbose=2, validation_split=0.10)

# Performance evaluation
target = model.predict(X_test, batch_size=32)
auc = roc_auc_score(y_test, target)
print("auc_roc:", auc)

# plot learning curve
learning_curve(history, output_dir1, str(Run_tag))

# Save no of training and test samples, and AUC in a text file
text_file = open(output_dir1 + '/'+  'AUC_' + str(Run_tag), "w")
text_file.writelines(
    ["Training_samples: ", str(X_train.shape[0]), "\nTest_samples: ", str(X_test.shape[0]), "\nAUC: ", str(auc)])
text_file.close()

# save the model
# serialize model to JSON
model_json = model.to_json()
with open(str(output_dir1) + '/'+  str(Run_tag) + '_model.json', "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(str(output_dir1) + '/'+  str(Run_tag) + '_model_weight.h5')
print("Saved EEGNet baseline to disk")
