
import keras
from keras.callbacks import ModelCheckpoint
from EEGModels import EEGNet
import numpy as np
from sklearn.metrics import roc_auc_score
from plot_learning_curve import learning_curve
from data_import import data_import
import os
from numpy import *
img_rows, img_cols = 64, 64
import seaborn as sns;
import tensorflow as tf
from keras import backend as k

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
img_rows, img_cols = 64, 64

# For cross session performance evaluation=> four sessions out of five will be treated as training data and reaming one as test data
# This script will iterate all the subjects and report AUC when sessions 1 to 4 are treated as training set and session 5 as test set, for all subjects
# 4 more similar scripts can be created considering training set containing other group of four sessions and test as remaining session.
# For example Training on -: session 1,2,4,5, and test on session 3

model = EEGNet() #obtained from the EEGNet models module
print (model.summary())
# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

dir = '/home/guest/PycharmProjects/sharaj_works/input_data/rsvp_session_wise/'
# Directory where all the data is kept in folders named with subjects e.g. dir/S1/T_s1_r1, T1_s1_r1 stands for target subject 1 session 1

dir2 = '/home/guest/PycharmProjects/sharaj_works/NSRE/rsvp_gen_results/subject_wise/'
# dir2 is the main output directory to store results

list = [1, 3, 4, 5, 6, 7, 8, 10] # 8 selected subjects (To avoid confusion in the article, they are considered as s1 to s8, respectively.

# let's iterate ove all subjects mentioned in list
for i in range(len(list)):
    fold=list[i]
    Run_tag = 'EEGNet_te_on_sess_5_subject_' + str(fold) # Run tag will dynamically create the sub-directories to store results
    print(Run_tag)
    # output_dir & output_dir1 will be created for each subject and test session results
    output_dir = dir2 + 's_rough_' +str(fold)+'/'
    output_dir1 = os.path.join(output_dir, Run_tag)
    output_dir1 =output_dir1 +'/'

    if not os.path.exists(output_dir1):
        os.mkdir(output_dir1)

    # uploading target & nonTarget session data for a subject
    a1 = np.load(dir + 's' + str(fold) + '/T_s' + str(fold) + '_r1.npy')
    a2 = np.load(dir + 's' + str(fold) + '/T_s' + str(fold) + '_r2.npy')
    a3 = np.load(dir + 's' + str(fold) + '/T_s' + str(fold) + '_r3.npy')
    a4 = np.load(dir + 's' + str(fold) + '/T_s' + str(fold) + '_r4.npy')
    a5 = np.load(dir + 's' + str(fold) + '/T_s' + str(fold) + '_r5.npy')

    b1 = np.load(dir + 's' + str(fold) + '/nT_s' + str(fold) + '_r1.npy')
    b2 = np.load(dir + 's' + str(fold) + '/nT_s' + str(fold) + '_r2.npy')
    b3 = np.load(dir + 's' + str(fold) + '/nT_s' + str(fold) + '_r3.npy')
    b4 = np.load(dir + 's' + str(fold) + '/nT_s' + str(fold) + '_r4.npy')
    b5 = np.load(dir + 's' + str(fold) + '/nT_s' + str(fold) + '_r5.npy')

    # creating training data
    T_tr = np.concatenate((a1, a2, a3, a4));    nT_tr = np.concatenate((b1, b2, b3, b4))
    y_T_tr = np.ones(T_tr.shape[0]);    y_nT_tr = np.zeros(nT_tr.shape[0])
    X_train = np.concatenate((T_tr, nT_tr), axis=0);    y_train = np.concatenate((y_T_tr, y_nT_tr), axis=0)

    # creating test data
    T_te = a5;    nT_te = b5
    y_T_te = np.ones(T_te.shape[0]);    y_nT_te = np.zeros(nT_te.shape[0])
    X_test = np.concatenate((T_te, nT_te), axis=0);    y_test = np.concatenate((y_T_te, y_nT_te), axis=0)

    # Shuffle and reshape the data to fit in keras model
    X_train, y_train = data_import(X_train, y_train)
    X_test, y_test = data_import(X_test, y_test)


    X_train = X_train.astype('float32');    y_train = y_train.astype('int32')
    X_test = X_test.astype('float32');    y_test = y_test.astype('int32')

    # fit the model
    history = model.fit(X_train, y_train, batch_size=32, epochs=2,
                    verbose=2, validation_split=0.10)
    # Performance evaluation

    target = model.predict(X_test, batch_size=32)
    auc = roc_auc_score(y_test, target)
    print("auc_roc:", auc)
    # plot learning curve
    learning_curve(history, output_dir1, str(Run_tag))

    # Save no of training and test samples, and AUC in a text file
    text_file = open(output_dir1 + 'AUC_'+str(Run_tag), "w")
    text_file.writelines(
        ["Training_samples: ",str(X_train.shape[0]),"\nTest_samples: ",str(X_test.shape[0]),"\nAUC: ",str(auc)])
    text_file.close()

    # save the model
    # serialize model to JSON
    model_json = model.to_json()
    with open(str(output_dir1) +  str(Run_tag)+'_model.json', "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(str(output_dir1) +  str(Run_tag)+'_model_weight.h5')
    print("Saved EEGNet_Real_baseline to disk")