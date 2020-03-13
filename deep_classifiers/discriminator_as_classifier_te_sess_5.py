
import keras
from numpy import *
img_rows, img_cols = 64, 64
import os
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten, GaussianNoise, multiply, Embedding,Dropout,ZeroPadding2D
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from plot_learning_curve import learning_curve
from keras import backend as K
from random import shuffle
from sklearn.metrics import roc_auc_score
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
# construct the model

##############################################################################################################
# this script will run over all the subjects and report AUC results for training on session 1-4 & testing on 5th session of all subjects
# simillar scripts can be created for test on session 1-4.
def Discriminator_Classifier(): #Discriminator as Classifier, model architecture

    model = Sequential()

    model.add(GaussianNoise(0.05, input_shape=(64, 64, 1)))  # Add this layer to prevent D from overfitting!
    model.add(Convolution2D(16, (3, 3), padding='same', input_shape=(64, 64, 1)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Convolution2D(32, (3, 3), kernel_initializer='he_normal', padding='same', strides=[2, 2]))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Convolution2D(64, (3, 3), kernel_initializer='he_normal', padding='same', strides=[2, 2]))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Convolution2D(128, (3, 3), kernel_initializer='he_normal', padding='same', strides=[2, 2]))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(2, kernel_initializer='he_normal', activation="softmax"))

    print(model.summary())

    input_conv = Input(shape=(64, 64, 1))
    out_put = model(input_conv)

    return Model(input_conv, out_put)

model = Discriminator_Classifier()
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
    Run_tag = 'D_C_te_on_sess_5_subject_' + str(fold) # Run tag will dynamically create the sub-directories to store results
    print(Run_tag)
    # output_dir & output_dir1 will be created for each subject and test session results
    output_dir = dir2 + 's' + str(fold) + '/'
    output_dir1 = os.path.join(output_dir, Run_tag)
    output_dir1 =output_dir1 +'/'

    if not os.path.exists(output_dir1):
        os.mkdir(output_dir1)

    # uploading target & nonTarget session data for a subject
    a1 = np.load(dir + 's' + str(fold) + '/T_s'+ str(fold)+ '_r1.npy')
    a2 = np.load(dir + 's' + str(fold) + '/T_s'+ str(fold)+ '_r2.npy')
    a3 = np.load(dir + 's' + str(fold) + '/T_s'+ str(fold)+ '_r3.npy')
    a4 = np.load(dir + 's' + str(fold) + '/T_s'+ str(fold)+ '_r4.npy')
    a5 = np.load(dir + 's' + str(fold) + '/T_s'+ str(fold)+ '_r5.npy')

    b1 = np.load(dir + 's' + str(fold) + '/nT_s'+ str(fold)+ '_r1.npy')
    b2 = np.load(dir + 's' + str(fold) + '/nT_s'+ str(fold)+ '_r2.npy')
    b3 = np.load(dir + 's' + str(fold) + '/nT_s'+ str(fold)+ '_r3.npy')
    b4 = np.load(dir + 's' + str(fold) + '/nT_s'+ str(fold)+ '_r4.npy')
    b5 = np.load(dir + 's' + str(fold) + '/nT_s'+ str(fold)+ '_r5.npy')

    # creating training data
    T_tr = np.concatenate((a1, a2, a3, a4));    nT_tr = np.concatenate((b1, b2, b3, b4))
    y_T_tr = np.ones(T_tr.shape[0]);    y_nT_tr = np.zeros(nT_tr.shape[0])
    X_train = np.concatenate((T_tr, nT_tr), axis=0);    y_train = np.concatenate((y_T_tr, y_nT_tr), axis=0)

    # creating test data
    T_te = a5;     nT_te = b5
    y_T_te = np.ones(T_te.shape[0]);    y_nT_te = np.zeros(nT_te.shape[0])
    X_test = np.concatenate((T_te, nT_te), axis=0); y_test = np.concatenate((y_T_te, y_nT_te), axis=0)

    # shuffle training
    ind_list = [i for i in range(X_train.shape[0])];    shuffle(ind_list)
    X_train = X_train[ind_list, :];    y_train = y_train[ind_list]

    # shuffle test
    ind_list = [i for i in range(X_test.shape[0])];    shuffle(ind_list)
    X_test = X_test[ind_list, :, :];    y_test = y_test[ind_list,]

    X_train = X_train.astype('float32');    y_train = y_train.astype('int32')
    X_test = X_test.astype('float32');    y_test = y_test.astype('int32')
    y_train = keras.utils.to_categorical(y_train, 2)
    y_test = keras.utils.to_categorical(y_test, 2)

    if K.image_data_format() == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    # optionally we can also save the training and test data samples info
    text_file = open(str(output_dir1) + str(Run_tag) + "_training_samples", "w")
    statement = [str(X_train.shape[0])]
    text_file.writelines(statement)
    text_file.close()

    text_file1 = open(str(output_dir1) + str(Run_tag) + "_test_samples", "w")
    statement1 = [str(X_test.shape[0])]
    text_file1.writelines(statement1)
    text_file1.close()

    # let's fit the model
    history = model.fit(X_train, y_train, batch_size=32, epochs=50,
                    verbose=2, validation_split=0.10)
    # Performance evaluation

    target = model.predict(X_test, batch_size=32)
    auc = roc_auc_score(y_test, target)
    print("auc_roc:", auc)
    # plot learning curve
    learning_curve(history, output_dir1, str(Run_tag))
    #
    text_file = open(output_dir1 + 'AUC_'+str(Run_tag), "w")
    text_file.writelines([str(auc)])
    text_file.close()

    # save the model
    # serialize model to JSON
    model_json = model.to_json()
    with open(str(output_dir1) +  str(Run_tag)+'_model.json', "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(str(output_dir1) +  str(Run_tag)+'_model_weight.h5')
    print("Saved D_C_te_on_sess_5_baseline to disk")







