
import keras
from random import shuffle
import os
os.environ["CUDA_VISIBLE_DEVICES"]= '0'
Channels, time_step = 64, 64

def data_import(X, Y):
    ind_list = [i for i in range(X.shape[0])]; shuffle(ind_list)
    X = X[ind_list, :,:];    Y = Y[ind_list]
    #keras defaul settings: "image_data_format": "channels_last"
    X = X.reshape(X.shape[0], Channels, time_step, 1) #reshape the data
    Y = keras.utils.to_categorical(Y, 2) #one hot encoding of labels
    return X, Y
