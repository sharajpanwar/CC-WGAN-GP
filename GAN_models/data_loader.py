
import keras
from random import shuffle
from sklearn.model_selection import train_test_split
import os
os.environ["CUDA_VISIBLE_DEVICES"]= '0'
Channels, time_step = 64, 64

def data_import(X, Y):
    ind_list = [i for i in range(X.shape[0])]; shuffle(ind_list)
    X = X[ind_list, :,:];    Y = Y[ind_list]
    #keras defaul settings: "image_data_format": "channels_last"
    X = X.reshape(X.shape[0], Channels, time_step, 1) #reshape the data
    X = X.astype('float32'); Y = Y.astype('int32')
    return X, Y

def data_import_split(X, Y):
    #keras defaul settings: "image_data_format": "channels_last"
    X = X.reshape(X.shape[0], Channels, time_step, 1) #reshape the data
    X = X.astype('float32'); Y = Y.astype('int32')
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test

def data_import_ch1(X, Y):
    ind_list = [i for i in range(X.shape[0])]; shuffle(ind_list)
    X = X[ind_list, :,:];    Y = Y[ind_list]
    #keras defaul settings: "image_data_format": "channels_last"
    X = X.reshape(X.shape[0], 1, time_step, 1) #reshape the data
    X = X.astype('float32'); Y = Y.astype('int32')
    return X, Y


