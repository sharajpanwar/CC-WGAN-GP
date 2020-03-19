
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"]= '0'

def learning_curve(history, dir, d):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(dir + '/'+ str(d) +'.png')
    # plt.show()
