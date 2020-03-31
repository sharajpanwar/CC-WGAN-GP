# Author: Sharaj Panwar, MSEE; Research Fellow at Brain Computer Interface Lab./& Open Cloud Institute at UTSA, San Antonio, Texas
# this module created for the experiments performed in https://arxiv.org/ftp/arxiv/papers/1911/1911.04379.pdf
import matplotlib.pyplot as plt
import os
import numpy as np
import scipy.io
os.environ["CUDA_VISIBLE_DEVICES"]= '0'

# generate vanilla eeg during training
def generate_eeg(generator_model, dir, epoch, freq):

    if epoch%freq==0:
        eeg_gen = generator_model.predict(np.random.normal(0, 1, (100, 120)))
        np.save(str(dir) + '/eeg_generated/' + str(epoch) + '.npy', eeg_gen) #comment this, if you don't want to save samples
        # generator_model.save(str(dir) + '/saved_model/' + 'generator_' + str(epoch) + '.h5')
        # uncomment if you want to save model

# generate conditional eeg during training (used in one channel GAN)
def generate_condi_eeg(generator_model, dir, epoch, freq):
    
    if epoch%freq==0:
        # randomly input to the generator
        noise = np.random.normal(0, 1, (100, 120)).astype(np.float32)
        nontarget_labels = np.random.randint(0, 1, 100).reshape(-1, 1)
        target_labels = np.random.randint(1, 2, 100).reshape(-1, 1)
        # generate target & non-target samples
        generated_target = generator_model.predict([noise, target_labels.astype('float32')])
        generated_nontarget = generator_model.predict([noise, nontarget_labels.astype('float32')])
        # comment below, if you don't want to save samples
        np.save(str(dir) + '/eeg_generated/target_' + str(epoch) + '.npy', generated_target)
        np.save(str(dir) + '/eeg_generated/nontarget_' + str(epoch) + '.npy', generated_nontarget)
        # comment below, if you don't want to save model
        # generator_model.save(str(dir) + '/saved_model/' + 'generator_' + str(epoch) + '.h5')

# save trained CC-GAN models, generated target & nontarget for best classifier AUC
def save_CC_GAN(generator_model, discriminator_model, dir):
        # randomly input to the generator
        noise = np.random.normal(0, 1, (784, 120)).astype(np.float32)
        nontarget_labels = np.random.randint(0, 1, 784).reshape(-1, 1)
        target_labels = np.random.randint(1, 2, 784).reshape(-1, 1)
        # generate target & non-target samples
        generated_target = generator_model.predict([noise, target_labels.astype('float32')])
        generated_nontarget = generator_model.predict([noise, nontarget_labels.astype('float32')])
        # Save target and nontarget samples as numpy data
        np.save(str(dir) + '/saved_model/target.npy', generated_target)
        np.save(str(dir) + '/saved_model/nontarget.npy', generated_nontarget)
        # Save Generator and Discriminator models
        generator_model.save(str(dir) + '/saved_model/' + 'CC_GAN_generator.h5')
        discriminator_model.save(str(dir) + '/saved_model/' + 'CC_GAN_discriminator.h5')

# classifier test AUC during training
def plot_AUC(AUC, dir):
    plt.plot(AUC, label="Classifier_AUC")
    plt.title('Classifier Performace')
    plt.savefig(str(dir) + '/evaluation/' + 'Classifier_AUC.png')
    plt.legend()
    plt.show()

# if you want to save weights and model architecture separate
def save(model, dir, model_name):
    model_path = str(dir) + '/saved_model/' + str(model_name) +'.json'
    weights_path = str(dir) + '/saved_model/' + str(model_name)+'_weights.h5'
    json_string = model.to_json()

    open(model_path, 'w').write(json_string)
    model.save_weights(weights_path)

# WGAN loss plots
def plot_losses(discrminator_loss, generator_loss, dir):
    plt.plot([i for i in range(discrminator_loss.shape[0])], 1 - np.array(discrminator_loss), label="D_loss")
    plt.plot(np.arange(0, discrminator_loss.shape[0], (discrminator_loss.shape[0]/generator_loss.shape[0])),
             1 - np.array(generator_loss), label="G_loss")
    plt.title('Loss-curve')
    plt.legend()
    plt.savefig(dir + '/evaluation/'+'loss_curve.png')
    plt.show()