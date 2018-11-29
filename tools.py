import os
import struct
import itertools
import numpy as np
import pickle as pkl
import scipy.io as sio
import matplotlib.pyplot as plt
from glob import glob
from scipy import interp
from pathlib import Path
from keras.callbacks import Callback
from keras.utils.np_utils import to_categorical
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, confusion_matrix, average_precision_score
import keras.backend as K
stored_data_path = './stored_data/'

def spectral_power(x):
    power_spectrum = np.fft.rfft(x)
    power_spectrum_abs = np.absolute(power_spectrum)
    power_spectrum_abs = power_spectrum_abs[1:51]
    return power_spectrum_abs 

def read_binary(filepath, n_channels, save_path='stored_fig/'):
    #elec_dict = dict({0:'FP1', 1:'FP2', 2:'F3', 3:'F4', 4:'C3', 5:'C4', 6:'P3', 7:'P4', 8:'O1', 9:'O2', 10:'F7', 11:'F8', \
    #    12:'T3', 13:'T4', 14:'T5', 15:'T6', 16:'FZ', 17:'CZ', 18:'PZ'})
    data = np.fromfile(filepath, dtype=np.int16)
    n_sample = data.shape[0] // n_channels
    data_reshaped = np.reshape(data, (n_sample, n_channels))
    data_19ch = data_reshaped[:, :19]    
    return data_19ch

def convert_to_spectral_data(data_time_domain, fs=256):
    data_spectral = np.zeros((data_time_domain.shape[0]//fs, 50, data_time_domain.shape[1]))
    for channelIndex in range(data_spectral.shape[2]):
        for secondIndex in range(data_time_domain.shape[0]//fs):
            dataFrag = data_time_domain[secondIndex:secondIndex+fs, channelIndex]
            if dataFrag.shape != (256, ):
                print(dataFrag.shape)
                raise ValueError('Wrong EEG shape...')
            data_spectral[secondIndex, :, channelIndex] = spectral_power(dataFrag)
    return data_spectral

def process_raw_data(filepath, n_channels):
    raw_data = read_binary(filepath, n_channels)
    eeg_image = convert_to_spectral_data(raw_data)  
    return eeg_image

def get_patients_dirs(mother_dir='/media/sslu-15/DATA/Research/Seizure_Data/surf30/'):
    #path under rec_
    #    for filename in filenames:
    #        print(os.path.join(dirpath, filename))
    files = [x[0] for x in os.walk(mother_dir)]
    files_rec = []
    for filename in files:
        if 'rec' in filename:
            files_rec.append(filename + '/')
    return files_rec

def get_patient_files(path, data_only=True):
    all_fpath = []
    if data_only == True:
        path = path + '*.data' #path/*.data include all data
    else:
        path = path + '*'
    for fpath in glob(path): # record all data detail path
        all_fpath.append(fpath)
    return len(all_fpath), all_fpath # number of data, all data name

def split_large_data(patient_ID, data, numberPieces=50):
    dataLen = data.shape[0]
    smallDataLen = dataLen // numberPieces
    for index in range(numberPieces):
        x_pkl_fname = stored_data_path + 'patient_ID_' + str(patient_ID) + '_x_' + str(index) + '.pkl'
        if index == numberPieces - 1:
            smallData = data[index*smallDataLen:]
        else:    
            smallData = data[index*smallDataLen:(index+1)*smallDataLen]
        with open(x_pkl_fname, 'wb') as infile:
            pkl.dump(smallData, infile)

def merge_small_data(patient_ID, numberPieces=50):
    data = []
    for index in range(numberPieces):
        x_pkl_fname = stored_data_path + 'patient_ID_' + str(patient_ID) + '_x_' + str(index) + '.pkl'        
        with open(x_pkl_fname, 'rb') as infile:
            smallData = pkl.load(infile)
        if index == 0:
            data.append(smallData)
            data = np.array(data[0])
        else:
            data = np.concatenate((data, smallData), axis=0)
    return data

def classes_prediction_accuracy(test_predict_classes, test_y):
    classes_unique = list(set(test_y))
    classes_count = []
    accuracy = []
    for class_unique in classes_unique:
        class_count = test_y.tolist().count(class_unique)
        classes_count.append(class_count)
    classes_stats = []
    for class_unique in classes_unique:
        correct = 0
        wrong = 0
        for truth, predict in zip(test_y, test_predict_classes):
            if truth == class_unique:
                if truth == predict:
                    correct += 1
                else:
                    wrong += 1
        classes_stats.append(correct)
        classes_stats.append(wrong)
    for index, class_unique in enumerate(classes_unique):
        print('class %d...' %(class_unique))
        print('correct: %d, wrong: %d' %(classes_stats[2*index], classes_stats[2*index+1]))
        print('correct rate: %f, wrong rate: %f' %(classes_stats[2*index]/classes_count[index], classes_stats[2*index+1]/classes_count[index]))
        accuracy.append(float('{0:.2f}'.format(100*classes_stats[2*index]/classes_count[index])))
        accuracy.append(float('{0:.2f}'.format(100*classes_stats[2*index+1]/classes_count[index])))
    return accuracy

class ModelAUCCheckpoint(Callback):
    def __init__(self, filepath, validation_data, verbose=0, save_weights_only=False):
        super(ModelAUCCheckpoint, self).__init__()
        self.verbose = verbose
        self.filepath = filepath
        self.X_val, self.y_val = validation_data
        self.save_weights_only = save_weights_only
        self.best = 0.

    def on_epoch_end(self, epoch, logs={}):
        filepath = self.filepath.format(epoch=epoch, **logs)
        y_pred = self.model.predict(self.X_val, verbose=0).flatten()
        current = roc_auc_score(self.y_val, y_pred, average='weighted')
        print("\ninterval evaluation - epoch: {:d} - score: {:.6f}".format(epoch, current))

        if current > self.best:
            if self.verbose > 0:
                print('\nEpoch %05d: ROC AUC improved from %0.5f to %0.5f,'
                      ' saving model to %s'
                      % (epoch, self.best,
                         current, filepath))
            self.best = current
            if self.save_weights_only:
                self.model.save_weights(filepath, overwrite=True)
            else:
                self.model.save(filepath, overwrite=True)
        else:
            if self.verbose > 0:
                print('\nEpoch %05d: ROC AUC did not improve' %
                      (epoch))

class EarlyStoppingByAUC(Callback):
    def __init__(self, validation_data,  patience=0, verbose=0):
        super(EarlyStoppingByAUC, self).__init__()
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.X_val, self.y_val = validation_data

    def on_train_begin(self, logs={}):
        self.wait = 0       # Allow instances to be re-used
        self.best = 0.0

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.X_val, verbose=0).flatten()
        current = roc_auc_score(self.y_val, y_pred, average='weighted')

        if current > self.best:
            self.best = current
            self.wait = 0
        else:
            if self.wait > self.patience:
                if self.verbose > 0:
                    print('\nEpoch %05d: early stopping' % (epoch))
                self.model.stop_training = True
            self.wait += 1

class LrReducer(Callback):
    def __init__(self, validation_data, patience=10, reduce_rate=0.5, verbose=0):
        super(Callback, self).__init__()
        self.patience = patience
        self.reduce_rate = reduce_rate
        self.verbose = verbose
        self.wait = 0
        self.X_val, self.y_val = validation_data

    def on_train_begin(self, logs={}):
        self.wait = 0
        self.best = 0.0

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.X_val, verbose=0).flatten()
        current = roc_auc_score(self.y_val, y_pred, average='weighted')

        if current > self.best:
            self.best = current
            self.wait = 0
        else:
            if self.wait > self.patience:
                lr = K.get_value(self.model.optimizer.lr)
                K.set_value(self.model.optimizer.lr, lr*self.reduce_rate)
                #lr = self.model.optimizer.lr.get_value()
                #self.model.optimizer.lr.set_value(lr*self.reduce_rate)
                if self.verbose > 0:
                    print('\nReduce Learning Rate to %f' % (lr*self.reduce_rate))
            self.wait += 1
