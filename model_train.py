import pickle as pkl
import math
import glob
import random
from collections import Counter

import h5py

from tqdm import tqdm

import numpy as np
np.random.seed(42)
import pandas as pd

from sklearn.preprocessing import normalize, scale
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.manifold import TSNE

from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Activation
from keras.layers.merge import concatenate
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dropout, Flatten
from keras.layers.pooling import MaxPooling2D
from keras.layers.noise import GaussianNoise
from keras.layers.recurrent import LSTM
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam, Nadam, Adadelta, RMSprop
from keras.regularizers import l1, l2
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras.utils import plot_model

import matplotlib.pyplot as plt
import matplotlib.figure as fig
import matplotlib.cm as cm

import tools
from tools import EarlyStoppingByAUC, ModelAUCCheckpoint, LrReducer
#import ipdb #ipdb.set_trace()

from keras import backend as K
K.set_image_dim_ordering('tf')

NB_EPOCH_MY_MODEL = 100
stored_data_path = './stored_data/'

def split_train_predict(patient_ID, n_cv=None, val_r=0.3, test_r=0.3, preictal_period_minutes=10, postictal_period_minutes=10, movie_second=30):
     #patient_data_metadata 0:y_raw, 1:ictal_begin_second, 2:ictal_end_second, 3:preictal_begin_second,
     #4:preictal_end_second, 5:interictal_begin_second, 6:interictal_end_second, 7:postictal_begin_second, 
     #8:postictal_end_second, 9:unrecorded_begin_second, 10:unrecorded_begin_second
     preictal_data_fname = stored_data_path + 'patient_ID_' + str(patient_ID) + '_preictal_' + str(preictal_period_minutes) + '_data.pkl'
     preictal_data_ti_fname = stored_data_path + 'patient_ID_' + str(patient_ID) + '_preictal_ti_' + str(preictal_period_minutes) + '_data.pkl'
     with open(preictal_data_fname, 'rb') as infile_preictal:
         preictal_movie = pkl.load(infile_preictal)
     with open(preictal_data_ti_fname, 'rb') as infile_preictal_ti:
         preictal_time_index = pkl.load(infile_preictal_ti)
     interictal_data_fname = stored_data_path + 'patient_ID_' + str(patient_ID) + '_interictal_' + str(preictal_period_minutes) + '_data.pkl'
     interictal_data_ti_fname = stored_data_path + 'patient_ID_' + str(patient_ID) + '_interictal_ti_' + str(preictal_period_minutes) + '_data.pkl'
     with open(interictal_data_fname, 'rb') as infile_interictal:
         interictal_movie = pkl.load(infile_interictal)
     with open(interictal_data_ti_fname, 'rb') as infile_interictal_ti:
         interictal_time_index = pkl.load(infile_interictal_ti)
     pkl_fname = 'patient_ID_' + str(patient_ID) + '_metadata_pre_' + str(preictal_period_minutes) + '_post_' + str(postictal_period_minutes) + '.pkl'
     with open(stored_data_path + pkl_fname, 'rb') as infile:
         patient_metadata = pkl.load(infile)
     preictal_class = [1] * len(preictal_time_index)
     interictal_class = [0] * len(interictal_time_index)    
     total_class = preictal_class + interictal_class
     total_time_index = preictal_time_index + interictal_time_index
     total_movie = preictal_movie + interictal_movie
     n_test_ictal = int(len(patient_metadata[1]) * test_r)
     assert n_test_ictal != 0
     n_train_ictal = len(patient_metadata[1]) - n_test_ictal
     assert n_train_ictal != 0
     n_validation_ictal = round(n_train_ictal * val_r)
     n_subtrain_ictal = n_train_ictal - n_validation_ictal
     print('training ictal count: %d' %(int(n_subtrain_ictal)))
     print('validation ictal count: %d' %(int(n_validation_ictal)))
     print('testing ictal count: %d' %(int(n_test_ictal)))
     test_split_time_index = patient_metadata[2][n_train_ictal-1]
     validation_split_time_index = patient_metadata[2][n_subtrain_ictal-1]
     print('validation_split_time_index: %d' %(int(validation_split_time_index)))    
     print('test_split_time_index: %d' %(int(test_split_time_index)))
     del patient_metadata
     subtrain_x, subtrain_y = [], []
     validation_x, validation_y = [], []
     test_x, test_y, test_time_index, stats = [], [], [], []
     subtrain_time_index, validation_time_index = [], []
     for t, c, movie in zip(total_time_index, total_class, total_movie):
         if t < validation_split_time_index:
             subtrain_x.append(movie)
             subtrain_y.append(c)
             subtrain_time_index.append(t)
         elif t >= validation_split_time_index and t < test_split_time_index:
             validation_x.append(movie)
             validation_y.append(c)
             validation_time_index.append(t)
         else:
             test_x.append(movie)
             test_y.append(c)
             test_time_index.append(t)
     subtrain_y = np.array(subtrain_y)
     validation_y = np.array(validation_y)
     test_y = np.array(test_y)    
     subtrain_time_index = np.array(subtrain_time_index)
     validation_time_index = np.array(validation_time_index)
     test_time_index = np.array(test_time_index)
     subtrain_y = subtrain_y.reshape((subtrain_y.shape[0], 1))
     validation_y = validation_y.reshape((validation_y.shape[0], 1))
     test_y = test_y.reshape((test_y.shape[0], 1))    
     subtrain_time_index = subtrain_time_index.reshape((subtrain_time_index.shape[0], 1))
     validation_time_index = validation_time_index.reshape((validation_time_index.shape[0], 1))
     test_time_index = test_time_index.reshape((test_time_index.shape[0], 1))
     np.savetxt(str(patient_ID) + '_subtrain_time_index.csv', np.concatenate((subtrain_time_index, subtrain_y), axis=1), delimiter=',')
     np.savetxt(str(patient_ID) + '_validation_time_index.csv', np.concatenate((validation_time_index, validation_y), axis=1), delimiter=',')
     np.savetxt(str(patient_ID) + '_test_time_index.csv', np.concatenate((test_time_index, test_y), axis=1), delimiter=',')                  
     print('total train sample: %d' %(len(subtrain_y)))
     print('total train preictal sample: %d' %(np.count_nonzero(subtrain_y)))
     print('total train interictal sample: %d' %(len(subtrain_y) - np.count_nonzero(subtrain_y)))
     print('total validation sample: %d' %(len(validation_y)))
     print('total validation preictal sample: %d' %(np.count_nonzero(validation_y)))
     print('total validation interictal sample: %d' %(len(validation_y) - np.count_nonzero(validation_y)))
     print('total test sample: %d' %(len(test_y)))
     print('total test preictal sample: %d' %(np.count_nonzero(test_y)))
     print('total test interictal sample: %d' %(len(test_y) - np.count_nonzero(test_y)))
     print('testing interval: %d' %((max(total_time_index) - test_split_time_index) / 3600))
     subtrain_x = np.array(subtrain_x)
     subtrain_y = np.array(subtrain_y)
     validation_x = np.array(validation_x)
     validation_y = np.array(validation_y)
     test_x = np.array(test_x)
     test_y = np.array(test_y)
     test_time_index = np.array(test_time_index)
     return subtrain_x, subtrain_y, validation_x, validation_y, test_x, test_y, test_time_index, test_split_time_index

def my_model(input_dim):
    image_input = Input(shape=input_dim)
    encoded_video = Conv2D(filters=16, kernel_size=(5, 5), padding='same', use_bias=False)(image_input)
    encoded_video = Activation('relu')(encoded_video)
    encoded_video = MaxPooling2D(pool_size=(2, 2), padding='same')(encoded_video)
    encoded_video = Conv2D(filters=32, kernel_size=(2, 2), padding='same', use_bias=False)(encoded_video)
    encoded_video = Activation('relu')(encoded_video)
    encoded_video = MaxPooling2D(pool_size=(2, 2), padding='same')(encoded_video)
    encoded_video = Conv2D(filters=64, kernel_size=(2, 2), padding='same', use_bias=False)(encoded_video)
    encoded_video = Activation('relu')(encoded_video)
    output = Flatten()(encoded_video)
    output = Dense(128, kernel_initializer='he_normal', kernel_regularizer=l2(0.02), bias_regularizer=l2(0.02))(output)
    ourput = Dropout(0.5)(output)
    output = Dense(1, kernel_initializer='he_normal')(output)
    output = Activation('sigmoid')(output)
    model = Model(inputs=image_input, outputs=output)
    optimizer = RMSprop(lr=0.0005, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=optimizer, loss='binary_crossentropy')
    print(model.summary())
    return model

def train_my_model(patient_ID, preictal_period_minutes, index):
    print('patient_ID: %s, preictal_period_minutes: %d' %(patient_ID, preictal_period_minutes))
    split_train_test_xy_fname = 'split_train_test_xy/patient_ID_' + str(patient_ID) + '_split_train_test_xy_' + str(index+2) + '.pkl'
    with open(stored_data_path + split_train_test_xy_fname, 'rb') as infile:
        split_train_test_xy = pkl.load(infile)
    subtrain_x = split_train_test_xy[0]
    subtrain_y = split_train_test_xy[1]
    validation_x = split_train_test_xy[2]
    validation_y = split_train_test_xy[3]
    test_x = split_train_test_xy[4]
    test_y = split_train_test_xy[5]

    input_dim = subtrain_x.shape[1:]
    class_count = Counter((subtrain_y.flatten()).tolist())
    class_weight = {}
    for c in class_count:
        class_weight[c] = len(subtrain_y) / class_count[c]
    print('class_weight: {}'.format(class_weight))
    model = my_model(input_dim)
    model_fname = 'stored_model/PID_' + str(patient_ID) + '_p_' + str(preictal_period_minutes) + \
        '_index_' + str(index+2) + '_weights.hdf5'
    early_stopping = utils.EarlyStoppingByAUC(validation_data=(validation_x, validation_y), patience=10, verbose=1)
    model_auc_cp = utils.ModelAUCCheckpoint(model_fname, validation_data=(validation_x, validation_y), \
        verbose=1, save_weights_only=False)
    ReduceLR = utils.LrReducer(validation_data=(validation_x, validation_y), patience=5, verbose=1)
    callbacks = [
        early_stopping,
        ReduceLR,
        model_auc_cp
    ]
    model.fit(
        subtrain_x, subtrain_y,
        epochs=NB_EPOCH_MY_MODEL,
        batch_size=64,
        validation_data=(validation_x, validation_y),
        #class_weight=class_weight, 
        verbose=1,
        callbacks=callbacks 
    )

    model_files = sorted(glob.glob('stored_model/*.hdf5'))
    for model_file in model_files:
        with h5py.File(model_file, 'a') as f:
            if 'optimizer_weights' in f.keys():
                del f['optimizer_weights']

    model = load_model(model_fname)
    val_predict_probas = model.predict(validation_x)
    fprs, tprs, thesholds = roc_curve(validation_y, val_predict_probas)
    best_thes = 1.0
    best_tpr_fpr = 0.0
    for fpr, tpr, theshold in zip(fprs, tprs, thesholds):
        if tpr - fpr > best_tpr_fpr:
            best_thes = theshold
            best_tpr_fpr = tpr - fpr
    test_predict_probas = model.predict(test_x)
    test_predict_labels = (test_predict_probas > best_thes).astype('int32')
    test_accuracy = utils.classes_prediction_accuracy(test_predict_labels, test_y)  
    test_score = roc_auc_score(test_y, test_predict_probas, average='weighted')
    print('test ROC AUC score is: %f' %(test_score))
    csv_fname = 'prediction_results.csv'
    with open(csv_fname, 'a') as csv_file:
        csv_file.write('\n')
        csv_file.write('patient_ID, preictal_period_minutes, test_segment, test_thes,')
        csv_file.write('interictal_accuracy, interictal_error, preictal_accuracy, preictal_error, test ROC_AUC score\n')
        csv_file.write('%d, %d, %d, %.3f,' %(int(patient_ID), preictal_period_minutes, (index+2), best_thes))
        for i in range(len(test_accuracy)):
            csv_file.write('%.2f,' %(test_accuracy[i]))
        csv_file.write('%.3f,' %(test_score))
    del subtrain_x, validation_x, test_x
    return best_thes

def false_alarm_test(patient_ID, preictal_period_minutes, index, theshold):
    #patient_data_metadata 0:y_raw, 1:ictal_begin_second, 2:ictal_end_second, 3:preictal_begin_second,
    #4:preictal_end_second, 5:interictal_begin_second, 6:interictal_end_second, 7:postictal_begin_second, 
    #8:postictal_end_second, 9:unrecorded_begin_second, 10:unrecorded_begin_second
    print('patient_ID %s false alarm test for %d-min preictal period ... ' %(str(patient_ID), preictal_period_minutes))
    pkl_fname = 'patient_ID_' + str(patient_ID) + '_metadata_pre_' + str(preictal_period_minutes) + '_post_5.pkl'
    with open(stored_data_path + pkl_fname, 'rb') as infile:
        patient_metadata = pkl.load(infile)
    y_chain = patient_metadata[0]
    ictal_end_second = patient_metadata[2]
    ictal_end_second = [int(0)] + ictal_end_second + [len(patient_metadata[0])]
    test_begin_second = ictal_end_second[index+2]
    test_end_second = ictal_end_second[index+3]
    x_pkl_fname = 'patient_ID_' + str(patient_ID) + '_x.pkl'
    with open(stored_data_path + x_pkl_fname, 'rb') as infile:
        x_chain = pkl.load(infile)
    x_chain_test = x_chain[test_begin_second:test_end_second]
    y_chain_test = y_chain[test_begin_second:test_end_second]
    test_seconds = y_chain_test.shape[0]
    test_hours = test_seconds / 3600
    del x_chain, y_chain
    x_chain_test = x_chain_test[:(x_chain_test.shape[0]//150)*150]
    y_chain_test = y_chain_test[:(y_chain_test.shape[0]//150)*150]
    x_test = []
    print('data normalization for data... ')
    for i in tqdm(range(x_chain_test.shape[0] - 30)):
        x_test.append(utils.data_normalization(x_chain_test[i:i+30]))
    x_test = np.array(x_test)
    print(x_test.shape)
    del x_chain_test
    y_test = y_chain_test[30:]
    y_test = np.array(y_test)
    del y_chain_test

    model_fname = 'stored_model/PID_' + str(patient_ID) + '_p_' + str(preictal_period_minutes) + \
        '_index_' + str(index+2) + '_weights.hdf5'
    model = load_model(model_fname)

    test_proba_predict = model.predict([x_test, x_bv_test, x_bh_test], verbose=2)
    test_predict_labels = (test_proba_predict > theshold).astype('int32')

    confusion_matrix_predict = np.zeros((2,2))
    for i in range(y_test.shape[0]):
        if np.count_nonzero(x_test[i]) >= 2400:
            if y_test[i] == 1: # preictal
                if test_predict_labels[i] == 0: # predict interictal
                    confusion_matrix_predict[0][1] += 1 # predict: miss
                else: # predict preictal
                    confusion_matrix_predict[0][0] += 1 # predict: true positive
            elif y_test[i] == 2: # interictal
                if test_predict_labels[i] == 0: # predict interictal
                    confusion_matrix_predict[1][1] += 1 # predict: true negative
                else: # predict preictal
                    confusion_matrix_predict[1][0] += 1 # predict: false alarm

    sensitivity_predict = confusion_matrix_predict[0][0] / (confusion_matrix_predict[0][0] + confusion_matrix_predict[0][1])
    specificity_predict = confusion_matrix_predict[1][1] / (confusion_matrix_predict[1][0] + confusion_matrix_predict[1][1])
    false_alarm_predict = confusion_matrix_predict[1][0]
    miss_predict = confusion_matrix_predict[0][1]

    with open('false_alarm_test.csv', 'a') as csv_file:
        csv_file.write('\n')
        csv_file.write('patient_ID, preictal_period_minutes, test_hours, test_segment, test_thes,')
        csv_file.write('sensitivity_predict, specificity_predict, false_alarm_predict, miss_predict\n')
        csv_file.write('%d, %d, %f, %d, %.3f,' %(int(patient_ID), preictal_period_minutes, test_hours, (index+2), theshold))
        csv_file.write('%f, %f, %d, %d\n' %(sensitivity_predict, specificity_predict, false_alarm_predict, miss_predict))
    del x_test, x_bv_test, x_bh_test

def patient_train_test(patient_ID, preictal_period_minutes):
    pkl_fname = 'patient_ID_' + str(patient_ID) + '_metadata_pre_' + str(preictal_period_minutes) + '_post_5.pkl'
    with open(stored_data_path + pkl_fname, 'rb') as infile:
        patient_metadata = pkl.load(infile)
    ictal_end_second = patient_metadata[2]
    ictal_end_second = [int(0)] + ictal_end_second + [len(patient_metadata[0])]
    for index in range(len(ictal_end_second)-4):
        split_train_test_xy_fname = 'split_train_test_xy/patient_ID_' + str(patient_ID) + '_split_train_test_xy_' + str(index+2) + '.pkl'
        with open(stored_data_path + split_train_test_xy_fname, 'rb') as infile:
            split_train_test_xy = pkl.load(infile)
        if np.count_nonzero(split_train_test_xy[1] == 0) < 100 or np.count_nonzero(split_train_test_xy[1] == 1) < 100 \
            or np.count_nonzero(split_train_test_xy[3] == 0) < 100 or np.count_nonzero(split_train_test_xy[3] == 1) < 100 \
            or np.count_nonzero(split_train_test_xy[5] == 0) < 100 or np.count_nonzero(split_train_test_xy[5] == 1) < 100:
            continue
        del split_train_test_xy
        thes = train_my_model(patient_ID, preictal_period_minutes, index)
        false_alarm_test(patient_ID, preictal_period_minutes, index, thes)

def main():
    #n_ictals: 102:11, 7302:8, 8902:5, 11002:8, 16202:8, 21602:11, 21902:6, 22602:21, 23902:5, 26102:8, 30002:14, 30802:9,
    #32502:8, 32702:6, 45402:5, 46702:6, 55202:9, 56402:7, 58602:22, 59102:7, 75202:9, 85202:10, 92102:7, 93902:9,
    #96002:9, 103002:8, 109502:10, 111902:8, 114902:12      
    patients_ID_unrepeated = ['102', '7302', '8902', '11002', '16202', '21602', '21902', '22602', '23902', '26102', '30002', '30802', \
        '32502', '32702', '45402', '46702', '55202', '56402', '58602', '59102', '75202', '79502', '85202', '92102', '93902', '96002', \
            '103002', '109502', '111902', '114902']
    patients_ID_unrepeated = ['11002', '16202', '26102', '30802', '32702', '45402', '55202']
    for patient_ID in patients_ID_unrepeated:
        split_train_predict(patient_ID)
        #patient_train_test(patient_ID, preictal_period_minutes=10)


if __name__ == '__main__':
    main()
