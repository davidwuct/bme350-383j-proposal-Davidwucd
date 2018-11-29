import os
import sys
import tools
import time
import numpy as np
import pandas as pd
import pickle as pkl
import scipy.io as sio
from tqdm import tqdm
from glob import glob
from itertools import *
from pathlib import Path
from datetime import datetime, timedelta
np.random.seed(42)

stored_data_path = './stored_data/'
data_path = '/media/sslu-15/DATA/Research/Seizure_Data/'

def read_seizure_list(filename):
    df_seizure_list = pd.read_csv(filename, sep=' |\t',  header=None, skiprows=4, engine='python')
    df_seizure_list.columns = ['str0', 'str1', 'str2', 'str3', 'str4', 'str5']
    df_seizure_list['onset_year'], df_seizure_list['onset_month'], df_seizure_list['onset_day'] = df_seizure_list['str0'].str.split('-').str
    df_seizure_list['onset_hour'], df_seizure_list['onset_minute'], df_seizure_list['onset_second'] = df_seizure_list['str1'].str.split(':').str
    df_seizure_list = df_seizure_list.apply(lambda x: pd.to_numeric(x, errors='ignore'))
    df_seizure_list['onset_second'] = df_seizure_list['onset_second'].astype(int)
    df_seizure_list['offset_year'], df_seizure_list['offset_month'], df_seizure_list['offset_day'] = df_seizure_list['str2'].str.split('-').str
    df_seizure_list['offset_hour'], df_seizure_list['offset_minute'], df_seizure_list['offset_second'] = df_seizure_list['str3'].str.split(':').str
    df_seizure_list = df_seizure_list.apply(lambda x: pd.to_numeric(x, errors='ignore'))
    df_seizure_list['offset_second'] = df_seizure_list['offset_second'].astype(int)
    df_seizure_list['onset_sample'] = (df_seizure_list['str4']//256*256).astype(int)
    df_seizure_list['offset_sample'] = (df_seizure_list['str5']//256*256).astype(int)
    df_seizure_list.drop(['str0', 'str1', 'str2', 'str3', 'str4', 'str5'], axis=1, inplace=True)
    df_seizure_list2 = df_seizure_list[['onset_year', 'onset_month', 'onset_day', 'onset_hour', 'onset_minute', 'onset_second', 'onset_sample', \
        'offset_year', 'offset_month', 'offset_day', 'offset_hour', 'offset_minute', 'offset_second', 'offset_sample']]   
    df_seizure_list2 = df_seizure_list2.astype(int)
    df_seizure_list2['onset_date'] = df_seizure_list2.apply(lambda row: datetime(row['onset_year'], row['onset_month'], row['onset_day'], row['onset_hour'], row['onset_minute'], row['onset_second']), axis=1)
    df_seizure_list2.drop(['onset_year', 'onset_month', 'onset_day', 'onset_hour', 'onset_minute', 'onset_second'], axis=1, inplace=True)
    df_seizure_list2['offset_date'] = df_seizure_list2.apply(lambda row: datetime(row['offset_year'], row['offset_month'], row['offset_day'],row['offset_hour'], row['offset_minute'], row['offset_second']), axis=1)
    df_seizure_list2.drop(['offset_year', 'offset_month', 'offset_day', 'offset_hour', 'offset_minute', 'offset_second'], axis=1, inplace=True)
    df_seizure_list2.to_csv('./stored_csv/seizure_info.csv', sep=',', index=False)
    return df_seizure_list2

def read_head(binary_files):#one patient ID's all Data
    df_head = pd.DataFrame(columns=('year', 'month', 'day', 'hour', 'minute', 'second', 'n_samples', 'file_name', 'patient_ID', 'relative_path', 'n_channels')) 
    count = 0
    for binary_file in binary_files:
        head_file = binary_file[:-5] + '.head'
        info = []
        with open(head_file, 'r') as infile:
            head = [next(infile) for x in range(2)]
            buf = []
            info = []
            # for line in head:
            #     for word in line.split('=')[1:]:
            #         buf += word.split()
            test_word = (head[0].split('=')[1:][0][:-5])#split time
            test_sample = (head[1].split('=')[1:][0][:-1])#split sample numbers
            test_time = (time.strptime(test_word, "%Y-%m-%d %H:%M:%S"))
            info = list(test_time[0:6])
            info.append(test_sample)
            # info += buf[0].split('-') + buf[1][:-4].split(':')
            # info.append(buf[2])
            info = [int(i) for i in info]
            info.append(binary_file.split('/')[10]) #file_name
            info.append(binary_file.split('/')[7][4:]) #patient_ID
            info.append(binary_file) #relative_path
        with open(head_file, 'r') as infile:
            lines = infile.readlines()
            info.append(lines[4].split('=')[1]) #n_channels
        df_head.loc[count] = info
        count +=1
    df_head['patient_ID'] = pd.to_numeric(df_head['patient_ID'], errors='ignore')
    df_head['n_channels'] = pd.to_numeric(df_head['n_channels'], errors='ignore')
    # df_head[['year', 'month', 'day', 'hour', 'minute', 'second', 'n_samples']] = \
    #     df_head[['year', 'month', 'day', 'hour', 'minute', 'second', 'n_samples']].astype(int)
    df_head.sort_values(by=['year', 'month', 'day', 'hour', 'minute', 'second'], inplace=True)
    df_head = df_head.reset_index(drop=True)
    df_head['begin_date'] = df_head.apply(lambda row: datetime(row['year'], row['month'], row['day'], row['hour'], row['minute'], row['second']), axis=1)
    df_head.drop(['year', 'month', 'day', 'hour', 'minute', 'second'], axis=1, inplace=True)
    df_head['record_seconds'] = df_head['n_samples'] // 256
    df_head['record_seconds'] = pd.to_timedelta(df_head['record_seconds'], unit='s')
    df_head['end_date'] = df_head['begin_date'] + df_head['record_seconds']
    df_head.drop(['record_seconds'], axis=1, inplace=True)
    return df_head    

def merge_df(head_files, seizure_list):
    df_head = read_head(head_files)
    df_seizure_list = read_seizure_list(seizure_list)
    df_head = df_head.assign(n_seizures=pd.Series(np.zeros(len(df_head['n_samples']), dtype=np.int)).values)
    df_head = df_head.assign(onset_date1=pd.Series(np.zeros(len(df_head['n_samples']), dtype=np.int)).values)
    df_head = df_head.assign(offset_date1=pd.Series(np.zeros(len(df_head['n_samples']), dtype=np.int)).values)
    df_head = df_head.assign(onset_sample1=pd.Series(np.zeros(len(df_head['n_samples']), dtype=np.int)).values)
    df_head = df_head.assign(offset_sample1=pd.Series(np.zeros(len(df_head['n_samples']), dtype=np.int)).values)
    df_head = df_head.assign(onset_date2=pd.Series(np.zeros(len(df_head['n_samples']), dtype=np.int)).values)
    df_head = df_head.assign(offset_date2=pd.Series(np.zeros(len(df_head['n_samples']), dtype=np.int)).values)
    df_head = df_head.assign(onset_sample2=pd.Series(np.zeros(len(df_head['n_samples']), dtype=np.int)).values)
    df_head = df_head.assign(offset_sample2=pd.Series(np.zeros(len(df_head['n_samples']), dtype=np.int)).values)
    df_head = df_head.assign(onset_date3=pd.Series(np.zeros(len(df_head['n_samples']), dtype=np.int)).values)
    df_head = df_head.assign(offset_date3=pd.Series(np.zeros(len(df_head['n_samples']), dtype=np.int)).values)
    df_head = df_head.assign(onset_sample3=pd.Series(np.zeros(len(df_head['n_samples']), dtype=np.int)).values)
    df_head = df_head.assign(offset_sample3=pd.Series(np.zeros(len(df_head['n_samples']), dtype=np.int)).values)
    df_head = df_head.assign(onset_date4=pd.Series(np.zeros(len(df_head['n_samples']), dtype=np.int)).values)
    df_head = df_head.assign(offset_date4=pd.Series(np.zeros(len(df_head['n_samples']), dtype=np.int)).values)
    df_head = df_head.assign(onset_sample4=pd.Series(np.zeros(len(df_head['n_samples']), dtype=np.int)).values)
    df_head = df_head.assign(offset_sample4=pd.Series(np.zeros(len(df_head['n_samples']), dtype=np.int)).values)
    df_head = df_head.assign(onset_date5=pd.Series(np.zeros(len(df_head['n_samples']), dtype=np.int)).values)
    df_head = df_head.assign(offset_date5=pd.Series(np.zeros(len(df_head['n_samples']), dtype=np.int)).values)
    df_head = df_head.assign(onset_sample5=pd.Series(np.zeros(len(df_head['n_samples']), dtype=np.int)).values)
    df_head = df_head.assign(offset_sample5=pd.Series(np.zeros(len(df_head['n_samples']), dtype=np.int)).values)
    for index_head, row_head in df_head.iterrows():
        seizures_count = 0  #determine seizure occur in which date set
        for index_seizure_list, row_seizure_list in df_seizure_list.iterrows():
            if row_seizure_list['onset_date'] > row_head['begin_date'] and row_seizure_list['onset_date'] < row_head['end_date']:
                seizures_count += 1
                if(df_head.iloc[index_head]['onset_date1'] == 0):
                    df_head.set_value(index_head, 'onset_date1', df_seizure_list.iloc[index_seizure_list]['onset_date'])
                    df_head.set_value(index_head, 'offset_date1', df_seizure_list.iloc[index_seizure_list]['offset_date'])
                    df_head.set_value(index_head, 'onset_sample1', df_seizure_list.iloc[index_seizure_list]['onset_sample'])
                    df_head.set_value(index_head, 'offset_sample1', df_seizure_list.iloc[index_seizure_list]['offset_sample'])
                elif(df_head.iloc[index_head]['onset_date2'] == 0):
                    df_head.set_value(index_head, 'onset_date2', df_seizure_list.iloc[index_seizure_list]['onset_date'])
                    df_head.set_value(index_head, 'offset_date2', df_seizure_list.iloc[index_seizure_list]['offset_date'])
                    df_head.set_value(index_head, 'onset_sample2', df_seizure_list.iloc[index_seizure_list]['onset_sample'])
                    df_head.set_value(index_head, 'offset_sample2', df_seizure_list.iloc[index_seizure_list]['offset_sample'])
                elif(df_head.iloc[index_head]['onset_date3'] == 0):
                    df_head.set_value(index_head, 'onset_date3', df_seizure_list.iloc[index_seizure_list]['onset_date'])
                    df_head.set_value(index_head, 'offset_date3', df_seizure_list.iloc[index_seizure_list]['offset_date'])
                    df_head.set_value(index_head, 'onset_sample3', df_seizure_list.iloc[index_seizure_list]['onset_sample'])
                    df_head.set_value(index_head, 'offset_sample3', df_seizure_list.iloc[index_seizure_list]['offset_sample'])
                elif(df_head.iloc[index_head]['onset_date4'] == 0):
                    df_head.set_value(index_head, 'onset_date4', df_seizure_list.iloc[index_seizure_list]['onset_date'])
                    df_head.set_value(index_head, 'offset_date4', df_seizure_list.iloc[index_seizure_list]['offset_date'])
                    df_head.set_value(index_head, 'onset_sample4', df_seizure_list.iloc[index_seizure_list]['onset_sample'])
                    df_head.set_value(index_head, 'offset_sample4', df_seizure_list.iloc[index_seizure_list]['offset_sample'])
                else:
                    df_head.set_value(index_head, 'onset_date5', df_seizure_list.iloc[index_seizure_list]['onset_date'])
                    df_head.set_value(index_head, 'offset_date5', df_seizure_list.iloc[index_seizure_list]['offset_date'])
                    df_head.set_value(index_head, 'onset_sample5', df_seizure_list.iloc[index_seizure_list]['onset_sample'])
                    df_head.set_value(index_head, 'offset_sample5', df_seizure_list.iloc[index_seizure_list]['offset_sample'])
        df_head.set_value(index_head, 'n_seizures', seizures_count)
    df_head2 = df_head[['patient_ID', 'file_name', 'relative_path', 'n_channels', 'n_samples', 'begin_date', 'end_date', 'n_seizures', \
        'onset_date1', 'offset_date1', 'onset_sample1', 'offset_sample1', 'onset_date2', 'offset_date2', 'onset_sample2', 'offset_sample2', \
            'onset_date3', 'offset_date3', 'onset_sample3', 'offset_sample3', 'onset_date4', 'offset_date4', 'onset_sample4', 'offset_sample4', \
                'onset_date5', 'offset_date5', 'onset_sample5', 'offset_sample5']]
    df_head3 = pd.DataFrame(index=range(2*len(df_head2['n_samples'])), columns = df_head2.columns)
    df_head3 = df_head3.fillna(0)
    insert_count = 0
    for index_head2, row_head2 in df_head2.iterrows():
        df_head3.iloc[2*index_head2,:] = df_head2.iloc[index_head2,:].values
        if index_head2 == 0:
            continue
        else: 
            if row_head2['begin_date'] != df_head2.loc[index_head2-1]['end_date']:
                insert_count += 1
                index_insert = 2 * index_head2 - 1
                df_head3.set_value(index_insert, 'patient_ID', df_head2.iloc[index_head2-1]['patient_ID'])                
                if int((row_head2['begin_date'] - df_head2.iloc[index_head2-1]['end_date']).total_seconds()) > 0:
                    df_head3.set_value(index_insert, 'file_name', 'unrecorded_gap')
                    df_head3.set_value(index_insert, 'begin_date', df_head2.iloc[index_head2-1]['end_date'])
                    df_head3.set_value(index_insert, 'end_date', df_head2.iloc[index_head2]['begin_date'])
                else:
                    df_head3.set_value(index_insert, 'file_name', 'recorded_overlap')
                    df_head3.set_value(index_insert, 'begin_date', df_head2.iloc[index_head2]['begin_date'])
                    df_head3.set_value(index_insert, 'end_date', df_head2.iloc[index_head2-1]['end_date'])                    
                df_head3.set_value(index_insert, 'relative_path', 0)
                df_head3.set_value(index_insert, 'n_channels', 0)
                df_head3.set_value(index_insert, 'n_samples', 0)
                df_head3.set_value(index_insert, 'n_seizures', 0)
                df_head3.set_value(index_insert, 'onset_date1', 0)
                df_head3.set_value(index_insert, 'offset_date1', 0)
                df_head3.set_value(index_insert, 'onset_sample1', 0)
                df_head3.set_value(index_insert, 'offset_sample1', 0)
                df_head3.set_value(index_insert, 'onset_date2', 0)
                df_head3.set_value(index_insert, 'offset_date2', 0)
                df_head3.set_value(index_insert, 'onset_sample2', 0)
                df_head3.set_value(index_insert, 'offset_sample2', 0)
                df_head3.set_value(index_insert, 'onset_date3', 0)
                df_head3.set_value(index_insert, 'offset_date3', 0)
                df_head3.set_value(index_insert, 'onset_sample3', 0)
                df_head3.set_value(index_insert, 'offset_sample3', 0)
                df_head3.set_value(index_insert, 'onset_date4', 0)
                df_head3.set_value(index_insert, 'offset_date4', 0)
                df_head3.set_value(index_insert, 'onset_sample4', 0)
                df_head3.set_value(index_insert, 'offset_sample4', 0)
                df_head3.set_value(index_insert, 'onset_date5', 0)
                df_head3.set_value(index_insert, 'offset_date5', 0)
                df_head3.set_value(index_insert, 'onset_sample5', 0)
                df_head3.set_value(index_insert, 'offset_sample5', 0)
    df_head3 = df_head3[(df_head3.T != 0).any()]#store each row data that not equal to 0 in dataframe
    df_head3.reset_index(drop=True, inplace=True)
    patient_ID = df_head3.iloc[0]['patient_ID']
    fname = 'data_timeline_' + str(patient_ID) + '.csv' 
    df_head3.to_csv('./stored_csv/'+fname, sep=',', index=False)
    return df_head3

def merge_patients_df(binary_files, seizurelist_path='seizurelist/'):
    patients_ID = []
    for binary_file in binary_files:
        patients_ID.append(binary_file.split('/')[7])# split path by /, add 7th subpath
    patients_ID_unrepeated = list(set(patients_ID))#use set to record unrepeat ID
    count = 0
    for patient_ID in patients_ID_unrepeated: #get patient ID's data
        binary_files_matched = [binary_file for binary_file in binary_files if patient_ID in binary_file]
        seizurelist_filename = seizurelist_path + 'seizurelist_' + patient_ID[4:] + '.txt'
        if count == 0:
            df_head_all = merge_df(binary_files_matched, seizurelist_filename)
        else:
            df_head = merge_df(binary_files_matched, seizurelist_filename)
            frames = [df_head_all, df_head]
            df_head_all = pd.concat(frames)
        count += 1
    df_head_all.sort_values(by=['patient_ID', 'begin_date'], inplace=True)
    df_head_all.reset_index(drop=True, inplace=True)
    df_head_all.to_csv('./stored_csv/data_timeline_all_patients.csv', sep=',', index=False)
    return df_head_all

def extract_begin_end_sample(y_raw):
    preictal_begin_sample, preictal_end_sample = [], []
    for i in range(len(y_raw)):
        if i == 0:
            if y_raw[i] == 1:
                preictal_begin_sample.append(i)
        elif i == len(y_raw) - 1:
            if y_raw[i] == 1:
                preictal_end_sample.append(i+1)
        else:
            if y_raw[i-1] != y_raw[i]:
                if y_raw[i-1] == 1:
                    preictal_end_sample.append(i)
                if y_raw[i] == 1:
                    preictal_begin_sample.append(i)     
    return preictal_begin_sample, preictal_end_sample

def create_preictal_data(df_timeline, preictal_period_minutes, preictal_step_second, postictal_period_minutes=10):
    print('creating preictal data...')
    patient_ID = df_timeline.iloc[0]['patient_ID']
    #x_pkl_fname = stored_data_path + 'patient_ID_' + str(patient_ID) + '_x.pkl'
    x_stats_fname = stored_data_path + 'patient_ID_' + str(patient_ID) + '_x_stats.pkl'
    with open(x_stats_fname, 'rb') as infile:
        dataStats = pkl.load(infile)
    x_image_chain = tools.merge_small_data(patient_ID)
    dataAvg = dataStats[0]
    dataStd = dataStats[1]
    preictal_video = []
    preictal_video_time_index, preictal_video_and_time_index = [], []
    begin_recorded_date = df_timeline.iloc[0]['begin_date']
    for _, row in df_timeline.iterrows():
        data_file_relative_begin_second = int((row['begin_date'] - begin_recorded_date).total_seconds())
        recorded_samples = row['n_samples']
        onset_sample_list, offset_sample_list, preictal_begin_sample_list, preictal_end_sample_list = [], [], [], []     
        if row['n_seizures'] > 0:
            onset_sample_list.append((row['onset_sample1']//256)*256)
            offset_sample_list.append((row['offset_sample1']//256)*256)
        if row['n_seizures'] > 1:
            onset_sample_list.append((row['onset_sample2']//256)*256)
            offset_sample_list.append((row['offset_sample2']//256)*256)
        if row['n_seizures'] > 2:
            onset_sample_list.append((row['onset_sample3']//256)*256)
            offset_sample_list.append((row['offset_sample3']//256)*256)
        if row['n_seizures'] > 3:
            onset_sample_list.append((row['onset_sample4']//256)*256)
            offset_sample_list.append((row['offset_sample4']//256)*256)
        if row['n_seizures'] > 4:
            onset_sample_list.append((row['onset_sample5']//256)*256)
            offset_sample_list.append((row['offset_sample5']//256)*256)
            # label for y
        if row['n_seizures'] > 0:        
            y_raw = [int(2)] * recorded_samples
            for onset_sample, offset_sample in zip(onset_sample_list, offset_sample_list):
                y_raw[onset_sample:offset_sample] = [int(0)] * (offset_sample - onset_sample)
            # label for class 3 (postictal)
            for i in range(len(offset_sample_list)):
                y_frag_remain_index, y_frag_remain_class, remain_label = [], [], [0]
                if recorded_samples - offset_sample_list[i] < 256*60*postictal_period_minutes:
                    y_frag_old = y_raw[offset_sample_list[i]:recorded_samples]
                    for j in range(len(y_frag_old)):
                        if y_frag_old[j] in remain_label:
                            y_frag_remain_index.append(j)
                            y_frag_remain_class.append(y_frag_old[j])
                    y_frag_update = [int(3)] * (recorded_samples - offset_sample_list[i])
                    for k in range(len(y_frag_remain_index)):
                        y_frag_update[y_frag_remain_index[k]] = y_frag_remain_class[k]
                    y_raw[offset_sample_list[i]:recorded_samples] = y_frag_update
                else:
                    y_frag_old = y_raw[offset_sample_list[i]:(offset_sample_list[i]+256*60*postictal_period_minutes)]
                    for j in range(len(y_frag_old)):
                        if y_frag_old[j] in remain_label:
                            y_frag_remain_index.append(j)
                            y_frag_remain_class.append(y_frag_old[j])
                    y_frag_update = [int(3)] * (256*60*postictal_period_minutes)
                    for k in range(len(y_frag_remain_index)):
                        y_frag_update[y_frag_remain_index[k]] = y_frag_remain_class[k]
                    y_raw[offset_sample_list[i]:(offset_sample_list[i]+256*60*postictal_period_minutes)] = y_frag_update
            # label for class 1 (preictal)
            for i in range(len(onset_sample_list)): 
                y_frag_remain_index, y_frag_remain_class, remain_label = [], [], [0, 3]
                if onset_sample_list[i] < 256*60*preictal_period_minutes:
                    y_frag_old = y_raw[:onset_sample_list[i]]
                    for j in range(len(y_frag_old)):
                        if y_frag_old[j] in remain_label:
                            y_frag_remain_index.append(j)
                            y_frag_remain_class.append(y_frag_old[j])
                    y_frag_update = [int(1)] * onset_sample_list[i]
                    for k in range(len(y_frag_remain_index)):
                        y_frag_update[y_frag_remain_index[k]] = y_frag_remain_class[k]
                    y_raw[:onset_sample_list[i]] = y_frag_update
                else:
                    y_frag_old = y_raw[(onset_sample_list[i]-256*60*preictal_period_minutes):onset_sample_list[i]]
                    for j in range(len(y_frag_old)):
                        if y_frag_old[j] in remain_label:
                            y_frag_remain_index.append(j)
                            y_frag_remain_class.append(y_frag_old[j])                                                    
                    y_frag_update = [int(1)] * (256*60*preictal_period_minutes)
                    for k in range(len(y_frag_remain_index)):
                        y_frag_update[y_frag_remain_index[k]] = y_frag_remain_class[k]
                    y_raw[(onset_sample_list[i]-256*60*preictal_period_minutes):onset_sample_list[i]] = y_frag_update

def create_data_from_df(df_timeline, preictal_period_minutes=10, postictal_period_minutes=10):
    #label class 0:ictal, 1:preictal 2:interictal 3:postictal 4:unrecorded_gap
    #classification priority: 4, 0, 3, 1, 2
    patient_data_metadata = []
    patient_ID = int(df_timeline.iloc[0]['patient_ID'])
    preictal_period_seconds = 60 * preictal_period_minutes
    postictal_period_seconds = 60 * postictal_period_minutes
    begin_recorded_date = df_timeline.iloc[0]['begin_date']
    end_recorded_date = df_timeline.iloc[len(df_timeline.index)-1]['end_date']
    total_recorded_seconds = int((end_recorded_date - begin_recorded_date).total_seconds())
    y_raw = np.zeros((total_recorded_seconds,), dtype=np.int)
    onset_second, offset_second = [], []
    # label for class 4 (unrecorded_gap) and class 0 (ictal)
    for _, row in df_timeline.iterrows():
        frag_begin_second = int((row['begin_date'] - begin_recorded_date).total_seconds())
        frag_end_second = int((row['end_date'] - begin_recorded_date).total_seconds())
        if row['file_name'] == 'unrecorded_gap':
            y_raw[frag_begin_second:frag_end_second] = [int(4)] * (frag_end_second - frag_begin_second)
        else:
            y_raw[frag_begin_second:frag_end_second] = [int(2)] * (frag_end_second - frag_begin_second)
            if row['n_seizures'] > 0:
                onset_second.append(int((row['onset_date1'] - begin_recorded_date).total_seconds()))
                offset_second.append(int((row['offset_date1'] - begin_recorded_date).total_seconds()))
            if row['n_seizures'] > 1:
                onset_second.append(int((row['onset_date2'] - begin_recorded_date).total_seconds()))
                offset_second.append(int((row['offset_date2'] - begin_recorded_date).total_seconds()))
            if row['n_seizures'] > 2:
                onset_second.append(int((row['onset_date3'] - begin_recorded_date).total_seconds()))
                offset_second.append(int((row['offset_date3'] - begin_recorded_date).total_seconds()))
            if row['n_seizures'] > 3:
                onset_second.append(int((row['onset_date4'] - begin_recorded_date).total_seconds()))
                offset_second.append(int((row['offset_date4'] - begin_recorded_date).total_seconds()))
            if row['n_seizures'] > 4:
                onset_second.append(int((row['onset_date5'] - begin_recorded_date).total_seconds()))
                offset_second.append(int((row['offset_date5'] - begin_recorded_date).total_seconds()))
    for on_sec, off_sec in zip(onset_second, offset_second):
        y_raw[on_sec:off_sec] = [int(0)] * (off_sec - on_sec)
    # label for class 3 (postictal)
    for i in range(len(offset_second)):
        y_frag_remain_index, y_frag_remain_class, remain_label = [], [], [0, 4]
        if total_recorded_seconds - offset_second[i] < postictal_period_seconds:
            y_frag_old = y_raw[offset_second[i]:total_recorded_seconds]
            for j in range(len(y_frag_old)):
                if y_frag_old[j] in remain_label:
                    y_frag_remain_index.append(j)
                    y_frag_remain_class.append(y_frag_old[j])
            y_frag_update = [int(3)] * (total_recorded_seconds - offset_second[i])
            for k in range(len(y_frag_remain_index)):
                y_frag_update[y_frag_remain_index[k]] = y_frag_remain_class[k]
            y_raw[offset_second[i]:total_recorded_seconds] = y_frag_update
        else:
            y_frag_old = y_raw[offset_second[i]:offset_second[i]+postictal_period_seconds]
            for j in range(len(y_frag_old)):
                if y_frag_old[j] in remain_label:
                    y_frag_remain_index.append(j)
                    y_frag_remain_class.append(y_frag_old[j])
            y_frag_update = [int(3)] * postictal_period_seconds
            for k in range(len(y_frag_remain_index)):
                y_frag_update[y_frag_remain_index[k]] = y_frag_remain_class[k]
            y_raw[offset_second[i]:offset_second[i]+postictal_period_seconds] = y_frag_update
    # label for class 1 (preictal)
    for i in range(len(onset_second)): 
        y_frag_remain_index, y_frag_remain_class, remain_label = [], [], [0, 3, 4]
        if onset_second[i] < preictal_period_seconds:
            y_frag_old = y_raw[:onset_second[i]]
            for j in range(len(y_frag_old)):
                if y_frag_old[j] in remain_label:
                    y_frag_remain_index.append(j)
                    y_frag_remain_class.append(y_frag_old[j])
            y_frag_update = [int(1)] * onset_second[i]
            for k in range(len(y_frag_remain_index)):
                y_frag_update[y_frag_remain_index[k]] = y_frag_remain_class[k]
            y_raw[:onset_second[i]] = y_frag_update
        else:
            y_frag_old = y_raw[onset_second[i]-preictal_period_seconds:onset_second[i]]
            for j in range(len(y_frag_old)):
                if y_frag_old[j] in remain_label:
                    y_frag_remain_index.append(j)
                    y_frag_remain_class.append(y_frag_old[j])                                                    
            y_frag_update = [int(1)] * preictal_period_seconds
            for k in range(len(y_frag_remain_index)):
                y_frag_update[y_frag_remain_index[k]] = y_frag_remain_class[k]
            y_raw[onset_second[i]-preictal_period_seconds:onset_second[i]] = y_frag_update
    ictal_begin_second, ictal_end_second, preictal_begin_second, preictal_end_second = [], [], [], []
    interictal_begin_second, interictal_end_second = [], []
    postictal_begin_second, postictal_end_second, unrecorded_begin_second, unrecorded_end_second = [], [], [], []
    for i in range(y_raw.shape[0]):
        if i == 0:
            if y_raw[i] == 0:
                ictal_begin_second.append(i)
            elif y_raw[i] == 1:
                preictal_begin_second.append(i)
            elif y_raw[i] == 2:
                interictal_begin_second.append(i)
            elif y_raw[i] == 3:
                postictal_begin_second.append(i)
            else:
                unrecorded_begin_second.append(i)
        elif i == y_raw.shape[0] - 1:
            if y_raw[i] == 0:
                ictal_end_second.append(i+1)
            elif y_raw[i] == 1:
                preictal_end_second.append(i+1)
            elif y_raw[i] == 2:
                interictal_end_second.append(i+1)
            elif y_raw[i] == 3:
                postictal_end_second.append(i+1)
            else:
                unrecorded_end_second.append(i+1)
        else:
            if y_raw[i-1] != y_raw[i]:
                if y_raw[i-1] == 0:
                    ictal_end_second.append(i)
                elif y_raw[i-1] == 1:
                    preictal_end_second.append(i)
                elif y_raw[i-1] == 2:
                    interictal_end_second.append(i)
                elif y_raw[i-1] == 3:
                    postictal_end_second.append(i)
                else:
                    unrecorded_end_second.append(i)
                if y_raw[i] == 0:
                    ictal_begin_second.append(i)
                elif y_raw[i] == 1:
                    preictal_begin_second.append(i)
                elif y_raw[i] == 2:
                    interictal_begin_second.append(i)
                elif y_raw[i] == 3:
                    postictal_begin_second.append(i)
                else:
                    unrecorded_begin_second.append(i)                

    x_pkl_fname = stored_data_path + 'patient_ID_' + str(patient_ID) + '_x_0.pkl'
    x_pkl_file = Path(x_pkl_fname)
    if not x_pkl_file.is_file():
        print('x_pkl_file does not exist. Generating new x_pkl_file...')
        count = 0
        x_image_chain = []
        x_image_chain_nonzero = []
        for _, row in df_timeline.iterrows():
            if row['file_name'] == 'unrecorded_gap':
                gap_second = int((row['end_date'] - row['begin_date']).total_seconds())
                x_zero_gap = np.zeros((gap_second, x_image_chain.shape[1], x_image_chain.shape[2]))
                x_image_chain = np.concatenate((x_image_chain, x_zero_gap), axis=0)
            elif row['file_name'] == 'recorded_overlap':
                overlap_recorded_second = int((row['end_date'] - row['begin_date']).total_seconds())
                x_image_chain = x_image_chain[:(-1)*overlap_recorded_second]
            else:
                x_image_frag = tools.process_raw_data(row['relative_path'], row['n_channels'])
                if count == 0:
                    x_image_chain.append(x_image_frag)
                    x_image_chain = np.array(x_image_chain[0])
                    x_image_chain_nonzero.append(x_image_frag)
                    x_image_chain_nonzero = np.array(x_image_chain_nonzero[0])                    
                else:
                    x_image_chain = np.concatenate((x_image_chain, x_image_frag), axis=0)
                    x_image_chain_nonzero = np.concatenate((x_image_chain_nonzero, x_image_frag), axis=0)
            count += 1
            print('x_image_chain shape:')
            print(x_image_chain.shape)
            dataAvg = np.average(x_image_chain_nonzero, axis=0)
            dataStd = np.std(x_image_chain_nonzero, axis=0)
            dataStats = np.array([dataAvg , dataStd])
        
        x_stats_fname = stored_data_path + 'patient_ID_' + str(patient_ID) + '_x_stats.pkl'
        with open(x_stats_fname, 'wb') as infile:
            pkl.dump(dataStats, infile)
        tools.split_large_data(patient_ID, x_image_chain)
        del x_image_chain, x_image_chain_nonzero, dataAvg, dataStd, dataStats
    else:
        print('x_pkl_file exists.')
        x_image_chain = tools.merge_small_data(patient_ID)
        if y_raw.shape[0] == x_image_chain.shape[0]:
            print('x,y matched!!!!!!!!!!!')
        else:
            print(x_image_chain.shape)
            print(y_raw.shape)
            raise ValueError('x,y not matched!!!')
        del x_image_chain
    #patient_data_metadata 0:y_raw, 1:ictal_begin_second, 2:ictal_end_second, 3:preictal_begin_second,
    #4:preictal_end_second, 5:interictal_begin_second, 6:interictal_end_second, 7:postictal_begin_second, 
    #8:postictal_end_second, 9:unrecorded_begin_second, 10:unrecorded_begin_second
    patient_data_metadata.append(y_raw)
    patient_data_metadata.append(ictal_begin_second)
    patient_data_metadata.append(ictal_end_second)
    patient_data_metadata.append(preictal_begin_second)
    patient_data_metadata.append(preictal_end_second)
    patient_data_metadata.append(interictal_begin_second)
    patient_data_metadata.append(interictal_end_second)
    patient_data_metadata.append(postictal_begin_second)
    patient_data_metadata.append(postictal_end_second)
    patient_data_metadata.append(unrecorded_begin_second)
    patient_data_metadata.append(unrecorded_end_second)

    pkl_fname = 'patient_ID_' + str(patient_ID) + '_metadata_pre_' + str(preictal_period_minutes) + '_post_' + str(postictal_period_minutes) + '.pkl'
    with open(stored_data_path + pkl_fname, 'wb') as infile:
        pkl.dump(patient_data_metadata, infile)

def create_data_from_df_all_patients(patients_ID, df_timeline_all, preictal_period_minutes=10, postictal_period_minutes=10):
    #patient_data_metadata 0:y_raw, 1:ictal_begin_second, 2:ictal_end_second, 3:preictal_begin_second,
    #4:preictal_end_second, 5:interictal_begin_second, 6:interictal_end_second, 7:postictal_begin_second, 
    #8:postictal_end_second, 9:unrecorded_begin_second, 10:unrecorded_begin_second
    patients_ID_unrepeated = patients_ID
    for patient_ID in patients_ID_unrepeated:
        print('create patient %s metadata from dataframe...' %(patient_ID))
        df_patient = df_timeline_all.loc[df_timeline_all['patient_ID'] == int(patient_ID)]
        create_data_from_df(df_patient, preictal_period_minutes=preictal_period_minutes)

def create_eeg_video(eeg_data, begin_sec_list, end_sec_list, video_second, n_samples=1, inter=False):
    video, time_index = [], []
    video_sample, time_index_sample = [], []
    zero_count = 0
    non_zero_count = 0
    eeg_video = eeg_data[0]
    print(eeg_video.shape)
    dataStats = eeg_data[1]
    dataAvg = dataStats[0]
    dataStd = dataStats[1]
    for b_sec, e_sec in zip(begin_sec_list, end_sec_list):
        if e_sec - b_sec + 1 >= video_second:
            for index in range(b_sec, e_sec-video_second):
                if np.count_nonzero(eeg_video[index:index+video_second]) >= 0.8 * 30 * 50 * 19: # 0.8*30*50*19
                    video.append(eeg_video[index:index+video_second])
                    time_index.append(index)
                    non_zero_count += 1
                else:
                    zero_count += 1
                    #import ipdb; ipdb.set_trace()
    print('zero count is: %d' %(zero_count))
    print('non zero count is: %d' %(non_zero_count))
    if inter == True and len(video) > n_samples:
        n_sample_one = len(video) // n_samples
        print('down sampling ratio: %d ...' %(int(n_sample_one)))
        video_sample = video[::n_sample_one]   
        time_index_sample = time_index[::n_sample_one]
    else:
        video_sample = video   
        time_index_sample = time_index       
    del video, time_index
    video_sample_normalize = []
    for vs in video_sample:
        video_sample_norm = (vs - dataAvg) / dataStd
        video_sample_normalize.append(video_sample_norm)
    video_sample_normalize_and_time_index = []
    video_sample_normalize_and_time_index.append(video_sample_normalize)
    video_sample_normalize_and_time_index.append(time_index_sample)
    return video_sample_normalize_and_time_index

def create_eeg_data(patient_ID, preictal_period_minutes, postictal_period_minutes=10, video_second=30, inter=False):
    pkl_fname = 'patient_ID_' + str(patient_ID) + '_metadata_pre_' + str(preictal_period_minutes) + '_post_' + str(postictal_period_minutes) + '.pkl'
    patient_metadata = []
    with open(stored_data_path + pkl_fname, 'rb') as infile:
        patient_metadata = pkl.load(infile)
    if inter == True:
        begin_second = patient_metadata[5]
    else:
        begin_second = patient_metadata[3]

    if inter == True:
        end_second = patient_metadata[6]
    else:
        end_second = patient_metadata[4]
    del patient_metadata
    interictal_n_samples = 1
    if inter == True:
        preictal_data_fname = stored_data_path + 'patient_ID_' + str(patient_ID) + '_preictal_' + str(preictal_period_minutes) + '_data.pkl'
        with open(preictal_data_fname, 'rb') as infile:
            preictal_video = pkl.load(infile)    
        interictal_n_samples = 5 * len(preictal_video)
        del preictal_video
        print('create interictal_video')
    else:
        print('create preictal_video')
    x_stats_fname = stored_data_path + 'patient_ID_' + str(patient_ID) + '_x_stats.pkl'
    with open(x_stats_fname, 'rb') as infile:
        dataStats = pkl.load(infile)
    x_image_chain = tools.merge_small_data(patient_ID)
    dataWithStats = []
    dataWithStats.append(x_image_chain)
    dataWithStats.append(dataStats)
    video_and_time_index = create_eeg_video(dataWithStats, begin_second, end_second, video_second, n_samples=interictal_n_samples, inter=inter)
    if inter == True:
        print('interictal video data: %d' %(len(video_and_time_index[0])))
        print('interictal video time index: %d' %(len(video_and_time_index[1])))
        interictal_data_fname = stored_data_path + 'patient_ID_' + str(patient_ID) + '_interictal_' + str(preictal_period_minutes) + '_data.pkl'
        interictal_data_ti_fname = stored_data_path + 'patient_ID_' + str(patient_ID) + '_interictal_ti_' + str(preictal_period_minutes) + '_data.pkl'
        with open(interictal_data_fname, 'wb') as infile_interictal:
            pkl.dump(video_and_time_index[0], infile_interictal)
        with open(interictal_data_ti_fname, 'wb') as infile_interictal_ti:
            pkl.dump(video_and_time_index[1], infile_interictal_ti)
    else:
        print('preictal video data: %d' %(len(video_and_time_index[0])))
        print('preictal video time index: %d' %(len(video_and_time_index[1])))
        preictal_data_fname = stored_data_path + 'patient_ID_' + str(patient_ID) + '_preictal_' + str(preictal_period_minutes) + '_data.pkl'
        preictal_data_ti_fname = stored_data_path + 'patient_ID_' + str(patient_ID) + '_preictal_ti_' + str(preictal_period_minutes) + '_data.pkl'
        with open(preictal_data_fname, 'wb') as infile_preictal:
            pkl.dump(video_and_time_index[0], infile_preictal)
        with open(preictal_data_ti_fname, 'wb') as infile_preictal_ti:
            pkl.dump(video_and_time_index[1], infile_preictal_ti)        

def main():
    #n_ictals: 102:11, 7302:8, 8902:5, 11002:8, 16202:8, 21602:11, 21902:6, 22602:21, 23902:5, 26102:8, 30002:14, 30802:9,
    #32502:8, 32702:6, 45402:5, 46702:6, 55202:9, 56402:7, 58602:22, 59102:7, 75202:9, 85202:10, 92102:7, 93902:9,
    #96002:9, 103002:8, 109502:10, 111902:8, 114902:12 
    rec_paths = tools.get_patients_dirs()
    binary_files_all = []
    for rec_path in rec_paths:
        n_path, binary_files = tools.get_patient_files(rec_path)
        binary_files_all += binary_files # get all data path in surf30
    df_head_all_patients = merge_patients_df(binary_files_all)
    patients_ID = []
    for binary_file in binary_files_all:
        patients_ID.append(binary_file.split('/')[7][4:])
    patients_ID_unrepeated = list(set(patients_ID))
    patients_ID_unrepeated = ['11002', '16202', '26102', '30802', '32702', '45402', '55202', '103002', '111902']
    preictal_period_minutes = 10
    create_data_from_df_all_patients(patients_ID_unrepeated, df_head_all_patients, preictal_period_minutes=preictal_period_minutes)
    for patient_ID in patients_ID_unrepeated:
        print(patient_ID)
        create_eeg_data(patient_ID, preictal_period_minutes, inter=False)
        create_eeg_data(patient_ID, preictal_period_minutes, inter=True)

if __name__ == '__main__':
    main()
