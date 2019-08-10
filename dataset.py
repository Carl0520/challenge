#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 20:29:03 2019

@author: gaoyi
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder 
label_encoder = LabelEncoder()
from torch.utils.data import DataLoader
import torch.utils.data as Data
import torch
# Task
task_name = 'ComParE2019_BabySounds'
classes   = ['Canonical','Crying','Junk','Laughing','Non-canonical']
lab_encoder = label_encoder.fit(classes)
classes2   = [0,1,2,3,4]


# Mapping each available feature set to tuple (number of features, offset/index of first feature, separator, header option)
feat_conf = {'ComParE':      (6373, 1, ';', 'infer'),
             'BoAW-125':     ( 250, 1, ';',  None),
             'BoAW-250':     ( 500, 1, ';',  None),
             'BoAW-500':     (1000, 1, ';',  None),
             'BoAW-1000':    (2000, 1, ';',  None),
             'BoAW-2000':    (4000, 1, ';',  None),
             'auDeep-40':    (1024, 2, ',', 'infer'),
             'auDeep-50':    (1024, 2, ',', 'infer'),
             'auDeep-60':    (1024, 2, ',', 'infer'),
             'auDeep-70':    (1024, 2, ',', 'infer'),
             'auDeep-fused': (4096, 2, ',', 'infer'),
             'IS10':(1582,1,';','infer')}


# Path of the features and labels
features_path = 'feature_path'
label_file    =  'label_csv_path'
df_labels_ = pd.read_csv(label_file)

print('start')
def get_feature(fea_set):
# Load features and labels
    num_feat = feat_conf[fea_set][0]
    ind_off  = feat_conf[fea_set][1]
    sep      = feat_conf[fea_set][2]
    header   = feat_conf[fea_set][3]

    X_train_ = pd.read_csv(features_path + task_name + '.' + fea_set + '.train.csv', sep=sep, header=header, usecols=range(ind_off,num_feat+ind_off), dtype=np.float32).values
    X_devel_ = pd.read_csv(features_path + task_name + '.' + fea_set + '.devel.csv', sep=sep, header=header, usecols=range(ind_off,num_feat+ind_off), dtype=np.float32).values
    X_test_  = pd.read_csv(features_path + task_name + '.' + fea_set + '.test.csv',  sep=sep, header=header, usecols=range(ind_off,num_feat+ind_off), dtype=np.float32).values
    
    
    y_train_ = df_labels_['label'][df_labels_['file_name'].str.startswith('train')].values
    y_devel_ = df_labels_['label'][df_labels_['file_name'].str.startswith('devel')].values
    
    return X_train_ , X_devel_, X_test_, df_labels_, y_train_, y_devel_


def preprocess(batch_size):
    X_train_C , X_devel_C, X_test_C, df_labels_C, y_train_C, y_devel_C = get_feature('ComParE')
    X_train_A , X_devel_A, X_test_A, df_labels_A, y_train_A, y_devel_A = get_feature('auDeep-fused')
    X_train_B , X_devel_B, X_test_B, df_labels_B, y_train_B, y_devel_B = get_feature('IS10')
    
    X_train_ = np.concatenate([X_train_A,X_train_B,X_train_C],1)
    X_devel_ = np.concatenate([X_devel_A,X_devel_B,X_devel_C],1)
    X_test_ = np.concatenate([X_test_A,X_test_B,X_test_C],1)
    
    y_train_ = y_train_A
    y_devel_ = y_devel_A
    #encode label
    y_train_ = lab_encoder.transform(y_train_)
    y_devel_ = lab_encoder.transform(y_devel_)
    
    
    X_traindevel_ = np.concatenate((X_train_, X_devel_))
    y_traindevel_ = np.concatenate((y_train_, y_devel_))
    
    
    # Upsampling / Balancing
    print('Upsampling ... ')
    num_samples_train      = []
    num_samples_traindevel = []
    for label in classes2:
        num_samples_train.append( len(y_train_[y_train_==label]) )
        num_samples_traindevel.append( len(y_traindevel_[y_traindevel_==label]) )
    for label, ns_tr, ns_trd in zip(classes2, num_samples_train, num_samples_traindevel):
        factor_tr    = np.max(num_samples_train) // ns_tr
        X_train_      = np.concatenate((X_train_, np.tile(X_train_[y_train_==label], (factor_tr-1, 1))))
        y_train_      = np.concatenate((y_train_, np.tile(y_train_[y_train_==label], (factor_tr-1))))
        factor_trd   = np.max(num_samples_traindevel) // ns_trd
        X_traindevel_ = np.concatenate((X_traindevel_, np.tile(X_traindevel_[y_traindevel_==label], (factor_trd-1, 1))))
        y_traindevel_ = np.concatenate((y_traindevel_, np.tile(y_traindevel_[y_traindevel_==label], (factor_trd-1))))
    
    # Feature normalisation
    scaler       = StandardScaler()
    X_train_      = scaler.fit_transform(X_train_)
    X_devel_      = scaler.transform(X_devel_)
    X_traindevel_ = scaler.fit_transform(X_traindevel_)
    X_test_       = scaler.transform(X_test_)
    
    
    train_set = Data.TensorDataset(torch.from_numpy(X_train_),torch.from_numpy(y_train_))
    train_loader_ = DataLoader(train_set,batch_size=batch_size, shuffle=True,drop_last=True)
    
    train_loader_2 = DataLoader(train_set,batch_size=batch_size, shuffle=False)
    
    devel_set = Data.TensorDataset(torch.from_numpy(X_devel_),torch.from_numpy(y_devel_))
    devel_loader_ = DataLoader(devel_set,batch_size=batch_size, shuffle=False)
    
    test_set = Data.TensorDataset(torch.from_numpy(X_test_))
    test_loader_ = DataLoader(test_set,batch_size=batch_size, shuffle=False)
    
    traindevel_set = Data.TensorDataset(torch.from_numpy(X_traindevel_),torch.from_numpy(y_traindevel_))
    traindevel_loader_ = DataLoader(traindevel_set,batch_size=batch_size, shuffle=True,drop_last=True)
    
    
    print('finish preprocess')
    
    return train_loader_, train_loader_2  , devel_loader_, test_loader_, traindevel_loader_
