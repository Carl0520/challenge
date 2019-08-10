#!/usr/bin/python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import recall_score, confusion_matrix
from sklearn.feature_selection import SelectPercentile,f_classif
import torch
from model import vae
import prior
import joblib
import matplotlib.pyplot as plt
np.random.seed(0)
# Task
task_name = 'ComParE2019_BabySounds'
classes   = ['Canonical','Crying','Junk','Laughing','Non-canonical']

# Enter your team name HERE
team_name = 'baseline'

# Enter your submission number HERE
submission_index = 3

# Option
show_confusion = True   # Display confusion matrix on devel

# Configuration
#feature_set = 'ComParE'  # For all available options, see the dictionary feat_conf
complexities = [3*1e-5]  # SVM complexities (linear kernel)


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
features_path = '/home/gaoyi/Interspeech_challenge/ComParE2019_BabySounds_new/features/'
label_file    =  '/home/gaoyi/Interspeech_challenge/ComParE2019_BabySounds_new/lab/labels.csv'

# Start
#print('\nRunning ' + task_name + ' ' + fea_set + ' baseline ... (this might take a while) \n')
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
    
    df_labels_ = pd.read_csv(label_file)
    y_train_ = df_labels_['label'][df_labels_['file_name'].str.startswith('train')].values
    y_devel_ = df_labels_['label'][df_labels_['file_name'].str.startswith('devel')].values
    
    return X_train_ , X_devel_, X_test_, df_labels_, y_train_, y_devel_


X_train_C , X_devel_C, X_test_C, df_labels_C, y_train_C, y_devel_C = get_feature('ComParE')
X_train_A , X_devel_A, X_test_A, df_labels_A, y_train_A, y_devel_A = get_feature('auDeep-fused')
X_train_B , X_devel_B, X_test_B, df_labels_B, y_train_B, y_devel_B = get_feature('IS10')

X_train = np.concatenate([X_train_A,X_train_B,X_train_C],1)
X_devel = np.concatenate([X_devel_A,X_devel_B,X_devel_C],1)
X_test = np.concatenate([X_test_A,X_test_B,X_test_C],1)
df_labels = df_labels_A

y_train = y_train_A
y_devel = y_devel_A
X_traindevel = np.concatenate((X_train, X_devel))
y_traindevel = np.concatenate((y_train_A, y_devel_A))


# Upsampling / Balancing
print('Upsampling ... ')
num_samples_train      = []
num_samples_traindevel = []
for label in classes:
    num_samples_train.append( len(y_train[y_train==label]) )
    num_samples_traindevel.append( len(y_traindevel[y_traindevel==label]) )
for label, ns_tr, ns_trd in zip(classes, num_samples_train, num_samples_traindevel):
    factor_tr    = np.max(num_samples_train) // ns_tr
    X_train      = np.concatenate((X_train, np.tile(X_train[y_train==label], (factor_tr-1, 1))))
    y_train      = np.concatenate((y_train, np.tile(y_train[y_train==label], (factor_tr-1))))
    factor_trd   = np.max(num_samples_traindevel) // ns_trd
    X_traindevel = np.concatenate((X_traindevel, np.tile(X_traindevel[y_traindevel==label], (factor_trd-1, 1))))
    y_traindevel = np.concatenate((y_traindevel, np.tile(y_traindevel[y_traindevel==label], (factor_trd-1))))

# Feature normalisation
scaler       = StandardScaler()
X_train      = scaler.fit_transform(X_train)
X_devel      = scaler.transform(X_devel)
X_traindevel = scaler.fit_transform(X_traindevel)
X_test       = scaler.transform(X_test)

#%%  augmentation


def aug(X_train_,y_train_,aug_lab,aug_num):
    
    decoder = vae.Decoder()
    decoder.load_state_dict(torch.load('/home/gaoyi/Interspeech_challenge/ComParE2019_BabySounds_new/src_aae/decoder_params_64_Traindevel_ACD.pkl'))
    #add 0
    x = prior.gaussian_mixture2(aug_num,2,5,0.5,0.1,label_indices=[aug_lab for i in range(aug_num)])
#    bias = np.zeros(x.shape)
#    bias[:,0]= -0.5
#    bias[:,1]= -0.5
#    x = x - bias 
    
    
    plt.figure()
    x0 = plt.scatter(x[:,0],x[:,1],s=5,c='red', alpha = 0.5,label='0')    
    plt.legend(handles=[x0])
    plt.ylim((-3, 3))
    plt.xlim((-3, 3))
    plt.show()
    
    new_x = decoder(torch.from_numpy(x).float())
    new_y = np.array([classes[aug_lab] for i in range(aug_num)])
    X_train_  = np.concatenate((X_train_,new_x.detach().numpy()))
    y_train_ = np.concatenate((y_train_,new_y))
    
    return X_train_ , y_train_


def aug_multi(X_train_,y_train_,aug_num):
    
    decoder = vae.Decoder()
    decoder.load_state_dict(torch.load('/home/gaoyi/Interspeech_challenge/ComParE2019_BabySounds_new/src_aae/decoder_params_64_Traindevel_ACD.pkl'))
    #add 0
    x1 = prior.gaussian_mixture2(aug_num[0],2,5,0.8,0.1,label_indices=[ 0 for i in range(aug_num[0])])
    x2 = prior.gaussian_mixture2(aug_num[1],2,5,0.8,0.1,label_indices=[ 1 for i in range(aug_num[1])])
    x3 = prior.gaussian_mixture2(aug_num[2],2,5,0.8,0.1,label_indices=[ 2 for i in range(aug_num[2])])
    x4 = prior.gaussian_mixture2(aug_num[3],2,5,0.8,0.1,label_indices=[ 3 for i in range(aug_num[3])])
    x5 = prior.gaussian_mixture2(aug_num[4],2,5,0.3,0.5,label_indices=[ 4 for i in range(aug_num[4])])
    #mask
   # x5 =x5[x5[:,0]>0.3]
    
    
#    bias = np.zeros(x4.shape)
#    bias[:,0]= -1.5
#    bias[:,1]= -0.75
#    x4 = x4 - bias 
   
    x = np.concatenate([x1,x2,x3,x4,x5])
    
    
    plt.figure()
    x0 = plt.scatter(x[:,0],x[:,1],s=5,c='red', alpha = 0.5,label='0')    
    plt.legend(handles=[x0])
    plt.ylim((-5, 5))
    plt.xlim((-5, 5))
    plt.show()
    
    new_x = decoder(torch.from_numpy(x).float())
    ls=[]
    for i in zip([0,1,2,3,4],[x1,x2,x3,x4,x5]):
        ls.append([classes[i[0]] for ii in range(len(i[1]))])
    new_y = np.concatenate(ls)
    X_train_  = np.concatenate((X_train_,new_x.detach().numpy()))
    y_train_ = np.concatenate((y_train_,new_y))
    
    return X_train_ , y_train_
    
#%% devel part
tgt_num = [300,0,0,200,800]
all_ls=[]
for aug_lab in [2]:
    num_ls=[]
    for aug_num_ls in [tgt_num]:
        aug_num = [aug_num_ls for i in range(5)]
        X_train3, y_train3 = aug_multi(X_traindevel,y_traindevel,aug_num_ls)
        uar_scores = []
        uar_scores2 = []
        cm=[]
        # Train SVM model with different complexities and evaluate
        for comp in complexities:
            for pp_f in [20,30]:
        #        print('\nComplexity {0:.6f}'.format(comp))
                fs_A = SelectPercentile(score_func=f_classif,percentile=pp_f).fit(X_train3,y_train3)
                X_train2 = fs_A.transform(X_train3)
                #X_devel2 = fs_A.transform(X_devel)
                clf = svm.LinearSVC(C=comp, random_state=0)
                clf.fit(X_train2, y_train3)
                y_pred_train = clf.predict(fs_A.transform(X_traindevel))
                #y_pred = clf.predict(X_devel2)
                uar_scores.append( recall_score(y_traindevel, y_pred_train, labels=classes, average='macro') )
                #uar_scores2.append( recall_score(y_devel, y_pred, labels=classes, average='macro') )
                cm.append(confusion_matrix(y_traindevel, y_pred_train, labels=classes))
        #        print('UAR on Train {0:.1f}'.format(uar_scores[-1]*100))
        #        print('UAR on Devel {0:.1f}'.format(uar_scores2[-1]*100))
        #        if show_confusion:
        #            print('Confusion matrix (Devel):')
        #            print(classes)
        #            print(confusion_matrix(y_devel, y_pred, labels=classes))
            print('finish c={}'.format(comp))
        
        # Train SVM model on the whole training data with optimum complexity and get predictions on test data
        optimum_complexity = complexities[int(np.argmax(uar_scores)/10)]
        print('lab={},num={}'.format(aug_lab,aug_num_ls))
        print('\nOptimum complexity: {0}, maximum UAR on Train:{2} ,Devel {1}\n'.format(optimum_complexity, np.max(uar_scores)*100,uar_scores[np.argmax(uar_scores)]*100))
        print(cm[np.argmax(uar_scores)])
        print(np.argmax(uar_scores)+1)
        
        num_ls.append(uar_scores)
    all_ls.append(num_ls)
    
joblib.dump(all_ls,'/home/gaoyi/Interspeech_challenge/ComParE2019_BabySounds_new/src_aae/result.pkl')

