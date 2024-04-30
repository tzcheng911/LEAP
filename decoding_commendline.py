#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 10:20:59 2023
Run decoding in terminal. Saved the accuracy and patterns.


@author: tzcheng
"""
import os 
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
import random
import mne
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne.preprocessing import Xdawn
from mne.decoding import (
    SlidingEstimator,
    GeneralizingEstimator,
    Scaler,
    cross_val_multiscore,
    LinearModel,
    get_coef,
    Vectorizer,
    CSP,
)

#%%####################################### Subject-by-subject decoding for each condition 
tic = time.time()
root_path='/media/tzcheng/storage2/CBS/'
subjects_dir = '/media/tzcheng/storage2/subjects/'
os.chdir(root_path)

stc1 = mne.read_source_estimate(root_path + 'cbs_b101/sss_fif/cbs_b101_mmr2_vector_morph-vl.stc')
times = stc1.times

## parameters
ts = 250 # -0.05s
te = 2750 # 0.45s
ROI_wholebrain = 'wholebrain' # ROI or wholebrain or sensor
k_feature = 'all' # ROI: 'all' features; whole brain: 500 features
n_cv = 5 # number of folds in SKfold
#%%####################################### Load adults
filename = 'vector'
filename_mmr1 = 'group_mmr1_mba_vector_morph'
filename_mmr2 = 'group_mmr2_pa_vector_morph'

fname_aseg = subjects_dir + 'fsaverage/mri/aparc+aseg.mgz'
label_names = np.asarray(mne.get_volume_labels_from_aseg(fname_aseg))
lh_ROI_label = [72,60,61,62] # STG and frontal pole
rh_ROI_label = [108,96,97,98] # STG and IFG (parsopercularis, parsorbitalis, parstriangularis)

if ROI_wholebrain == 'ROI':
    mmr1 = np.load(root_path + 'cbsb_meg_analysis/MEG/MMR/' + filename_mmr1 + '_roi.npy',allow_pickle=True)
    mmr2 = np.load(root_path + 'cbsb_meg_analysis/MEG/MMR/' + filename_mmr2 + '_roi.npy',allow_pickle=True)
elif ROI_wholebrain == 'wholebrain':
    mmr1 = np.load(root_path + 'cbsb_meg_analysis/MEG/MMR/' + filename_mmr1 + '.npy',allow_pickle=True)
    mmr2 = np.load(root_path + 'cbsb_meg_analysis/MEG/MMR/' + filename_mmr2 + '.npy',allow_pickle=True)
else:
    print("Need to decide whether to use ROI or whole brain as feature.")


# ## preserve the subject order
rand_ind = np.arange(0,len(mmr1))
random.Random(0).shuffle(rand_ind)
X = np.concatenate((mmr1[rand_ind,:],mmr2[rand_ind,:]),axis=0)

X = np.concatenate((mmr1,mmr2),axis=0)
X = X[:,:,ts:te] 
y = np.concatenate((np.repeat(0,len(mmr1)),np.repeat(1,len(mmr1)))) #0 is for mmr1 and 1 is for mmr2

# ## random shuffle X and y
# # rand_ind = np.arange(0,len(X))
# # random.Random(1).shuffle(rand_ind)
# # X = X[rand_ind,:,:]
# # y = y[rand_ind]

# # prepare a series of classifier applied at each time sample
# clf = make_pipeline(
#     StandardScaler(),  # z-score normalization
#     SelectKBest(f_classif, k=k_feature),  # select features for speed
#     LinearModel(),
#     )
# time_decod = SlidingEstimator(clf, scoring="roc_auc")
    
# # Run cross-validated decoding analyses

# scores_observed = cross_val_multiscore(time_decod, X, y, cv=n_cv,n_jobs=2)
# score = np.mean(scores_observed, axis=0)

# time_decod.fit(X, y) # not changed after shuffling the initial
# # # Retrieve patterns after inversing the z-score normalization step:
# patterns = get_coef(time_decod, "patterns_", inverse_transform=True)

# toc = time.time()
# print('It takes ' + str((toc - tic)/60) + 'min to run decoding')

# np.save(root_path + 'cbsb_meg_analysis/decoding/baby_roc_auc_' + filename + '_morph_kall_mba_pa.npy',scores_observed)
# np.save(root_path + 'cbsb_meg_analysis/decoding/baby_patterns_' + filename + '_morph_kall_mba_pa.npy', patterns)

# #%%####################################### Run permutation
# filename = 'vector'
# filename_mmr1 = 'group_mmr1_mba_vector_morph'
# filename_mmr2 = 'group_mmr2_pa_vector_morph'

# scores_observed = np.load(root_path + '/cbsb_meg_analysis/decoding/baby_roc_auc_' + filename + '_morph_kall_mba_pa.npy') # only get the -0.05 to 0.45 s window
# ind = np.where(scores_observed.mean(axis = 0) > np.percentile(scores_observed.mean(axis = 0),q = 95))
# peaks_time =  times[ts:te][ind]

# if ROI_wholebrain == 'ROI':
#     mmr1 = np.load(root_path + 'cbsb_meg_analysis/MEG' + filename_mmr1 + '_roi.npy',allow_pickle=True)
#     mmr2 = np.load(root_path + 'cbsb_meg_analysis/MEG' + filename_mmr2 + '_roi.npy',allow_pickle=True)
# elif ROI_wholebrain == 'wholebrain':
#     mmr1 = np.load(root_path + 'cbsb_meg_analysis/MEG/' + filename_mmr1 + '.npy',allow_pickle=True)
#     mmr2 = np.load(root_path + 'cbsb_meg_analysis/MEG/' + filename_mmr2 + '.npy',allow_pickle=True)
# else:
#     print("Need to decide whether to use ROI or whole brain as feature.")
# X = np.concatenate((mmr1,mmr2),axis=0)[:,:,ts:te]
# X = X[:,:,ind[0]] 
# y = np.concatenate((np.repeat(0,len(mmr1)),np.repeat(1,len(mmr1)))) #0 is for mmr1 and 1 is for mmr2

# import copy
# import random
# n_perm=100
# scores_perm=[]
# for i in range(n_perm):
#     print('Iteration' + str(i))
#     yp = copy.deepcopy(y)
#     random.shuffle(yp)
#     clf = make_pipeline(
#         StandardScaler(),  # z-score normalization
#         SelectKBest(f_classif, k=k_feature),  # select features for speed
#         LinearModel(),
#         )
#     time_decod = SlidingEstimator(clf, scoring="roc_auc",n_jobs=6)
#     # Run cross-validated decoding analyses:
#     scores = cross_val_multiscore(time_decod, X, yp, cv=5, n_jobs=6)
#     scores_perm.append(np.mean(scores,axis=0))
# scores_perm_array=np.asarray(scores_perm)
# np.savez(root_path + 'cbsb_meg_analysis/baby_' + filename + '_scores_' + str(n_perm) +'perm_kall_new',scores_perm_array =scores_perm_array, peaks_time=peaks_time)

# toc = time.time()
# print('It takes ' + str((toc - tic)/60) + 'min to run 100 iterations of kall decoding')
