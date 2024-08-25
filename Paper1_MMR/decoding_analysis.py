#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 12:31:26 2023

@author: tzcheng
"""
import os 
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import time
import copy
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

#%%####################################### MEG decoding sliding estimator
root_path='/home/tzcheng/Documents/GitHub/Paper1_MMR/data/' # change to where you save the npy files
os.chdir(root_path)
## parameters
times = np.linspace(-0.1,0.6,3501)

ts = 0 # -0.1s
te = 3501 # 0.6s

k_feature = 'all' # 'all' or any k features
n_cv = 5

#%%####################################### Load subjects: change file for adults and infants
tic = time.time()

mmr1 = np.load(root_path + 'adults/adult_group_mmr1_mba_vector_morph.npy',allow_pickle=True) 
mmr2 = np.load(root_path + 'adults/adult_group_mmr2_pa_vector_morph.npy',allow_pickle=True)

X = np.concatenate((mmr1,mmr2),axis=0)
X = X[:,:,ts:te] 
y = np.concatenate((np.repeat(0,len(mmr1)),np.repeat(1,len(mmr1)))) #0 is for mmr1 and 1 is for mmr2

# prepare a series of classifier applied at each time sample
clf = make_pipeline(
    StandardScaler(),  # z-score normalization
    SelectKBest(f_classif, k=k_feature),  # select features for speed
    LinearModel(),
    )
time_decod = SlidingEstimator(clf, scoring="roc_auc")

# Run cross-validated decoding analyses
scores_observed = cross_val_multiscore(time_decod, X, y, cv=n_cv, n_jobs=None) 
score = np.mean(scores_observed, axis=0)

time_decod.fit(X, y)

# Retrieve patterns after inversing the z-score normalization step
patterns = get_coef(time_decod, "patterns_", inverse_transform=True)
toc = time.time()
print('It takes ' + str((toc - tic)/60) + 'min to run decoding')

# np.save(root_path + 'roc_auc_kall_replicate_new_acc.npy',scores_observed)
# np.save(root_path + 'patterns_kall_replicate_new_pattern.npy',patterns)

#%%####################################### Run permutation
ind = np.where(scores_observed.mean(axis = 0) > np.percentile(scores_observed.mean(axis = 0),q = 95))
peaks_time =  times[ts:te][ind]

X = np.concatenate((mmr1,mmr2),axis=0)[:,:,ts:te]
X = X[:,:,ind[0]] 
y = np.concatenate((np.repeat(0,len(mmr1)),np.repeat(1,len(mmr1)))) #0 is for mmr1 and 1 is for mmr2

n_perm=1000
scores_perm=[]
for i in range(n_perm):
    print('Iteration' + str(i))
    yp = copy.deepcopy(y)
    random.shuffle(yp)
    clf = make_pipeline(
        StandardScaler(),  # z-score normalization
        SelectKBest(f_classif, k=k_feature),  # select features for speed
        LinearModel(),
        )
    time_decod = SlidingEstimator(clf, scoring="roc_auc",n_jobs=None)
    # Run cross-validated decoding analyses:
    scores = cross_val_multiscore(time_decod, X, yp, cv=5, n_jobs=None)
    scores_perm.append(np.mean(scores,axis=0))
scores_perm_array=np.asarray(scores_perm)
np.savez(root_path + 'roc_auc_1000perm_kall',scores_perm_array =scores_perm_array, peaks_time=peaks_time)

toc = time.time()
print('It takes ' + str((toc - tic)/60) + 'min to run 100 iterations of kall decoding')