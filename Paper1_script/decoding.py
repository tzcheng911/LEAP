#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 12:31:26 2023
@author: tzcheng
""" 
import numpy as np
import copy
import random
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from mne.decoding import (
    cross_val_multiscore,
    LinearModel,
    get_coef,
)
#%%####################################### decoding for single channel EEG
root_path='/media/tzcheng/storage/CBS/'

## MMR
ts = 500
te = 1750

std = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_std_eeg.npy')
dev1 = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_dev1_eeg.npy')
dev2 = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_dev2_eeg.npy')

MMR1 = dev1 - std
MMR2 = dev2 - std

X = np.concatenate((MMR1,MMR2),axis=0)
X = X[:,ts:te]
y = np.concatenate((np.repeat(0,len(dev1)),np.repeat(1,len(dev2))))

clf = make_pipeline(
    StandardScaler(),  # z-score normalization
    LogisticRegression(solver="liblinear")  # liblinear is faster than lbfgs
)
scores = cross_val_multiscore(clf, X, y, cv=5, n_jobs=None) # takes about 10 mins to run
score = np.mean(scores, axis=0)
print("Accuracy: %0.1f%%" % (100 * score,))

## cABR 
std = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_std_cabr_eeg_200.npy')
dev1 = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_dev1_cabr_eeg_200.npy')
dev2 = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_dev2_cabr_eeg_200.npy')

X = np.concatenate((std,dev1,dev2),axis=0)
y = np.concatenate((np.repeat(0,len(std)),np.repeat(1,len(dev1)),np.repeat(2,len(dev2))))

clf = make_pipeline(
    StandardScaler(),  # z-score normalization
    LogisticRegression(solver="liblinear")  # liblinear is faster than lbfgs
)
scores = cross_val_multiscore(clf, X, y, cv=5, n_jobs=None) 
score = np.mean(scores, axis=0)
print("Accuracy: %0.1f%%" % (100 * score,))

## Run permutation on MMR or cABR
n_perm=500
scores_perm=[]
for i in range(n_perm):
    yp = copy.deepcopy(y)
    random.shuffle(yp)
    clf = make_pipeline(
        StandardScaler(),  # z-score normalization
        LogisticRegression(solver="liblinear")  
        )
    # Run cross-validated decoding analyses:
    scores = cross_val_multiscore(clf, X, yp, cv=5, n_jobs=None) # X can be MMR or cABR
    scores_perm.append(np.mean(scores,axis=0))
    print("Iteration " + str(i))
scores_perm_array=np.asarray(scores_perm)

plt.figure()
plt.hist(scores_perm_array,bins=30,color='k')
plt.vlines(score,ymin=0,ymax=12,color='r',linewidth=2)
plt.vlines(np.percentile(scores_perm_array,97.5),ymin=0,ymax=12,color='grey',linewidth=2)
plt.ylabel('Count',fontsize=20)
plt.xlabel('Accuracy',fontsize=20)
