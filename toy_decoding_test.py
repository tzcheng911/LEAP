#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 10:48:22 2023
Run on terminal.

@author: tzcheng
"""
import os 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import mne
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
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
#%%####################################### Trial-by-trial decoding for each individual

root_path='/media/tzcheng/storage/CBS/'
epochs = mne.read_epochs(root_path+'cbs_A111/sss_fif/cbs_A111_01_otp_raw_sss_proj_fil50_mmr_e.fif')
epochs.crop(0,0.3)
epochs_d = epochs["Deviant1","Deviant2"]
X = epochs_d.get_data()  # MEG signals features: n_epochs, n_meg_channels, n_times make sure the first dimension is epoch
y = epochs_d.events[:, 2]  # target: standard, deviant1 and 2

## SlidingEstimator
clf = make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear"))
time_decod = SlidingEstimator(clf, n_jobs=None, scoring="roc_auc", verbose=True)
scores = cross_val_multiscore(time_decod, X, y, cv=3, n_jobs=None)
print("SlidingEstimator: %0.1f%%" % (100 * np.mean(scores, axis=0),))