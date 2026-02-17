#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 12:09:14 2026

@author: tzcheng
"""

#%%####################################### Import library  
import mne
import os
import math
import matplotlib.pyplot as plt
from scipy import stats, signal
from scipy.signal import butter, filtfilt, hilbert
from scipy.io import savemat
import numpy as np
from scipy.io import wavfile
import time
from mne.decoding import (
    cross_val_multiscore,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import random

#%%####################################### Define functions

def do_brainstem_trial_by_trial_decoding(root_path,n_trials='ntrial_all'):
    ## Trial-by-trial decoding for spa vs. eng: not working well
    subjects_eng=['104','106','107','108','110','111','112','113','118','121','123','124','126','129','133']
    subjects_spa=['203','204','205','206','211','212','213','214','215','220','221','222','223','224','225'] # trim 226 so be n = 15 too
    # subjects_spa=['203','204','205','206','211','212','213','214','215','220','221','222','223','224','225','226'] 
    
    ## SVM showed higher accuracy in trial-by-trial decoding
    clf = make_pipeline(
        StandardScaler(),
        # SVC(kernel='rbf',gamma='auto')  
        SVC(kernel='linear', C=1,class_weight='balanced')
       )
    
    all_score_svm = []

    for se, sp in zip(subjects_eng,subjects_spa):
        print("Trial-by-trial decoding for subj " + se)
        ## do eng first because they are the focus 
        epochs_eng_p10 = mne.read_epochs(root_path + 'preprocessed/' + n_trials + '/eng/brainstem_' + se  +'_p10_01_cabr_e_all.fif').pick_types(eeg=True, exclude=[])
        epochs_eng_n40 = mne.read_epochs(root_path + 'preprocessed/' + n_trials + '/eng/brainstem_' + se  +'_n40_01_cabr_e_all.fif').pick_types(eeg=True, exclude=[])
        
        ## could do spa too
        epochs_spa_p10 = mne.read_epochs(root_path + 'preprocessed/' + n_trials + '/spa/brainstem_' + sp  +'_p10_01_cabr_e_all.fif').pick_types(eeg=True, exclude=[])
        epochs_spa_n40 = mne.read_epochs(root_path + 'preprocessed/' + n_trials + '/spa/brainstem_' + sp  +'_n40_01_cabr_e_all.fif').pick_types(eeg=True, exclude=[])
        
        X_p10 = np.squeeze(epochs_eng_p10.get_data())
        X_n40 = np.squeeze(epochs_eng_n40.get_data())

        X = np.concatenate((X_p10,X_n40),axis=0)
        y = np.concatenate((np.repeat(0,len(X_p10)),np.repeat(1,len(X_n40))))
                       
        scores = cross_val_multiscore(clf, X, y, cv=5, n_jobs=4) 
        score = np.mean(scores, axis=0)
        print("Trial-by-trial decoding accuracy: %0.1f%%" % (100 * score,))
        all_score_svm.append(score)
    return all_score_svm

#%%####################################### Run the functions
root_path= '/media/tzcheng/storage/Brainstem/EEG/'
trial_by_trial_decoding_acc = do_brainstem_trial_by_trial_decoding(root_path) 
np.save(root_path + 'trial_by_trial_acc_eng_p10n40_ntrial_all.npy',trial_by_trial_decoding_acc)
