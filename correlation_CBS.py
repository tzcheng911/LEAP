#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 13:23:58 2023
1. calculate EEG & MEG correlation in adults
2. calculate correlation between adult and baby MEG
@author: tzcheng
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy import stats,signal
import os

root_path='/media/tzcheng/storage/CBS/'
subjects_dir = '/media/tzcheng/storage2/subjects/'

#%%######################################## load MEG
stc1 = mne.read_source_estimate(root_path + 'cbs_A101/sss_fif/cbs_A101_mmr2_morph-vl.stc')
stc2 = mne.read_source_estimate(root_path + 'cbs_A101/sss_fif/cbs_A101_mmr2_morph-vl.stc')
MEG_mmr1 = np.load(root_path + 'meeg_mmr_analysis/group_mmr1_vector_morph.npy')
MEG_mmr2 = np.load(root_path + 'meeg_mmr_analysis/group_mmr2_vector_morph.npy')
stc1.data = MEG_mmr1.mean(axis=0)
stc2.data = MEG_mmr2.mean(axis=0)

subject = 'fsaverage'
src = mne.read_source_spaces(subjects_dir + 'fsaverage/bem/fsaverage-vol-5-src.fif')

# stc.plot(src,clim=dict(kind="value",pos_lims=[0,2,5]),subject=subject, subjects_dir=subjects_dir)
stc1.plot(src,subject=subject, subjects_dir=subjects_dir)
stc2.plot(src,subject=subject, subjects_dir=subjects_dir)

#%%######################################## load EEG
EEG_mmr1 = np.load(root_path + 'meeg_mmr_analysis/group_mmr1_eeg.npy')
EEG_mmr2 = np.load(root_path + 'meeg_mmr_analysis/group_mmr2_eeg.npy')

#%%######################################## corr across EEG and MEG time series averaged across subjects
mean_EEG_mmr2 = EEG_mmr2.mean(axis =0)
mean_MEG_mmr2 = MEG_mmr2.mean(axis =0).mean(axis=0)


corr_p = pearsonr(mean_EEG_mmr2[501:2000], mean_MEG_mmr2[501:2000]) # 0 ms to 200 ms
xcorr = signal.correlate(mean_EEG_mmr2,mean_MEG_mmr2)
lags = signal.correlation_lags(len(mean_EEG_mmr2),len(mean_MEG_mmr2))
xcorr = np.abs(xcorr)

fig,(ax1,ax2,ax3) = plt.subplots(3,1)
ax1.plot(stc1.times,mean_EEG_mmr2)
ax2.plot(stc1.times,mean_MEG_mmr2)
ax3.plot(lags,xcorr)

max_xcorr = np.max(xcorr)
max_xcorr_ind = np.argmax(xcorr)
stc1.times[max_xcorr_ind - len(stc1.times)] # the time when the max corr happened