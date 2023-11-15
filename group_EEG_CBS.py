#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 13:13:28 2023
See Christina eeg_group
@author: tzcheng
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats,signal
from scipy.io import wavfile
import os

#%%########################################
root_path='/media/tzcheng/storage/CBS/'
subjects_dir = '/media/tzcheng/storage2/subjects/'

os.chdir(root_path)

## parameters 
subj = [] # A104 got some technical issue
for file in os.listdir():
    if file.startswith('cbs_A'):
        subj.append(file)

runs = ['01','02']
run = runs [0]

#%% output the time series in npy files
group_mmr1 = []
group_mmr2 = []
group_std = []
group_dev = []
group_dev1 = []
group_dev2 = []    
group_std1 = []
group_std2 = []
# run 1 group average
for s in subj:

    ## Could do MMR or cABR
    # for the normal direction
    std = mne.read_evokeds(root_path + s +'/eeg/' + 'cbs_' + s + '_' + run + '_evoked_substd_cabr.fif',allow_maxshield = True)[0]
    dev1 = mne.read_evokeds(root_path + s +'/eeg/' + 'cbs_' + s + '_' + run + '_evoked_dev1_cabr.fif',allow_maxshield = True)[0]
    dev2 = mne.read_evokeds(root_path + s +'/eeg/' + 'cbs_' + s + '_' + run + '_evoked_dev2_cabr.fif',allow_maxshield = True)[0]
    # mmr1 = dev1.data - std.data
    # mmr2 = dev2.data - std.data

    # for the reverse
    # std1 = mne.read_evokeds(root_path + s +'/eeg/' + s + '_' + run + '_evoked_substd1_reverse_mmr.fif',allow_maxshield = True)[0]
    # std2 = mne.read_evokeds(root_path + s +'/eeg/' + s + '_' + run + '_evoked_substd2_reverse_mmr.fif',allow_maxshield = True)[0]
    # dev = mne.read_evokeds(root_path + s +'/eeg/' + s + '_' + run + '_evoked_dev_reverse_mmr.fif',allow_maxshield = True)[0]
    # mmr1 = dev.data - std1.data
    # mmr2 = dev.data - std2.data

    # for the first and last /ba/
    # std = mne.read_evokeds(root_path + s +'/eeg/' + s + '_' + run + '_evoked_substd_ba_mmr.fif',allow_maxshield = True)[0]
    # dev1 = mne.read_evokeds(root_path + s +'/eeg/' + s + '_' + run + '_evoked_dev1_ba_mmr.fif',allow_maxshield = True)[0]
    # dev2 = mne.read_evokeds(root_path + s +'/eeg/' + s + '_' + run + '_evoked_dev2_ba_mmr.fif',allow_maxshield = True)[0]
    # mmr1 = dev1.data - std.data
    # mmr2 = dev2.data - std.data

    # group_mmr1.append(mmr1)
    # group_mmr2.append(mmr2)
    # group_std1.append(std1.data)
    # group_std2.append(std2.data)
    group_dev1.append(dev1.data)
    group_dev2.append(dev2.data)
    group_std.append(std.data)
    # group_dev.append(dev.data)


group_std = np.squeeze(np.asarray(group_std),1)   
group_dev1 = np.squeeze(np.asarray(group_dev1),1)   
group_dev2 = np.squeeze(np.asarray(group_dev2),1)
# group_std1 = np.squeeze(np.asarray(group_std1),1)   
# group_std2 = np.squeeze(np.asarray(group_std2),1)      
# group_mmr1 = np.squeeze(np.asarray(group_mmr1),1)
# group_mmr2 = np.squeeze(np.asarray(group_mmr2),1)
# group_dev = np.squeeze(np.asarray(group_dev),1)
# group_std = np.squeeze(np.asarray(group_std),1)  

# np.save(root_path + 'cbsA_meeg_analysis/group_std_ba_eeg.npy',group_std)
# np.save(root_path + 'cbsA_meeg_analysis/group_mmr1_eeg.npy',group_mmr1)
# np.save(root_path + 'cbsA_meeg_analysis/group_mmr2_eeg.npy',group_mmr2)
# np.save(root_path + 'cbsA_meeg_analysis/group_std1_reverse_eeg.npy',group_std1)
# np.save(root_path + 'cbsA_meeg_analysis/group_std2_reverse_eeg.npy',group_std2)
np.save(root_path + 'cbsA_meeg_analysis/group_std_cabr_eeg.npy',group_std)
np.save(root_path + 'cbsA_meeg_analysis/group_dev1_cabr_eeg.npy',group_dev1)
np.save(root_path + 'cbsA_meeg_analysis/group_dev2_cabr_eeg.npy',group_dev2)
# np.save(root_path + 'cbsA_meeg_analysis/group_dev1_ba_eeg.npy',group_dev1)
# np.save(root_path + 'cbsA_meeg_analysis/group_dev2_ba_eeg.npy',group_dev2)
# np.save(root_path + 'cbsA_meeg_analysis/group_dev_reverse_eeg.npy',group_dev)