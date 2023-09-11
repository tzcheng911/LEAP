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
import seaborn as sns
from scipy import stats,signal
from scipy.io import wavfile
import os

def plot_err(group_data,color,t):
    group_avg=np.mean(group_data,axis=0)
   #plt.figure()
    err=np.std(group_data,axis=0)/np.sqrt(group_data.shape[0])
    up=group_avg+err
    lw=group_avg-err
    t=np.linspace(-100,600,3501)
    plt.plot(t,group_avg,color=color)
    plt.fill_between(t,up,lw,color=color,alpha=0.5)

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

#%% calculating group mmr
mmr1_group = []
mmr2_group = []

# block 1 group average
for s in subj:
    std = mne.read_evokeds(root_path + s +'/eeg/cbs_' + s + '_' + run + '_evoked_substd_mmr.fif',allow_maxshield = True)
    dev1 = mne.read_evokeds(root_path + s +'/eeg/cbs_' + s + '_' + run + '_evoked_dev1_mmr.fif',allow_maxshield = True)
    dev2 = mne.read_evokeds(root_path + s +'/eeg/cbs_' + s + '_' + run + '_evoked_dev2_mmr.fif',allow_maxshield = True)
    mmr1 = dev1[0].data - std[0].data
    mmr2 = dev2[0].data - std[0].data
    mmr1_group.append(mmr1)
    mmr2_group.append(mmr2)
    
group_mmr1=np.squeeze(np.asarray(mmr1_group),1)
group_mmr2=np.squeeze(np.asarray(mmr2_group),1)

np.save(root_path + 'meeg_mmr_analysis/group_mmr1_eeg.npy',group_mmr1)
np.save(root_path + 'meeg_mmr_analysis/group_mmr2_eeg.npy',group_mmr2)