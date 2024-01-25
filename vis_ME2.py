#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 11:39:32 2024

@author: tzcheng
"""
## Import library  
import matplotlib.pyplot as plt
import numpy as np
import os 

import mne
from mne.datasets import somato
from mne.time_frequency import tfr_morlet

#%%#######################################   
tmin = 1.0
tmax = 9.0
fmin = 0.5
fmax = 5

age = '7mo/'
runs = ['_01','_02','_03','_04']
root_path='/media/tzcheng/storage/ME2_MEG/Zoe_analyses/'
os.chdir(root_path + age)

subjects = []

for file in os.listdir():
    if file.startswith('me2_'): 
        subjects.append(file)

#%%####################################### visualize individual sensor and source
subj = subjects[2]
for run in runs: 
    epochs = mne.read_epochs(root_path + age + subj + '/sss_fif/' + subj + run + '_otp_raw_sss_proj_fil50_epoch.fif')
    evoked = epochs['Trial_Onset'].average()
    sfreq = epochs.info["sfreq"]
    # epochs.compute_psd(fmin=0.5, fmax=15.0, tmin= 0, tmax = 10).plot(picks="data", exclude="bads")
    # evoked.compute_psd(fmin=0.5, fmax=5.0, tmin= 0, tmax = 10).plot(picks="data", exclude="bads")
    # evoked.compute_psd(fmin=fmin, fmax=fmax, tmin= tmin, tmax = tmax).plot(average = True,picks="data", exclude="bads")
    evoked.compute_psd("welch",
    n_fft=int(sfreq * (tmax - tmin)),
    n_overlap=0,
    n_per_seg=None,
    tmin=tmin,
    tmax=tmax,
    fmin=fmin,
    fmax=fmax,
    window="boxcar",
    verbose=False,).plot(average=True,picks="data", exclude="bads")