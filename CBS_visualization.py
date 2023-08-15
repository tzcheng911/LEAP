#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 16:47:33 2023

@author: tzcheng
"""
## Import library  
import mne
import matplotlib
import numpy as np
import os

########################################
root_path='/media/tzcheng/storage/CBS/'
os.chdir(root_path)

## Visualize epochs
runs = ['_01','_02']

subj = [] 
for file in os.listdir():
    if file.startswith('cbs_A'):
        subj.append(file)

run = runs[0]
s = subj[0]

file_in = root_path + '/' + s + '/sss_fif/' + s
fwd = mne.read_forward_solution(file_in + '-fwd.fif')
cov = mne.read_cov(file_in + run + '_erm_otp_raw_sss_proj_fil50-cov.fif')
epoch = mne.read_epochs(file_in + run + '_otp_raw_sss_proj_fil50_mmr_e.fif')

subject = s
subjects_dir = '/media/tzcheng/storage2/subjects/'

evoked_s = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_fil50_evoked_substd_mmr.fif')[0]
evoked_d1 = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_fil50_evoked_dev1_mmr.fif')[0]
evoked_d2 = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_fil50_evoked_dev2_mmr.fif')[0]

inverse_operator = mne.minimum_norm.make_inverse_operator(epoch.info, fwd, cov,loose=1,depth=0.8)
stc_std = mne.minimum_norm.apply_inverse((evoked_s), inverse_operator)
stc_dev1 = mne.minimum_norm.apply_inverse((evoked_d1), inverse_operator)
stc_dev2 = mne.minimum_norm.apply_inverse((evoked_d2), inverse_operator)
src = inverse_operator['src']

mmr1 = stc_dev1 - stc_std
mmr2 = stc_dev2 - stc_std

mmr1.plot(src,subject=subject, subjects_dir=subjects_dir)
mmr2.plot(src,subject=subject, subjects_dir=subjects_dir)