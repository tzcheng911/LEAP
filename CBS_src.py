#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 15:01:42 2023
Used for looping subjects to save their fwd and stc + morphed data
BEM, src and trans files should be saved from the coreg process done manually
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

runs = ['_01','_02']
subj = [] 
for file in os.listdir():
    if file.startswith('cbs_A'):
        subj.append(file)

for s in subj:
    for run in runs:
        print(s)
        file_in = root_path + '/' + s + '/sss_fif/' + s
        fwd = mne.read_forward_solution(file_in + '-fwd.fif')
        trans = mne.read_trans(file_in +'-trans.fif')
        cov = mne.read_cov(file_in + run + '_erm_otp_raw_sss_proj_fil50-cov.fif')
        epoch = mne.read_epochs(file_in + run + '_otp_raw_sss_proj_fil50_mmr_e.fif')
         
        subject = s
        subjects_dir = '/media/tzcheng/storage2/subjects/'
        
        evoked_s = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_fil50_evoked_substd_mmr.fif')[0]
        evoked_d1 = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_fil50_evoked_dev1_mmr.fif')[0]
        evoked_d2 = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_fil50_evoked_dev2_mmr.fif')[0]
        
        # mne.viz.plot_compare_evokeds(evoked_s, picks=["MEG0721"], combine="mean")
        # mne.viz.plot_compare_evokeds(evoked_s, picks="meg", axes="topo") # plot all of them
        # epoch.plot_sensors(kind='3d', ch_type='mag', ch_groups='position')
        # chs = ["MEG0721","MEG0631","MEG0741","MEG1821"]
        # mne.viz.plot_compare_evokeds(evoked_s, picks=chs, combine="mean", show_sensors="upper right")
        
        inverse_operator = mne.minimum_norm.make_inverse_operator(epoch.info, fwd, cov,loose=1,depth=0.8)
        standard = mne.minimum_norm.apply_inverse((evoked_s), inverse_operator)
        dev1 = mne.minimum_norm.apply_inverse((evoked_d1), inverse_operator)
        dev2 = mne.minimum_norm.apply_inverse((evoked_d2), inverse_operator)
        src = inverse_operator['src']
        
        standard.save(file_in + '_substd_mmr', overwrite=True)
        dev1.save(file_in + '_dev1_mmr', overwrite=True)
        dev2.save(file_in + '_dev2_mmr', overwrite=True)
        src.save(file_in + '_src', overwrite=True)
      
        mmr1 = dev1 - standard
        mmr2 = dev2 - standard
        mmr1.save(file_in + '_mmr1', overwrite=True)
        mmr2.save(file_in + '_mmr2', overwrite=True)
    
    
    # mmr1.plot(src,subject=subject, subjects_dir=subjects_dir)
    # mmr2.plot(src,subject=subject, subjects_dir=subjects_dir)

# deviant.plot(subject=subject, subjects_dir=subjects_dir, hemi='both')
# mmr.plot(subject=subject, subjects_dir=subjects_dir, hemi='both')