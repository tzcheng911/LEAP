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

## Visualize epochs
subjects_dir = '/media/tzcheng/storage2/subjects/'

subject = 'vMMR_901'
run = '_3'

root_path = '/media/tzcheng/storage/vmmr/'
fwd = mne.read_forward_solution(root_path + subject + '/source/'+ subject + '_1_fwd.fif')
cov = mne.read_cov(root_path + subject + '/sss_fif/' + subject + run + '_erm_raw_sss_proj_fil50-cov')
epoch = mne.read_epochs(root_path + subject + '/sss_fif/' + subject + run + '_raw_sss_proj_fil50_e.fif')

evoked_s = mne.read_evokeds(root_path + subject + '/sss_fif/' + subject + run + '_raw_sss_proj_fil50_evoked_s.fif')[0]
evoked_d = mne.read_evokeds(root_path + subject + '/sss_fif/' + subject + run + '_raw_sss_proj_fil50_evoked_d.fif')[0]

# mne.viz.plot_compare_evokeds(evoked_s, picks=["MEG0721"], combine="mean")
# mne.viz.plot_compare_evokeds(evoked_s, picks="meg", axes="topo") # plot all of them
# epoch.plot_sensors(kind='3d', ch_type='mag', ch_groups='position')
# chs = ["MEG0721","MEG0631","MEG0741","MEG1821"]
# mne.viz.plot_compare_evokeds(evoked_s, picks=chs, combine="mean", show_sensors="upper right")

inverse_operator = mne.minimum_norm.make_inverse_operator(epoch.info, fwd, cov,loose=0.2,depth=0.8)
standard = mne.minimum_norm.apply_inverse((evoked_s), inverse_operator)
deviant = mne.minimum_norm.apply_inverse((evoked_d), inverse_operator)

mmr = deviant - standard

# standard.plot(subject=subject, subjects_dir=subjects_dir, hemi='both')
# deviant.plot(subject=subject, subjects_dir=subjects_dir, hemi='both')
mmr.plot(subject='sample', subjects_dir=subjects_dir, hemi='both')