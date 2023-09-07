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

root_path='/media/tzcheng/storage/CBS/'
os.chdir(root_path)
subjects_dir = '/media/tzcheng/storage2/subjects/'

runs = ['_01','_02']

subj = [] 
for file in os.listdir():
    if file.startswith('cbs_A'):
        subj.append(file)

run = runs[0]
s = subj[2]

subject = s

########################################
# visualize stc
stc = mne.read_source_estimate(root_path + s + '/sss_fif/' + s + '_mmr2_morph-vl.stc')
src = mne.read_source_spaces(subjects_dir + '/' + s + '/bem/' + s + '-vol-5-src.fif')
# src = mne.read_source_spaces(root_path + s + '/sss_fif/' + s +'_src')  # similar
src = mne.read_source_spaces('/media/tzcheng/storage2/subjects/fsaverage/bem/fsaverage-vol-5-src.fif')

initial_time = 0.1
brain = stc.plot(
    src,
    subject='fsaverage',
    subjects_dir=subjects_dir,
    initial_time=initial_time,
    mode='glass_brain'
)

########################################
# compute and visualize stc and sensor level
file_in = root_path + '/' + s + '/sss_fif/' + s
fwd = mne.read_forward_solution(file_in + '-fwd.fif')
cov = mne.read_cov(file_in + run + '_erm_otp_raw_sss_proj_fil50-cov.fif')

## Cortical
epoch = mne.read_epochs(file_in + run + '_otp_raw_sss_proj_fil50_mmr_e.fif')

evoked_s = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_fil50_evoked_substd_mmr.fif')[0]
evoked_d1 = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_fil50_evoked_dev1_mmr.fif')[0]
evoked_d2 = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_fil50_evoked_dev2_mmr.fif')[0]

## Visualize sensor level
mne.viz.plot_compare_evokeds(evoked_s, picks=["MEG0721"], combine="mean")
mne.viz.plot_compare_evokeds(evoked_s, picks="meg", axes="topo") # plot all of them
epoch.plot_sensors(kind='3d', ch_type='mag', ch_groups='position')
chs = ["MEG0721","MEG0631","MEG0741","MEG1821"]
mne.viz.plot_compare_evokeds(evoked_s, picks=chs, combine="mean", show_sensors="upper right")

## Visualize source level
inverse_operator = mne.minimum_norm.make_inverse_operator(epoch.info, fwd, cov,loose=1,depth=0.8)
stc_std = mne.minimum_norm.apply_inverse((evoked_s), inverse_operator, pick_ori='vector')
stc_dev1 = mne.minimum_norm.apply_inverse((evoked_d1), inverse_operator, pick_ori='vector')
stc_dev2 = mne.minimum_norm.apply_inverse((evoked_d2), inverse_operator, pick_ori='vector')
src = inverse_operator['src']

mmr1 = stc_dev1 - stc_std
mmr2 = stc_dev2 - stc_std

mmr1.plot(src,subject=subject, subjects_dir=subjects_dir)
mmr2.plot(src,subject=subject, subjects_dir=subjects_dir)

## Subcortical
epoch = mne.read_epochs(file_in + run + '_epochs_subcortical.fif')

evoked_s = mne.read_evokeds(file_in + run + '_evoked_substd_cabr.fif')[0]
evoked_d1 = mne.read_evokeds(file_in + run + '_evoked_dev1_cabr.fif')[0]
evoked_d2 = mne.read_evokeds(file_in + run + '_evoked_dev2_cabr.fif')[0]

evoked_s.crop(tmin=-0.1, tmax=0.2)

mne.viz.plot_compare_evokeds(evoked_s, picks="meg", axes="topo") # plot all of them

