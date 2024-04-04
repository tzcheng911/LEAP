#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:41:57 2024

@author: tzcheng
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats,signal
import os

#%%#######################################   
tmin = 1.0
tmax = 9.0
fmin = 0.5
fmax = 5

age = '7mo/' # '11mo/' 
runs = ['_01','_02','_03','_04']
root_path='/media/tzcheng/storage/ME2_MEG/Zoe_analyses/' # /media/tzcheng/storage/BabyRhythm/ for adults
subjects_dir = '/media/tzcheng/storage2/subjects/'

os.chdir(root_path + age)

subjects = []

for file in os.listdir():
    if file.startswith('me2_'): 
        subjects.append(file)


#%% output the sensor time series in npy files
group = []

for s in subjects:
    print('Extracting ' + s + ' data')
    for run in runs:
        file_in = root_path + s + '/sss_fif/' + s + run
        evoked = mne.read_evokeds(file_in + '_otp_raw_sss_proj_fil50_evoked')[0]
    group.append(evoked.data)
    group=np.asarray(group)
    np.save('/media/tzcheng/storage/ME2_MEG/Zoe_analyses/me2_meg_analysis/group' + run + '_sensor.npy',group)

#%% output the source time series in npy files
group_stc_lcmv = []
group_stc_mne = []
group_stc_lcmv_roi = []
group_stc_mne_roi = []

src = mne.read_source_spaces('/media/tzcheng/storage2/subjects/fsaverage/bem/fsaverage-vol-5-src.fif') # for morphing data
fname_aseg = subjects_dir + 'fsaverage' + '/mri/aparc+aseg.mgz'

for s in subjects:
    print('Extracting ' + s + ' data')
    for run in runs:
        file_in = root_path + s + '/sss_fif/' + s + run    
        stc_lcmv = mne.read_source_estimate(file_in+'_stc_lcmv_morph-vl.stc')
        stc_mne = mne.read_source_estimate(file_in+'_stc_mne_morph-vl.stc')
        group_stc_lcmv.append(stc_lcmv.data)
        group_stc_mne.append(stc_mne.data)
        
        label_names = mne.get_volume_labels_from_aseg(fname_aseg)
        stc_lcmv_roi = mne.extract_label_time_course(stc_lcmv,fname_aseg,src,mode='mean',allow_empty=True)
        stc_mne_roi = mne.extract_label_time_course(stc_mne,fname_aseg,src,mode='mean',allow_empty=True)
        group_stc_lcmv_roi.append(stc_lcmv_roi)
        group_stc_mne_roi.append(stc_mne_roi)
    
        group_stc_lcmv = np.asarray(group_stc_lcmv)
        group_stc_mne = np.asarray(group_stc_mne)
        group_stc_lcmv_roi = np.asarray(group_stc_lcmv_roi)
        group_stc_mne_roi = np.asarray(group_stc_mne_roi)
        
        np.save('/media/tzcheng/storage/ME2_MEG/Zoe_analyses/me2_meg_analysis/group' + run + '_stc_lcmv.npy',group_stc_lcmv)
        np.save('/media/tzcheng/storage/ME2_MEG/Zoe_analyses/me2_meg_analysis/group' + run + '_stc_mne.npy',group_stc_mne)
        np.save('/media/tzcheng/storage/ME2_MEG/Zoe_analyses/me2_meg_analysis/group' + run + '_stc_lcmv_roi.npy',group_stc_lcmv_roi)
        np.save('/media/tzcheng/storage/ME2_MEG/Zoe_analyses/me2_meg_analysis/group' + run + '_stc_mne_roi.npy',group_stc_mne_roi)

