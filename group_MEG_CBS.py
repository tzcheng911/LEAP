#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 17:57:34 2023

@author: tzcheng
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats,signal
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

#%% output the sensor time series in npy files
group_mmr1=[]
group_mmr2=[]

for s in subj:
    print('Extracting ' + s + ' data')
    file_in = root_path + s + '/sss_fif/' + s
    
    # stc_std=mne.read_source_estimate(file_in+'_std_vector_morph-vl.stc')
    dev1=mne.read_evokeds(file_in+'_01_otp_raw_sss_proj_fil50_evoked_dev1_mmr.fif')[0]
    dev2=mne.read_evokeds(file_in+'_01_otp_raw_sss_proj_fil50_evoked_dev2_mmr.fif')[0]
    std=mne.read_evokeds(file_in+'_01_otp_raw_sss_proj_fil50_evoked_substd_mmr.fif')[0]
   
    mmr1 = dev1.data - std.data
    mmr2 = dev2.data - std.data
    
    # group_std.append(stc_std.data)
    group_mmr1.append(mmr1)
    group_mmr2.append(mmr2)
    
group_mmr1=np.asarray(group_mmr1)
group_mmr2=np.asarray(group_mmr2)
np.save(root_path + 'cbsA_meeg_analysis/group_mmr1_sensor.npy',group_mmr1)
np.save(root_path + 'cbsA_meeg_analysis/group_mmr2_sensor.npy',group_mmr2)


#%% output the source time series in npy files
group_mmr1 = []
group_mmr2 = []
group_mmr1_roi = []
group_mmr2_roi = []
group_std = []
group_std1 = []
group_std2 = []
group_std1_roi = []
group_std2_roi = []
group_dev1 = []
group_dev2 = []
group_std_roi = []
group_dev1_roi = []
group_dev2_roi = []

# #extract ROIS for morphing data
src = mne.read_source_spaces('/media/tzcheng/storage2/subjects/fsaverage/bem/fsaverage-vol-5-src.fif') # for morphing data
fname_aseg = subjects_dir + 'fsaverage' + '/mri/aparc+aseg.mgz'

for s in subj:
    print('Extracting ' + s + ' data')
    file_in = root_path + s + '/sss_fif/' + s
    
    # stc_std=mne.read_source_estimate(file_in+'_std_None_morph-vl.stc')
    # stc_dev1=mne.read_source_estimate(file_in+'_dev1_None_morph-vl.stc')
    # stc_dev2=mne.read_source_estimate(file_in+'_dev2_None_morph-vl.stc')
    # stc_std1=mne.read_source_estimate(file_in+'_std1_reverse_None_morph-vl.stc')
    # stc_std2=mne.read_source_estimate(file_in+'_std2_reverse_None_morph-vl.stc')
    stc_mmr1=mne.read_source_estimate(file_in+'_mmr1_mba_None_morph-vl.stc')
    stc_mmr2=mne.read_source_estimate(file_in+'_mmr2_pa_None_morph-vl.stc')
    # group_std.append(stc_std.data)
    # group_dev1.append(stc_dev1.data)
    # group_dev2.append(stc_dev2.data)
    group_mmr1.append(stc_mmr1.data)
    group_mmr2.append(stc_mmr2.data)
    # group_std1.append(stc_std1.data)
    # group_std2.append(stc_std2.data)
    
    # #extract ROIS for non-morphing data
    # src = mne.read_source_spaces(file_in +'_src')
    # fname_aseg = subjects_dir + s + '/mri/aparc+aseg.mgz'
    
    label_names = mne.get_volume_labels_from_aseg(fname_aseg)
    
    # std_roi=mne.extract_label_time_course(stc_std,fname_aseg,src,mode='mean',allow_empty=True)
    # dev1_roi=mne.extract_label_time_course(stc_dev1,fname_aseg,src,mode='mean',allow_empty=True)
    # dev2_roi=mne.extract_label_time_course(stc_dev2,fname_aseg,src,mode='mean',allow_empty=True)

    mmr1_roi=mne.extract_label_time_course(stc_mmr1,fname_aseg,src,mode='mean',allow_empty=True)
    mmr2_roi=mne.extract_label_time_course(stc_mmr2,fname_aseg,src,mode='mean',allow_empty=True)
    # std1_roi=mne.extract_label_time_course(stc_std1,fname_aseg,src,mode='mean',allow_empty=True)
    # std2_roi=mne.extract_label_time_course(stc_std2,fname_aseg,src,mode='mean',allow_empty=True)
    
    # group_std_roi.append(std_roi)
    # group_dev1_roi.append(dev1_roi)
    # group_dev2_roi.append(dev2_roi)
    group_mmr1_roi.append(mmr1_roi)
    group_mmr2_roi.append(mmr2_roi)
    # group_std1_roi.append(std1_roi)
    # group_std2_roi.append(std2_roi)

# group_std = np.asarray(group_std)
# group_dev1 = np.asarray(group_dev1)
# group_dev2 = np.asarray(group_dev2)
# group_std_roi = np.asarray(group_std_roi)
# group_dev1_roi = np.asarray(group_dev1_roi)
# group_dev2_roi = np.asarray(group_dev2_roi)
group_mmr1=np.asarray(group_mmr1)
group_mmr2=np.asarray(group_mmr2)
# group_std1=np.asarray(group_std1)
# group_std2=np.asarray(group_std2)
# group_std1_roi=np.asarray(group_std1_roi)
# group_std2_roi=np.asarray(group_std2_roi)
# group_std_roi = np.asarray(group_std_roi)
group_mmr1_roi=np.asarray(group_mmr1_roi)
group_mmr2_roi=np.asarray(group_mmr2_roi)

# np.save(root_path + 'cbsA_meeg_analysis/group_std_None_morph.npy',group_std)
# np.save(root_path + 'cbsA_meeg_analysis/group_dev1_None_morph.npy',group_dev1)
# np.save(root_path + 'cbsA_meeg_analysis/group_dev2_None_morph.npy',group_dev2)
# np.save(root_path + 'cbsA_meeg_analysis/group_std_None_morph_roi.npy',group_std_roi)
# np.save(root_path + 'cbsA_meeg_analysis/group_dev1_None_morph_roi.npy',group_dev1_roi)
# np.save(root_path + 'cbsA_meeg_analysis/group_dev2_None_morph_roi.npy',group_dev2_roi)
# np.save(root_path + 'cbsA_meeg_analysis/group_std_vector_morph.npy',group_std)
# np.save(root_path + 'cbsA_meeg_analysis/group_std_vector_morph_roi.npy',group_std_roi)
np.save(root_path + 'cbsA_meeg_analysis/MEG/group_mmr1_mba_None_morph.npy',group_mmr1)
np.save(root_path + 'cbsA_meeg_analysis/MEG/group_mmr2_pa_None_morph.npy',group_mmr2)
np.save(root_path + 'cbsA_meeg_analysis/MEG/group_mmr1_mba_None_morph_roi.npy',group_mmr1_roi)
np.save(root_path + 'cbsA_meeg_analysis/MEG/group_mmr2_pa_None_morph_roi.npy',group_mmr2_roi)
# np.save(root_path + 'cbsA_meeg_analysis/MEG/magnitude_method/group_std1_reverse_None_morph.npy',group_std1)
# np.save(root_path + 'cbsA_meeg_analysis/MEG/magnitude_method/group_std2_reverse_None_morph.npy',group_std2)
# np.save(root_path + 'cbsA_meeg_analysis/MEG/magnitude_method/group_std1_reverse_None_morph_roi.npy',group_std1_roi)
# np.save(root_path + 'cbsA_meeg_analysis/MEG/magnitude_method/group_std2_reverse_None_morph_roi.npy',group_std2_roi)