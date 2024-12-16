#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 09:15:32 2024
Generate random duple and triple group npy file 

@author: tzcheng
"""

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

age = '11mo' # '7mo', '11mo' or '' for adults br
runs = ['_02']
resample_or_not = True
rfs = 250
root_path='/media/tzcheng/storage/ME2_MEG/Zoe_analyses/' 
# root_path='/media/tzcheng/storage/BabyRhythm/' # for adults
subjects_dir = '/media/tzcheng/storage2/subjects/'
os.chdir(root_path + age)
subjects = []

for file in os.listdir():
    if file.startswith('me2_'): 
    # if file.startswith('br_'): 
        subjects.append(file)

for run in runs:
    print('Performing run ' + run[-1])
    # output the sensor time series in npy files
    group_randduple = []
    group_randtriple = []
    
    for s in subjects:
        print('Extracting ' + s + ' data')
        file_in = root_path + age + '/' + s + '/sss_fif/' + s + run
        evoked_randduple = mne.read_evokeds(file_in + '_otp_raw_sss_proj_fil50_mag6pT_evoked_randduple.fif')[0]
        evoked_randtriple = mne.read_evokeds(file_in + '_otp_raw_sss_proj_fil50_mag6pT_evoked_randtriple.fif')[0]

        group_randduple.append(evoked_randduple.data)
        group_randtriple.append(evoked_randtriple.data)

    group_randduple = np.asarray(group_randduple)
    group_randtriple = np.asarray(group_randtriple)
    np.save('/media/tzcheng/storage/ME2_MEG/Zoe_analyses/me2_meg_analysis/7mo_group' + run + '_mag6pT_randduple_sensor.npy',group_randduple)
    np.save('/media/tzcheng/storage/ME2_MEG/Zoe_analyses/me2_meg_analysis/7mo_group' + run + '_mag6pT_randtriple_sensor.npy',group_randtriple)
    
    #%% output the source time series in npy files
    group_stc_mne_rd = []
    group_stc_mne_rt = []
    group_stc_mne_rd_roi = []
    group_stc_mne_rt_roi = []
    
    src = mne.read_source_spaces('/media/tzcheng/storage2/subjects/fsaverage/bem/fsaverage-vol-5-src.fif') # for morphing data
    fname_aseg = subjects_dir + 'fsaverage' + '/mri/aparc+aseg.mgz'
    
    for s in subjects:
        print('Extracting ' + s + ' data')
        file_in = root_path + age + '/' + s + '/sss_fif/' + s + run    
        stc_mne_rd = mne.read_source_estimate(file_in+'_stc_mne_morph_mag6pT_randduple-vl.stc')
        stc_mne_rt = mne.read_source_estimate(file_in+'_stc_mne_morph_mag6pT_randtriple-vl.stc')

        if resample_or_not:
            stc_mne_rd.data = stc_mne_rd.data.astype('float64')
            stc_mne_rd = stc_mne_rd.resample(sfreq = rfs)
            stc_mne_rt.data = stc_mne_rt.data.astype('float64')
            stc_mne_rt = stc_mne_rt.resample(sfreq = rfs)
        else:
            print("No resampling has been performed")
        group_stc_mne_rd.append(stc_mne_rd.data)
        group_stc_mne_rt.append(stc_mne_rt.data)
            
        label_names = mne.get_volume_labels_from_aseg(fname_aseg)
        stc_mne_rd_roi = mne.extract_label_time_course(stc_mne_rd,fname_aseg,src,mode='mean',allow_empty=True)
        group_stc_mne_rd_roi.append(stc_mne_rd_roi)
        stc_mne_rt_roi = mne.extract_label_time_course(stc_mne_rt,fname_aseg,src,mode='mean',allow_empty=True)
        group_stc_mne_rt_roi.append(stc_mne_rt_roi)

    group_stc_mne_rd = np.asarray(group_stc_mne_rd)
    group_stc_mne_rt = np.asarray(group_stc_mne_rt)
    group_stc_mne_rd_roi = np.asarray(group_stc_mne_rd_roi)
    group_stc_mne_rt_roi = np.asarray(group_stc_mne_rt_roi)
    
    np.save('/media/tzcheng/storage/ME2_MEG/Zoe_analyses/me2_meg_analysis/' + age +'_group' + run + '_stc_rs_mne_mag6pT_randduple_roi.npy',group_stc_mne_rd_roi) 
    np.save('/media/tzcheng/storage/ME2_MEG/Zoe_analyses/me2_meg_analysis/' + age +'_group' + run + '_stc_rs_mne_mag6pT_randtriple_roi.npy',group_stc_mne_rt_roi)
    np.save('/media/tzcheng/storage/ME2_MEG/Zoe_analyses/me2_meg_analysis/' + age +'_group' + run + '_stc_rs_mne_mag6pT_randduple_morph.npy',group_stc_mne_rd)
    np.save('/media/tzcheng/storage/ME2_MEG/Zoe_analyses/me2_meg_analysis/' + age +'_group' + run + '_stc_rs_mne_mag6pT_randtriple_morph.npy',group_stc_mne_rt)