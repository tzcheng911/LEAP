#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:03:17 2024
Conducted inverse solution and group averaged for dss cleaned data source localization.

@author: tzcheng
"""

#%%####################################### Import library  
import mne
from mne.decoding import UnsupervisedSpatialFilter
from mne.preprocessing import ICA
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne.preprocessing import Xdawn
import matplotlib.pyplot as plt 
import numpy as np
import os
from scipy import stats,signal
from numpy import dot
from numpy.linalg import norm
from scipy.stats import pearsonr
import scipy as sp
import os
import pandas as pd
import scipy.stats as stats
from scipy.io import wavfile, loadmat
import time

from sklearn.decomposition import PCA, FastICA
import random


def do_inverse_FFR(s,evokeds_inv,run,nspeech,morph,n_top,n_trial,n_filter):
    root_path='/media/tzcheng/storage2/CBS/'
    subjects_dir = '/media/tzcheng/storage2/subjects/'

    file_in = root_path + s + '/sss_fif/' + s
    fwd = mne.read_forward_solution(file_in + '-fwd.fif')
    cov = mne.read_cov(file_in + run + '_erm_otp_raw_sss_proj_' + n_filter + '_ffr-cov.fif')
    epoch = mne.read_epochs(file_in + run + '_otp_raw_sss_proj_' + n_filter + '_ffr_e_' + str(n_trial) + '.fif')
    inverse_operator = mne.minimum_norm.make_inverse_operator(epoch.info, fwd, cov,loose=1,depth=0.8)
    evokeds_inv_stc = mne.minimum_norm.apply_inverse((evokeds_inv), inverse_operator, pick_ori = None)

    if morph == True:
        print('Morph ' + s +  ' src space to common cortical space.')
        fname_src_fsaverage = subjects_dir + 'fsaverage/bem/fsaverage-vol-5-src.fif'
        src_fs = mne.read_source_spaces(fname_src_fsaverage)
        morph = mne.compute_source_morph(
            inverse_operator["src"],
            subject_from=s,
            subjects_dir=subjects_dir,
            niter_affine=[10, 10, 5],
            niter_sdr=[10, 10, 5],  # just for speed
            src_to=src_fs,
            verbose=True)
        evokeds_inv_stc_fsaverage = morph.apply(evokeds_inv_stc)
        evokeds_inv_stc_fsaverage.save(file_in + '_' + nspeech + str(n_top) + n_filter + str(n_trial) + '_morph', overwrite=True)

    else: 
        print('No morphing has been performed. The individual results may not be good to average.')
        evokeds_inv_stc.save(file_in + '_' + nspeech + '_pcffr80450_' + str(n_top), overwrite=True)

def group_stc(subj,baby_or_adult,n_top, n_trial, n_filter):
    group_std = []
    group_dev1 = []
    group_dev2 = []
    group_std_roi = []
    group_dev1_roi = []
    group_dev2_roi = []
    root_path='/media/tzcheng/storage2/CBS/'
    subjects_dir = '/media/tzcheng/storage2/subjects/'
    src = mne.read_source_spaces('/media/tzcheng/storage2/subjects/fsaverage/bem/fsaverage-vol-5-src.fif') # for morphing data
    fname_aseg = subjects_dir + 'fsaverage' + '/mri/aparc+aseg.mgz'

    for s in subj:
        print('Extracting ' + s + ' data')
        file_in = root_path + s + '/sss_fif/' + s
        
        stc_std=mne.read_source_estimate(file_in+'_substd' + str(n_top) + n_filter + str(n_trial) + '_morph-vl.stc')
        stc_dev1=mne.read_source_estimate(file_in+'_dev1' + str(n_top) + n_filter + str(n_trial) + '_morph-vl.stc')
        stc_dev2=mne.read_source_estimate(file_in+'_dev2' + str(n_top) + n_filter + str(n_trial) + '_morph-vl.stc')
        group_std.append(stc_std.data)
        group_dev1.append(stc_dev1.data)
        group_dev2.append(stc_dev2.data)
            
        std_roi=mne.extract_label_time_course(stc_std,fname_aseg,src,mode='mean',allow_empty=True)
        dev1_roi=mne.extract_label_time_course(stc_dev1,fname_aseg,src,mode='mean',allow_empty=True)
        dev2_roi=mne.extract_label_time_course(stc_dev2,fname_aseg,src,mode='mean',allow_empty=True)
        group_std_roi.append(std_roi)
        group_dev1_roi.append(dev1_roi)
        group_dev2_roi.append(dev2_roi)
       
    group_std = np.asarray(group_std)
    group_dev1 = np.asarray(group_dev1)
    group_dev2 = np.asarray(group_dev2)
    group_std_roi = np.asarray(group_std_roi)
    group_dev1_roi = np.asarray(group_dev1_roi)
    group_dev2_roi = np.asarray(group_dev2_roi)
    np.save(root_path + baby_or_adult + '/MEG/FFR/group_ba_' + str(n_top) + '_' + n_filter + '_' + str(n_trial) +'_morph.npy',group_std)
    np.save(root_path + baby_or_adult + '/MEG/FFR/group_mba_' + str(n_top) + '_' + n_filter + '_' + str(n_trial) + '_morph.npy',group_dev1)
    np.save(root_path + baby_or_adult + '/MEG/FFR/group_pa_' + str(n_top) + '_' + n_filter + '_' + str(n_trial) + '_morph.npy',group_dev2)
    np.save(root_path + baby_or_adult + '/MEG/FFR/group_ba_' + str(n_top) + '_' + n_filter + '_' + str(n_trial) + '_morph_roi.npy',group_std_roi)
    np.save(root_path + baby_or_adult + '/MEG/FFR/group_mba_' + str(n_top) + '_' + n_filter + '_' + str(n_trial) + '_morph_roi.npy',group_dev1_roi)
    np.save(root_path + baby_or_adult + '/MEG/FFR/group_pa_' + str(n_top) + '_' + n_filter + '_' + str(n_trial) + '_morph_roi.npy',group_dev2_roi)

#%%#######################################
root_path='/media/tzcheng/storage2/CBS/'
os.chdir(root_path)
## Parameters

n_top = 'replicate' # 3, 10 or 'dss'
n_trial = '200'
runs = ['_01','_02']
cond = ['substd','dev1','dev2']
sounds = ['ba','mba','pa']
n_filter = 'f' # 'f': 80-2000 Hz; 'f80450': 80-450 Hz
baby_or_adult = 'cbsA_meeg_analysis' # baby (cbsb_meg_analysis) or adult (cbsA_meeg_analysis)

run = runs[0]
morph = True


subjects = [] 
for file in os.listdir():
    if file.startswith('cbs_A'):
        subjects.append(file)

# group_sensor = np.empty([len(subjects),3,306,1101])

for ns,s in enumerate(subjects):
    print(s)
    for nspeech, speech in enumerate(cond):
        file_in = root_path + s + '/sss_fif/' + s
        evokeds = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_' + n_filter + '_evoked_' + speech + '_ffr_' + str(n_trial) +'.fif')[0]
        # dss_clean_meg = loadmat(root_path + 'mat/MEG_' + n_filter + '/dss_output/' + sounds[nspeech] + '/clean_' + sounds[nspeech] + '_' + s + '_MEG_epoch_f80450.mat') # need to change the path for different conditions
        # dss_clean_meg = dss_clean_meg['megclean2']
        # evokeds.data = dss_clean_meg.mean(0) # average across all trials
        do_inverse_FFR(s,evokeds,run,speech,morph,n_top,n_trial,n_filter)
#         group_sensor[ns,nspeech,:,:] = dss_clean_meg.mean(0)

# np.save(root_path + baby_or_adult + '/MEG/FFR/group_ba_' + str(n_top) + '_f80450_' + str(n_trial) + '_sensor.npy',group_sensor[:,0,:,:])
# np.save(root_path + baby_or_adult + '/MEG/FFR/group_mba_' + str(n_top) + '_f80450_' + str(n_trial) +'_sensor.npy',group_sensor[:,1,:,:])
# np.save(root_path + baby_or_adult + '/MEG/FFR/group_pa_' + str(n_top) + '_f80450_' + str(n_trial) +'_sensor.npy',group_sensor[:,2,:,:])

group_stc(subjects,baby_or_adult,n_top,n_trial,n_filter)