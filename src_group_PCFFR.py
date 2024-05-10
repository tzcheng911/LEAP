#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 16:01:53 2024
Conducted dimension reduction for each subject using spectrum analysis. 
Keep the components that include peaks between 90-100 Hz.
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
import seaborn as sns
import pandas as pd
import scipy.stats as stats
from scipy.io import wavfile
import time

from sklearn.decomposition import PCA, FastICA
import random

def select_PC(data,sfreq,tmin,tmax,fmin,fmax):
    X = data.transpose()
    
    pca = PCA()
    pca.fit(X) 
    pca_data = pca.fit_transform(X) 
    
    psds, freqs = mne.time_frequency.psd_array_welch(
        pca_data.transpose()[:,100:],sfreq, # could replace with label time series
        n_fft=len(pca_data.transpose()[0,100:]), # the higher the better freq resolution
        n_overlap=0,
        n_per_seg=None,
        fmin=fmin,
        fmax=fmax,)
    
    ## peak detection
    ind_components = np.argsort(psds[:,5:7].mean(axis=1))[-3:] # do top three PCs for now
    
    # plt.figure()
    # plt.plot(freqs,psds.transpose())
    # plt.plot(freqs,psds[ind_components,:].transpose(),color='red',linestyle='dashed')
    
    Xhat = np.dot(pca.transform(X)[:,ind_components], pca.components_[ind_components,:])
    Xhat += np.mean(X, axis=0) 
    Xhat = Xhat.transpose()    
    return pca_data,ind_components,Xhat

def do_inverse_FFR(s,evokeds_inv,run,nspeech,morph):
    root_path='/media/tzcheng/storage2/CBS/'
    subjects_dir = '/media/tzcheng/storage2/subjects/'

    file_in = root_path + s + '/sss_fif/' + s
    fwd = mne.read_forward_solution(file_in + '-fwd.fif')
    cov = mne.read_cov(file_in + run + '_erm_otp_raw_sss_proj_f_ffr-cov.fif')
    epoch = mne.read_epochs(file_in + run + '_otp_raw_sss_proj_f_cABR_e.fif')
    inverse_operator = mne.minimum_norm.make_inverse_operator(epoch.info, fwd, cov,loose=1,depth=0.8)

    ori = None # FFR should use None
    evokeds_inv_stc = mne.minimum_norm.apply_inverse((evokeds_inv), inverse_operator, pick_ori = ori)

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
        evokeds_inv_stc_fsaverage.save(file_in + '_' + nspeech +'_pcffr_morph', overwrite=True)

    else: 
        print('No morphing has been performed. The individual results may not be good to average.')
        evokeds_inv_stc.save(file_in + '_' + nspeech + '_pcffr' + str(ori), overwrite=True)

def group_stc(subj,baby_or_adult):
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
        
        stc_std=mne.read_source_estimate(file_in+'_substd_pcffr_morph-vl.stc')
        stc_dev1=mne.read_source_estimate(file_in+'_dev1_pcffr_morph-vl.stc')
        stc_dev2=mne.read_source_estimate(file_in+'_dev2_pcffr_morph-vl.stc')
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
    np.save(root_path + baby_or_adult + '/MEG/FFR/group_ba_pcffr_morph.npy',group_std)
    np.save(root_path + baby_or_adult + '/MEG/FFR/group_mba_pcffr_morph.npy',group_dev1)
    np.save(root_path + baby_or_adult + '/MEG/FFR/group_pa_pcffr_morph.npy',group_dev2)
    np.save(root_path + baby_or_adult + '/MEG/FFR/group_ba_pcffr_morph_roi.npy',group_std_roi)
    np.save(root_path + baby_or_adult + '/MEG/FFR/group_mba_pcffr_morph_roi.npy',group_dev1_roi)
    np.save(root_path + baby_or_adult + '/MEG/FFR/group_pa_pcffr_morph_roi.npy',group_dev2_roi)

#%%#######################################
root_path='/media/tzcheng/storage2/CBS/'
os.chdir(root_path)
## Parameters
tmin = 0
tmax = 0.13 
fmin = 50
fmax = 150
sfreq = 5000

runs = ['_01','_02']
cond = ['substd','dev1','dev2']
baby_or_adult = 'cbsA_meeg_analysis' # baby or adult

run = runs[0]
morph = True
source = False

subjects = [] 
for file in os.listdir():
    if file.startswith('cbs_A'):
        subjects.append(file)

group_sensor = np.empty([len(subjects),3,306,1101])

for ns,s in enumerate(subjects):
    print(s)
    for nspeech, speech in enumerate(cond):
        file_in = root_path + s + '/sss_fif/' + s
        evokeds = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_f_evoked_' + speech + '_cabr.fif')[0]
        data = evokeds.get_data()
        pca_data,ind_components,data_topPC = select_PC(data,sfreq,tmin,tmax,fmin,fmax)
        evokeds.data = data_topPC
        group_sensor[ns,nspeech,:,:] = data_topPC
        do_inverse_FFR(s,evokeds,run,speech,morph)
group_stc(subjects,baby_or_adult)
np.save(root_path + baby_or_adult + '/MEG/FFR/group_sensor_pcffr.npy',group_sensor)
np.save(root_path + baby_or_adult + '/MEG/FFR/group_ba_pcffr_sensor.npy',group_sensor[:,0,:,:])
np.save(root_path + baby_or_adult + '/MEG/FFR/group_mba_pcffr_sensor.npy',group_sensor[:,1,:,:])
np.save(root_path + baby_or_adult + '/MEG/FFR/group_pa_pcffr_sensor.npy',group_sensor[:,2,:,:])


        
        