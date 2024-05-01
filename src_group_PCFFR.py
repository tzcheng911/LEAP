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
        pca_data.transpose(),sfreq, # could replace with label time series
        n_fft=int(sfreq * (tmax - tmin)),
        n_overlap=0,
        n_per_seg=None,
        fmin=fmin,
        fmax=fmax,)
    
    ## peak detection
    ind_components = np.argsort(psds[:,5:7].mean(axis=1))[-3:] # do top three PCs for now
    
    plt.figure()
    plt.plot(freqs,psds.transpose())
    plt.plot(freqs,psds[ind_components,:].transpose(),color='red',linestyle='dashed')
    
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
speech = ['substd','dev1','dev2']
run = runs[0]
morph = True

subjects = [] 
for file in os.listdir():
    if file.startswith('cbs_A101'):
        subjects.append(file)

for s in subjects:
    print(s)
    for nspeech in speech:
        file_in = root_path + s + '/sss_fif/' + s
        evokeds = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_f_evoked_' + nspeech + '_cabr.fif')[0]
        data = evokeds.get_data()
        pca_data,ind_components,data_topPC = select_PC(data,sfreq,tmin,tmax,fmin,fmax)
        evokeds.data = data_topPC
        do_inverse_FFR(s,evokeds,run,nspeech,morph)
        
        