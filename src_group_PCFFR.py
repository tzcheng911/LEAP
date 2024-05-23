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
import pandas as pd
import scipy.stats as stats
from scipy.io import wavfile
import time

from sklearn.decomposition import PCA, FastICA
import random

def select_PC(data,sfreq,fmin,fmax,lb,hb,n_top):
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
    fl = np.where(freqs>lb)[0][0]
    fh = np.where(freqs<hb)[0][-1]
    ind_components = np.argsort(psds[:,fl:fh].mean(axis=1))[::-1][:n_top] # do top three PCs for now
    explained_variance_ratio = pca.explained_variance_ratio_[ind_components]
    
    # plt.figure()
    # plt.plot(freqs,psds.transpose())
    # plt.plot(freqs,psds[ind_components,:].transpose(),color='red',linestyle='dashed')
    
    Xhat = np.dot(pca.transform(X)[:,ind_components], pca.components_[ind_components,:])
    Xhat += np.mean(X, axis=0) 
    Xhat = Xhat.transpose()    
    return pca_data.transpose(),ind_components,explained_variance_ratio,Xhat

def do_inverse_FFR(s,evokeds_inv,run,nspeech,morph,n_top,n_trial):
    root_path='/media/tzcheng/storage2/CBS/'
    subjects_dir = '/media/tzcheng/storage2/subjects/'

    file_in = root_path + s + '/sss_fif/' + s
    fwd = mne.read_forward_solution(file_in + '-fwd.fif')
    cov = mne.read_cov(file_in + run + '_erm_otp_raw_sss_proj_f80450_ffr-cov.fif')
    epoch = mne.read_epochs(file_in + run + '_otp_raw_sss_proj_f80450_ffr_e_' + str(n_trial) + '.fif')
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
        evokeds_inv_stc_fsaverage.save(file_in + '_' + nspeech +'_pcffr80450_' + str(n_top) + str(n_trial) + '_morph', overwrite=True)

    else: 
        print('No morphing has been performed. The individual results may not be good to average.')
        evokeds_inv_stc.save(file_in + '_' + nspeech + '_pcffr80450_' + str(n_top), overwrite=True)

def group_stc(subj,baby_or_adult,n_top, n_trial):
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
        
        stc_std=mne.read_source_estimate(file_in+'_substd_pcffr80450_' + str(n_top) + str(n_trial) + '_morph-vl.stc')
        stc_dev1=mne.read_source_estimate(file_in+'_dev1_pcffr80450_' + str(n_top) + str(n_trial) + '_morph-vl.stc')
        stc_dev2=mne.read_source_estimate(file_in+'_dev2_pcffr80450_' + str(n_top) + str(n_trial) + '_morph-vl.stc')
        group_std.append(stc_std.data)
        group_dev1.append(stc_dev1.data)
        group_dev2.append(stc_dev2.data)
            
        std_roi=mne.extract_label_time_course(stc_std,fname_aseg,src,mode='mean',allow_empty=True)
        dev1_roi=mne.extract_label_time_course(stc_dev1,fname_aseg,src,mode='mean',allow_empty=True)
        dev2_roi=mne.extract_label_time_course(stc_dev2,fname_aseg,src,mode='mean',allow_empty=True)
        group_std_roi.append(std_roi)
        group_dev1_roi.append(dev1_roi)
        group_dev2_roi.append(dev2_roi)
  
    n_trial = 'all'        
    group_std = np.asarray(group_std)
    group_dev1 = np.asarray(group_dev1)
    group_dev2 = np.asarray(group_dev2)
    group_std_roi = np.asarray(group_std_roi)
    group_dev1_roi = np.asarray(group_dev1_roi)
    group_dev2_roi = np.asarray(group_dev2_roi)
    np.save(root_path + baby_or_adult + '/MEG/FFR/group_ba_pcffr80450_' + str(n_top) + '_' + str(n_trial) +'_morph.npy',group_std)
    np.save(root_path + baby_or_adult + '/MEG/FFR/group_mba_pcffr80450_' + str(n_top) + '_' + str(n_trial) + '_morph.npy',group_dev1)
    np.save(root_path + baby_or_adult + '/MEG/FFR/group_pa_pcffr80450_' + str(n_top) + '_' + str(n_trial) + '_morph.npy',group_dev2)
    np.save(root_path + baby_or_adult + '/MEG/FFR/group_ba_pcffr80450_' + str(n_top) + '_' + str(n_trial) + '_morph_roi.npy',group_std_roi)
    np.save(root_path + baby_or_adult + '/MEG/FFR/group_mba_pcffr80450_' + str(n_top) + '_' + str(n_trial) + '_morph_roi.npy',group_dev1_roi)
    np.save(root_path + baby_or_adult + '/MEG/FFR/group_pa_pcffr80450_' + str(n_top) + '_' + str(n_trial) + '_morph_roi.npy',group_dev2_roi)

#%%#######################################' + str(n_top) + '
root_path='/media/tzcheng/storage2/CBS/'
os.chdir(root_path)
## Parameters
fmin = 50
fmax = 150
sfreq = 5000
lb = 90
hb = 100
n_top = 10 # change to 10
n_trial = 'all'
runs = ['_01','_02']
cond = ['substd','dev1','dev2']
baby_or_adult = 'cbsb_meg_analysis' # baby or adult

run = runs[0]
morph = True

subjects = [] 
for file in os.listdir():
    if file.startswith('cbs_b'):
        subjects.append(file)

group_sensor = np.empty([len(subjects),3,306,1101])
group_pca = np.empty([len(subjects),3,306,1101])
group_pc_info = np.empty([len(subjects),3,n_top,2]) # Last dim: first is the ind, 2nd is the explained var ratio of the corresponding PC

for ns,s in enumerate(subjects):
    print(s)
    for nspeech, speech in enumerate(cond):
        file_in = root_path + s + '/sss_fif/' + s
        evokeds = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_f80450_evoked_' + speech + '_ffr_' + str(n_trial) +'.fif')[0]
        data = evokeds.get_data()
        pca_data,ind_components,explained_variance_ratio,data_topPC = select_PC(data,sfreq,fmin,fmax,lb,hb,n_top)
        evokeds.data = data_topPC
        group_sensor[ns,nspeech,:,:] = data_topPC
        group_pca[ns,nspeech,:,:] = pca_data
        group_pc_info[ns,nspeech,:,0] = ind_components
        group_pc_info[ns,nspeech,:,1] = explained_variance_ratio
        do_inverse_FFR(s,evokeds,run,speech,morph,n_top,n_trial)
group_stc(subjects,baby_or_adult,n_top,n_trial)
np.save(root_path + baby_or_adult + '/MEG/FFR/group_ba_pcffr80450_' + str(n_top) + '_' + str(n_trial) + '_sensor.npy',group_sensor[:,0,:,:])
np.save(root_path + baby_or_adult + '/MEG/FFR/group_mba_pcffr80450_' + str(n_top) + '_' + str(n_trial) +'_sensor.npy',group_sensor[:,1,:,:])
np.save(root_path + baby_or_adult + '/MEG/FFR/group_pa_pcffr80450_' + str(n_top) + '_' + str(n_trial) +'_sensor.npy',group_sensor[:,2,:,:])

np.save(root_path + baby_or_adult + '/MEG/FFR/group_f80450_top_' + str(n_top) + '_' + str(n_trial) +'pc_data.npy',group_pca)
np.save(root_path + baby_or_adult + '/MEG/FFR/group_f80450_top_' + str(n_top) + '_' + str(n_trial) +'pc_info.npy',group_pc_info)