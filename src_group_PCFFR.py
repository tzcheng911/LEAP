#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 16:01:53 2024
Conducted dimension reduction for each subject using spectrum analysis. 
Keep the components that include peaks between 90-100 Hz.
Could be used in cbs_A and cbs_b
cbs_b outliers: cbs_b116, cbs_b118
@author: tzcheng
"""

#%%####################################### Import library  
import mne
from mne.decoding import UnsupervisedSpatialFilter
from mne.preprocessing import ICA
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne.beamformer import apply_lcmv, make_lcmv

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
    # ind_components = np.argsort(psds[:,fl:fh].mean(axis=1))[::-1][:n_top] # do top n PC
    ind_components = np.argsort(np.max(psds[:,fl:fh],axis=1))[::-1][:n_top] # modify from getting the max mean value to get the max max value between the [lb, hb]
    explained_variance_ratio = pca.explained_variance_ratio_[ind_components]
    
    ## plot top 3 PC's spectrum
    plt.figure()
    plt.plot(freqs,psds.transpose())
    plt.plot(freqs,psds[ind_components,:].transpose(),color='red',linestyle='dashed')
    print('variance explained: ' + str(explained_variance_ratio))
    
    ## display top n PCs weight in spatial location 
    w = pca.components_ # (n_components, n_features)
    picks = mne.pick_types(evokeds.info, meg='mag') # meg='grad' or meg='mag'
    info_sel = mne.pick_info(evokeds.info, picks)
    weights = w[:, picks]
    # for i in range(len(ind_components)):
    #     mne.viz.plot_topomap(weights[i], info_sel)
    #     plt.title(f'PC {i+1}')
    #     plt.show()

    ## keep only top 3 PC's data: PC projection to all the channels
    Xhat = np.dot(pca_data[:,ind_components], w[ind_components,:])
    Xhat += np.mean(X, axis=0) 
    Xhat = Xhat.transpose() 
    return pca_data.transpose(),ind_components,w,explained_variance_ratio,Xhat

def do_inverse_FFR(model, s,evokeds_inv,run,condition,morph,n_top,n_trial,hp,lp):
    root_path='/media/tzcheng/storage2/CBS/'
    subjects_dir = '/media/tzcheng/storage2/subjects/'

    file_in = root_path + s + '/sss_fif/' + s
    fwd = mne.read_forward_solution(file_in + '-fwd.fif')
    cov = mne.read_cov(file_in + run + '_erm_otp_raw_sss_proj_f' + str(hp) + str(lp) + '_ffr-cov.fif')
    ## separate filter for each condition
    # noise_cov = mne.read_cov(file_in + '_' + condition + run + '_otp_raw_sss_proj_f' + str(hp) + str(lp) + '_ffr_e-noise-cov.fif')
    # data_cov = mne.read_cov(file_in + '_' + condition + run + '_otp_raw_sss_proj_f' + str(hp) + str(lp) + '_ffr_e-data-cov.fif')
    ## common filter for both conditions
    noise_cov = mne.read_cov(file_in + run + '_otp_raw_sss_proj_f' + str(hp) + str(lp) + '_ffr_e-noise-cov.fif')
    data_cov = mne.read_cov(file_in + run + '_otp_raw_sss_proj_f' + str(hp) + str(lp) + '_ffr_e-data-cov.fif')
    inverse_operator = mne.minimum_norm.make_inverse_operator(evokeds_inv.info, fwd, cov,loose=1,depth=0.8)
    
    if model == 'mne':    
        evokeds_inv_stc = mne.minimum_norm.apply_inverse((evokeds_inv), inverse_operator, pick_ori = None)
    elif model == 'beamformer':
        filters = make_lcmv(
            evokeds_inv.info,
            fwd,
            data_cov,
            reg=0,
            noise_cov=noise_cov,
            pick_ori="max-power",
            weight_norm="unit-noise-gain",
            rank='info',
            reduce_rank = True,
            depth = 0.8
        )
        evokeds_inv_stc = apply_lcmv(evokeds_inv, filters)
        # evokeds_inv_stc.plot(src = src)
    else: 
        print("specify the inverse model to use")
    
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
        evokeds_inv_stc_fsaverage.save(file_in + '_pcffr' + str(hp) + str(lp) + '_ntrial' + str(n_trial)  + '_' + str(n_top) + '_' +  condition + run + '_morph', overwrite=True)

    else: 
        print('No morphing has been performed. The individual results may not be good to average.')
        evokeds_inv_stc.save(file_in + '_pcffr' + str(hp) + str(lp) + '_ntrial' + str(n_trial)  + '_' + str(n_top) + condition + run + '_test_morph', overwrite=True)

def group_stc(subj,condition,n_trial,n_top,hp,lp):
    root_path='/media/tzcheng/storage2/CBS/'
    subjects_dir = '/media/tzcheng/storage2/subjects/'
    src = mne.read_source_spaces('/media/tzcheng/storage2/subjects/fsaverage/bem/fsaverage-vol-5-src.fif') # for morphing data
    fname_aseg = subjects_dir + 'fsaverage' + '/mri/aparc+aseg.mgz'

    print('Extracting ' + s + ' data')
    file_in = root_path + s + '/sss_fif/' + s
    stc=mne.read_source_estimate(file_in + '_pcffr' + str(hp) + str(lp) + '_ntrial' + str(n_trial) + '_' + str(n_top) + '_' + condition + run + '_morph-vl.stc') 
    stc_roi=mne.extract_label_time_course(stc,fname_aseg,src,mode='mean',allow_empty=True)
    stc_data = stc.data
    stc_roi_data = stc_roi.data
    return stc_data,stc_roi_data

#%%#######################################'
root_path='/media/tzcheng/storage2/CBS/'
os.chdir(root_path)
## Parameters
do_PCA = True ## if True, assign n_top cuz it cannot be 0
morph = True

## preproc parameters
n_top = 3 # change to 10
n_trial = 200
lp = 2000 # try 200 (suggested by Nike) or 450 (from Coffey paper) or 2000 zc and coffey paper
hp = 80
runs = ['_01','_02']
cond0 = ['substd','dev1','dev2']
cond1 = ['p10','n40','p40']
baby_or_adult = 'cbsb_meg_analysis' # baby or adult
inverse_model = 'beamformer'
run = runs[0]

## PCA parameters
fmin = 50
fmax = 150
sfreq = 5000
# note that the freq resolution is not high so chec kthe freqs in psds to see what freq cutoff it is selected 
lb = 80 # change from 90 to 80 to include the peaks for spanish speakers 05/01/2026
hb = 110

subjects = [] 
for file in os.listdir():
    if file.startswith('cbs_b'):
        subjects.append(file)
print(subjects)
group_morph = np.empty([len(subjects),len(cond0),14629,1101])
group_roi = np.empty([len(subjects),len(cond0),114,1101])
group_sensor = np.empty([len(subjects),len(cond0),306,1101])
group_pca = np.empty([len(subjects),len(cond0),306,1101])
group_pca_weight = np.empty([len(subjects),len(cond0),306,306])
group_pc_info = np.empty([len(subjects),len(cond0),n_top,2]) # Last dim: first is the ind, 2nd is the explained var ratio of the corresponding PC

for ns,s in enumerate(subjects):
    print(s)
    for nspeech, speech in enumerate(cond0):
        file_in = root_path + s + '/sss_fif/' + s
        evokeds = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_f' + str(hp) + str(lp) + '_evoked_' + speech + '_ffr_' + str(n_trial) +'.fif')[0]
        data = evokeds.get_data()
        if do_PCA:
            print('Run src on PCA-reduced signals')
            pca_data,ind_components,pca_weight,explained_variance_ratio,data_topPC = select_PC(data,sfreq,fmin,fmax,lb,hb,n_top)
            evokeds.data = data_topPC
            group_sensor[ns,nspeech,:len(data_topPC),:] = data_topPC
            group_pca[ns,nspeech,:,:] = pca_data
            group_pca_weight[ns,nspeech,:len(data_topPC),:len(data_topPC)] = pca_weight
            group_pc_info[ns,nspeech,:,0] = ind_components
            group_pc_info[ns,nspeech,:,1] = explained_variance_ratio
        else:
            print('Run src on non-PCA signals')
            group_sensor[ns,nspeech,:len(data),:] = data
        do_inverse_FFR(inverse_model,s,evokeds,run,cond1[nspeech],morph,n_top,n_trial,hp,lp)
        group_morph[ns,nspeech,:,:],group_roi[ns,nspeech,:,:] = group_stc(s,cond1[nspeech],n_trial,n_top,hp,lp)

for ncondition,condition in enumerate(cond1):
    head = root_path + baby_or_adult + '/MEG/FFR/ntrial_200/group_pcffr' + str(hp) + str(lp) + '_ntrial' + str(n_trial) + '_' + str(n_top) + '_' + condition
    np.save(head + '_sensor_common_beamformer.npy',group_sensor[:,ncondition,:,:])        
    np.save(head + '_morph_common_beamformer.npy',group_morph[:,ncondition,:,:])
    np.save(head + '_roi_common_beamformer.npy',group_roi[:,ncondition,:,:])
    if do_PCA:
        np.save(head + '_top_' + str(n_top) + '_pc_data_common_beamformer.npy',group_pca[:,ncondition,:,:])
        np.save(head + '_top_' + str(n_top) + '_pc_weight_common_beamformer.npy',group_pc_info[:,ncondition,:,:])
        np.save(head + '_top_' + str(n_top) + '_pc_info_common_beamformer.npy',group_pc_info[:,ncondition,:,:])