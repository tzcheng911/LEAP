#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 11:39:32 2024

@author: tzcheng
"""
## Import library  
import matplotlib.pyplot as plt
import numpy as np
import os 

import mne
from mne.datasets import somato
from mne.time_frequency import tfr_morlet

#%%#######################################   
tmin = 1.0
tmax = 9.0
fmin = 0.5
fmax = 5

age = '7mo/'
runs = ['_01','_02','_03','_04']
root_path='/media/tzcheng/storage/ME2_MEG/Zoe_analyses/'
subjects_dir = '/media/tzcheng/storage2/subjects/'

os.chdir(root_path + age)

subjects = []

for file in os.listdir():
    if file.startswith('me2_'): 
        subjects.append(file)

#%%####################################### individual analysis
subj = subjects[2]

#%%####################################### visualize individual sensor PSD
for run in runs: 
    epochs = mne.read_epochs(root_path + age + subj + '/sss_fif/' + subj + run + '_otp_raw_sss_proj_fil50_epoch.fif')
    evoked = epochs['Trial_Onset'].average()
    sfreq = epochs.info["sfreq"]
    # epochs.compute_psd(fmin=0.5, fmax=15.0, tmin= 0, tmax = 10).plot(picks="data", exclude="bads")
    # evoked.compute_psd(fmin=0.5, fmax=5.0, tmin= 0, tmax = 10).plot(picks="data", exclude="bads")
    # evoked.compute_psd(fmin=fmin, fmax=fmax, tmin= tmin, tmax = tmax).plot(average = True,picks="data", exclude="bads")
    evoked.compute_psd("welch",
    n_fft=int(sfreq * (tmax - tmin)),
    n_overlap=0,
    n_per_seg=None,
    tmin=tmin,
    tmax=tmax,
    fmin=fmin,
    fmax=fmax,
    window="boxcar",
    verbose=False,).plot(average=True,picks="data", exclude="bads")

#%%####################################### visualize individual source PSD
for run in runs: 
    epochs = mne.read_epochs(root_path + age + subj + '/sss_fif/' + subj + run + '_otp_raw_sss_proj_fil50_epoch.fif')
    evoked = epochs['Trial_Onset'].average()
    
    trans=mne.read_trans(root_path + age + subj + '/sss_fif/' + subj + '_trans.fif')
    src=mne.read_source_spaces(subjects_dir + subj + '/bem/' + subj + '-vol-5-src.fif')
#    src=mne.read_source_spaces(subjects_dir + subj + '/bem/' + subj + '-oct-6-src.fif')
    bem=mne.read_bem_solution(subjects_dir +  subj + '/bem/' + subj + '-5120-5120-5120-bem-sol.fif')
    fwd=mne.make_forward_solution(epochs.info,trans,src,bem,meg=True,eeg=False)
    
    cov = mne.read_cov(root_path + age + subj + '/sss_fif/' + subj + run + '_erm_otp_raw_sss_proj_fil50-cov.fif')
    inverse_operator = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, cov,loose=1,depth=0.8)
    
    stc = mne.minimum_norm.apply_inverse(evoked, inverse_operator)
    sfreq = epochs.info["sfreq"]

#    label = mne.read_labels_from_annot(subject=subj, parc='aparc',subjects_dir=subjects_dir)
    fname_aseg = subjects_dir + subj + '/mri/aparc+aseg.mgz'
    label = mne.get_volume_labels_from_aseg(fname_aseg)
    label_tc=mne.extract_label_time_course(stc,fname_aseg,src,mode='mean',allow_empty=True)

    psds, freqs = mne.time_frequency.psd_array_welch(
    label_tc,sfreq, # could replace with label time series
    n_fft=int(sfreq * (tmax - tmin)),
    n_overlap=0,
    n_per_seg=None,
    fmin=fmin,
    fmax=fmax,)
        
    plt.figure()
    plt.plot(freqs,psds[55,:])
    plt.title(label[55])

#%%####################################### visualize group results
subjects_dir = '/media/tzcheng/storage2/subjects/'
root_path='/media/tzcheng/storage/ME2_MEG/Zoe_analyses/me2_meg_analysis/'

br_mne = np.load(root_path + 'br_group_01_stc_mne.npy')
br_lcmv = np.load(root_path + 'br_group_01_stc_lcmv.npy')

stc1 = mne.read_source_estimate('/media/tzcheng/storage/BabyRhythm/br_03/sss_fif/br_03_01_stc_lcmv_morph-vl.stc')
times = stc1.times
src = mne.read_source_spaces(subjects_dir + 'fsaverage/bem/fsaverage-vol-5-src.fif')

stc1.data = br_mne.mean(axis=0)
stc1.data = br_lcmv.mean(axis=0)
stc1.plot(src = src,clim=dict(kind="value",lims=[5,5.5,8]))

#%% temp for plotting
subjects_dir = '/media/tzcheng/storage2/subjects/'
root_path='/media/tzcheng/storage/ME2_MEG/Zoe_analyses/me2_meg_analysis/'

stc1 = mne.read_source_estimate('/media/tzcheng/storage/BabyRhythm/br_03/sss_fif/br_03_01_stc_lcmv_morph-vl.stc')
times = stc1.times
src = mne.read_source_spaces(subjects_dir + 'fsaverage/bem/fsaverage-vol-5-src.fif')

# seven_mo_mne1 = np.load('7mo_0_15_group_04_stc_mne.npy')
# seven_mo_mne2 = np.load('7mo_15_32_group_04_stc_mne.npy')
# stc1.data = 0.5*(seven_mo_mne1.mean(0) + seven_mo_mne2.mean(0))
eleven_mo_mne = np.load('11mo_group_01_stc_mne.npy')
stc1.data = eleven_mo_mne.mean(0)
stc1.plot(src = src,clim=dict(kind="value",lims=[5,5.5,8]))