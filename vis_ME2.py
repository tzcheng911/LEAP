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
from scipy.io import wavfile
from scipy import stats,signal

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

#%%####################################### visualize group results time series
subjects_dir = '/media/tzcheng/storage2/subjects/'
root_path='/media/tzcheng/storage/ME2_MEG/Zoe_analyses/me2_meg_analysis/'

br_mne = np.load(root_path + 'br_group_01_stc_mne.npy')
# br_lcmv = np.load(root_path + 'br_group_01_stc_lcmv.npy')

stc1 = mne.read_source_estimate('/media/tzcheng/storage/BabyRhythm/br_03/sss_fif/br_03_01_stc_lcmv_morph-vl.stc')
times = stc1.times
src = mne.read_source_spaces(subjects_dir + 'fsaverage/bem/fsaverage-vol-5-src.fif')

stc1.data = br_mne.mean(axis=0)
# stc1.data = br_lcmv.mean(axis=0)
stc1.plot(src = src,clim=dict(kind="value",lims=[5,5.5,8]))

#%%####################################### visualize audio in frequency
fs, random_audio = wavfile.read('/media/tzcheng/storage/ME2_MEG/Zoe_analyses/Stimuli/Random.wav')
fs, duple_audio = wavfile.read('/media/tzcheng/storage/ME2_MEG/Zoe_analyses/Stimuli/Duple300.wav')
fs, triple_audio = wavfile.read('/media/tzcheng/storage/ME2_MEG/Zoe_analyses/Stimuli/Triple300.wav')
plt.figure()
plt.plot(np.linspace(0,len(random_audio)/fs,len(random_audio)),random_audio)
plt.figure()
plt.plot(np.linspace(0,len(duple_audio)/fs,len(duple_audio)),duple_audio)
plt.figure()
plt.plot(np.linspace(0,len(triple_audio)/fs,len(triple_audio)),triple_audio)

psds, freqs = mne.time_frequency.psd_array_welch(
random_audio,fs, # could replace with label time series
n_fft=len(random_audio),
n_overlap=0,
n_per_seg=None,
fmin=fmin,
fmax=fmax,)

plt.figure()
plt.plot(freqs,psds)

# Downsample
fs_new = 1000
num_random = int((len(random_audio)*fs_new)/fs)
num_duple = int((len(duple_audio)*fs_new)/fs) 
num_triple = int((len(triple_audio)*fs_new)/fs)  
            
random_audio = signal.resample(random_audio, num_random, t=None, axis=0, window=None)
duple_audio = signal.resample(duple_audio, num_duple, t=None, axis=0, window=None)
triple_audio = signal.resample(triple_audio, num_triple, t=None, axis=0, window=None)

plt.figure()
plt.plot(np.linspace(0,len(random_audio)/fs_new,len(random_audio)),random_audio)
plt.figure()
plt.plot(np.linspace(0,len(duple_audio)/fs_new,len(duple_audio)),duple_audio)
plt.figure()
plt.plot(np.linspace(0,len(triple_audio)/fs_new,len(triple_audio)),triple_audio)