#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 16:53:55 2024
ME2 Analysis
1. Frequency
2. Connectivity anlaysis
3. ML sliding estimator
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
from mne_connectivity import spectral_connectivity_epochs
from mne_connectivity.viz import plot_connectivity_circle
from mne.viz import circular_layout


def plot_err(group_data,color,t):
    group_avg=np.mean(group_data,axis=0)
    err=np.std(group_data,axis=0)/np.sqrt(group_data.shape[0])
    up=group_avg+err
    lw=group_avg-err
    #t=np.linspace(-100,600,3501)
    plt.plot(t,group_avg,color=color)
    plt.fill_between(t,up,lw,color=color,alpha=0.5)
    
#%%####################################### Load the files
subjects_dir = '/media/tzcheng/storage2/subjects/'
root_path='/media/tzcheng/storage/ME2_MEG/Zoe_analyses/'
fname_aseg = subjects_dir + 'fsaverage/mri/aparc+aseg.mgz'
label_names = np.asarray(mne.get_volume_labels_from_aseg(fname_aseg))
nROI = [61,63] # Auditory (STG 72,108), Motor (precentral 66 102 and paracentral 59, 95), Basal ganglia group (7,8,9,16,26,27,28,31) (left and then right)
nV = 10020 # need to manually look up from the whole-brain plot

fs, audio = wavfile.read(root_path + 'Stimuli/Random.wav') # Random, Duple300, Triple300
MEG_v = np.load(root_path + 'me2_meg_analysis/br_group_03_stc_mne.npy') # 01,02,03,04
MEG_roi = np.load(root_path + 'me2_meg_analysis/br_group_03_stc_mne_roi.npy') # 01,02,03,04

stc1 = mne.read_source_estimate('/media/tzcheng/storage/BabyRhythm/br_03/sss_fif/br_03_01_stc_lcmv_morph-vl.stc')
times = stc1.times
src = mne.read_source_spaces(subjects_dir + 'fsaverage/bem/fsaverage-vol-5-src.fif')

stc1.data = MEG_v.mean(axis=0)
stc1.plot(src = src,clim=dict(kind="value",lims=[5,5.5,8]))

#%%####################################### Frequency analysis: Are there peak amplitude in the beat and meter rates? which ROI, which source?
fmin = 0.5
fmax = 5
MEG_fs = 1000

## Frequency spectrum of the audio 
psds, freqs = mne.time_frequency.psd_array_welch(
audio,fs, # could replace with label time series
n_fft=len(audio),
n_overlap=0,
n_per_seg=None,
fmin=fmin,
fmax=fmax,)

plt.figure()
plt.plot(freqs,psds)

## Frequency spectrum of the MEG
psds, freqs = mne.time_frequency.psd_array_welch(
MEG_roi,MEG_fs, # could replace with label time series
n_fft=np.shape(MEG_roi)[2],
n_overlap=0,
n_per_seg=None,
fmin=fmin,
fmax=fmax,)

plt.figure()
plot_err(psds[:,nROI,:],'k',freqs)
plt.title(label_names[nROI])

psds, freqs = mne.time_frequency.psd_array_welch(
MEG_v,MEG_fs, # take about 2 mins
n_fft=np.shape(MEG_v)[2],
n_overlap=0,
n_per_seg=None,
fmin=fmin,
fmax=fmax,)

plt.figure()
plot_err(psds[:,nV,:],'k',freqs)

## Whole-brain frequency spectrum
stc1.data = psds.mean(axis=0)
stc1.tmin = freqs[0] # hack into the time with freqs
stc1.tstep = np.diff(freqs)[0] # hack into the time with freqs
stc1.plot(src = src,clim=dict(kind="value",lims=[1,3,5]))

#%%####################################### Connectivity analysis
con_methods = ["pli", "wpli2_debiased", "ciplv"]
con = spectral_connectivity_epochs(
    MEG_roi,
    method=con_methods,
    mode="multitaper",
    sfreq=MEG_fs,
    fmin=fmin,
    fmax=fmax,
    faverage=True,
    mt_adaptive=True,
    n_jobs=1,
)

## visualization

#%%##### 
# Downsample
fs_new = 1000
num = int((len(audio)*fs_new)/fs)        
audio = signal.resample(audio, num, t=None, axis=0, window=None)