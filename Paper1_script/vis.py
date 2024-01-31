#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 12:51:22 2024
Visualization for Figure 2
@author: tzcheng
"""
import numpy as np
import time
import mne
import scipy.stats as stats
from scipy import stats,signal
from mne import spatial_src_adjacency
from mne.stats import spatio_temporal_cluster_1samp_test
import sklearn 
import matplotlib.pyplot as plt 
from scipy.io import wavfile

def plot_err(data,color,t):
    group_avg=np.mean(data,axis=0)
   #plt.figure()
    err=np.std(data,axis=0)/np.sqrt(data.shape[0])
    up=group_avg+err
    lw=group_avg-err
    plt.plot(t,group_avg,color=color)
    plt.fill_between(t,up,lw,color=color,alpha=0.5)

root_path='/media/tzcheng/storage/CBS/'

#%%####################################### visualize MMR
std = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_std_eeg.npy')
dev1 = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_dev1_eeg.npy')
dev2 = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_dev2_eeg.npy')

times = np.linspace(-0.1,0.6,3501)

MMR1 = dev1 - std
MMR2 = dev2 - std

plt.figure()
plot_err(MMR1,'r',times)
plot_err(MMR2,'b',times)

plt.title('MMRs')
plt.legend(['Native MMR','','Nonative MMR',''])
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.xlim([-0.05,0.45])

#%%####################################### visualize audio and cABR
## cABR 
std = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_std_cabr_eeg_200.npy')
dev1 = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_dev1_cabr_eeg_200.npy')
dev2 = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_dev2_cabr_eeg_200.npy')

## Audio files
fs, std_audio = wavfile.read('/media/tzcheng/storage/CBS/stimuli/+10.wav')
fs, dev1_audio = wavfile.read('/media/tzcheng/storage/CBS/stimuli/-40.wav')
fs, dev2_audio = wavfile.read('/media/tzcheng/storage/CBS/stimuli/+40.wav')
# Downsample
fs_new = 5000
num_std = int((len(std_audio)*fs_new)/fs)
num_dev = int((len(dev2_audio)*fs_new)/fs)  # #sample_new/fs_new=#sample/fs find number of samples in the resampled data       
std_audio = signal.resample(std_audio, num_std, t=None, axis=0, window=None)
dev1_audio = signal.resample(dev1_audio, num_dev, t=None, axis=0, window=None)
dev2_audio = signal.resample(dev2_audio, num_dev, t=None, axis=0, window=None)

## plot std, dev1, dev2
plt.figure()
plot_err(std,'k',np.linspace(-0.02,0.2,1101))
plt.xlim([0,0.2])
plt.figure()
plt.plot(np.linspace(0,0.1,500),std_audio)
plt.xlim([0,0.1])

plt.figure()
plot_err(dev1,'k',np.linspace(-0.02,0.2,1101))
plt.xlim([0,0.2])
plt.figure()
plt.plot(np.linspace(0,0.13,650),dev1_audio)
plt.xlim([0,0.13])

plt.figure()
plot_err(dev2,'k',np.linspace(-0.02,0.2,1101))
plt.xlim([0,0.2])
plt.figure()
plt.plot(np.linspace(0,0.13,650),dev2_audio)
plt.xlim([0,0.13])