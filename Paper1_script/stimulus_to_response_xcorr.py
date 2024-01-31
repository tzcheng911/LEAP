#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 13:23:58 2023
Visualize and calculate EEG & MEG correlation in adults for MMR and Standard.

Correlation between EEG & MEG time series for...
1. averaged across all subjects and all vertices
2. averaged across all vertices for each subject (mean, max of window; point-by-point correlation)
3. averaged for each ROI
4. averaged for each vertice

Correlation methods: pearson r, xcorr, cosine similarity

@author: tzcheng
"""

import mne
import numpy as np
from numpy import dot
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy import stats,signal
import scipy as sp
import os
import seaborn as sns
import pandas as pd
import scipy.stats as stats
from scipy.io import wavfile

#%%######################################## xcorr between cABR and audio
root_path='/media/tzcheng/storage/CBS/'

## Load FFR from 0
std = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_std_cabr_eeg_200.npy')[:,100:]
dev1 = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_dev1_cabr_eeg_200.npy')[:,100:]
dev2 = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_dev2_cabr_eeg_200.npy')[:,100:]

# plt.figure()
# plt.plot(np.linspace(0,0.13,650),dev2_audio_r)

## Load real audio
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

audio0 = dev2_audio
EEG0 = dev2
times0_audio = np.linspace(0,len(audio0)/5000,len(audio0))
times0_eeg = np.linspace(0,len(EEG0[0])/5000,len(EEG0[0]))

## dev 1: noise burst from 40 ms (200th points)
ts = 200 + 100
te = 650
audio = audio0[ts:te] # try 0.042 to 0.13 s for dev2
EEG = EEG0[:,ts:te]
times_audio = times0_audio[ts:te]
times_eeg = times0_eeg[ts:te]
lags = signal.correlation_lags(len(audio),len(EEG[0]))
lags_s = lags/5000

## dev 2: noise burst from 0 ms (100th points)
ts = 100
te = 650
audio = audio0[ts:te] # try 0.02 to 0.13 s for dev2
EEG = EEG0[:,ts:te]
times_audio = times0_audio[ts:te]
times_eeg = times0_eeg[ts:te]
lags = signal.correlation_lags(len(audio),len(EEG[0]))
lags_s = lags/5000

## std: noise burst from 0 ms (100th points)
ts = 100
te = 500
audio = audio0[ts:te] # try 0.02 to 0.1 s for std
EEG = EEG0[:,ts:te]
times_audio = times0_audio[ts:te]
times_eeg = times0_eeg[ts:te]
lags = signal.correlation_lags(len(audio),len(EEG[0]))
lags_s = lags/5000

## Initialize 
xcorr_all_s = []
xcorr_lag_all_s = []


plt.figure()
plt.plot(times_audio,audio)

plt.figure()
plt.plot(times_eeg,EEG.mean(axis=0))

a = (audio - np.mean(audio))/np.std(audio)
a = a / np.linalg.norm(a)

## For averaged results: raw audio
b = (EEG.mean(axis=0) - np.mean(EEG.mean(axis=0)))/np.std(EEG.mean(axis=0))
b = b / np.linalg.norm(b)
xcorr = signal.correlate(a,b,mode='full')
xcorr = abs(xcorr)
xcorr_max = max(xcorr)
xcorr_maxlag = np.argmax(xcorr)
print("max lag: ", str(lags_s[xcorr_maxlag]*1000))
print("max xcorr: ", str(xcorr_max))
plt.figure()
plt.plot(lags_s,xcorr)

## For averaged results: rectified audio
# r_audio = abs(audio)
# a = (r_audio - np.mean(r_audio))/np.std(r_audio)
# a = a / np.linalg.norm(a)
# b = (EEG.mean(axis=0) - np.mean(EEG.mean(axis=0)))/np.std(EEG.mean(axis=0))
# b = b / np.linalg.norm(b)
# xcorr = signal.correlate(a,b,mode='full')
# xcorr = abs(xcorr)
# xcorr_max = max(xcorr)
# xcorr_maxlag = np.argmax(xcorr)
# lags_s[xcorr_maxlag]*1000

## For each individual
for s in np.arange(0,len(std),1):
    b = (EEG[s,:] - np.mean(EEG[s,:]))/np.std(EEG[s,:])
    
    ## the way matlab do xcorr normalization: the max is 1 if do a zero lag autocorrealtoin
    b = b / np.linalg.norm(b)

    xcorr = signal.correlate(a,b,mode='full')
    xcorr = abs(xcorr)
    xcorr_all_s.append(np.max(xcorr))
    xcorr_lag_all_s.append(np.argmax(xcorr))

print('abs xcorr between FFR & audio: ' + str(np.array(xcorr_all_s).mean()) + '(' + str(np.array(xcorr_all_s).std()) +')')
print('abs xcorr lag between FFR & audio (ms): ' + str(np.array(lags_s[xcorr_lag_all_s]*1000).mean())+ '(' + str(np.array(lags_s[xcorr_lag_all_s]*1000).std()) +')')

fig,axs = plt.subplots(len(EEG),1,sharex=True,sharey=True)
fig.suptitle('FFR')

for s in np.arange(0,len(EEG),1):
    axs[s].plot(times_eeg,EEG[s,:])
    
pa_coef = np.load('xcorr_coef_pa.npy')
ba_coef = np.load('xcorr_coef_ba.npy')
mba_coef = np.load('xcorr_coef_mba.npy')
mba_lag = np.load('xcorr_lag_mba.npy')
ba_lag = np.load('xcorr_lag_ba.npy')
pa_lag = np.load('xcorr_lag_pa.npy')

np.mean(pa_lag)
np.std(pa_lag)

X = mba_lag - ba_lag
stats.ttest_1samp(X,0)

plt.figure()
plot_err(std,'k',times0_eeg*1000)
plt.xlabel('Time (ms)')

plt.xlim([-100, 600])
