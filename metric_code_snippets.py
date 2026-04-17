#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 11:53:04 2026
Calculate power spectrum of audio and EEG
Calculate cross-correlation between audio and EEG

@author: tzcheng
"""
#%%####################################### Packages and functions needed
import mne
import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt 

def zscore_and_normalize(x):
    x = (x - np.mean(x)) / np.std(x) 
    xx = x / np.linalg.norm(x)
    return xx

#%%####################################### Load audio
fs_eeg = 5000
fs_audio, n40_audio = wavfile.read(yourownpath + 'n40.wav')

#%%####################################### Spectrum analysis
fmin = 50
fmax = 150

psds, freqs = mne.time_frequency.psd_array_welch(
    n40_audio,fs_audio,
    n_fft=len(signal),
    n_overlap=0,
    n_per_seg=None,
    fmin=fmin,
    fmax=fmax,)
plt.figure()
plt.title('Spectrum')
plt.plot(freqs,psds)
plt.xlim([60, 140])

#%%####################################### Cross correlation analysis
## Resample audio 
num = int((len(n40_audio)*fs_eeg)/fs_audio)    
audio_rs = signal.resample(n40_audio, num, t=None, axis=0, window=None)
times_audio = np.linspace(0,len(audio_rs)/fs_eeg,len(audio_rs))

## Slice audio to match the EEG
# Katherina uses the same code slicing EEG to slice audio
EEG = xxxx_win # for each of the condition
audio = yyyy_win

## Calculate the lags
lag_window_s=(-0.014, -0.007) 
lags = signal.correlation_lags(len(audio),EEG.shape[-1])
lags_s = lags/fs_eeg
lag_min, lag_max = np.array(lag_window_s)
lag_mask = (lags_s >= lag_min) & (lags_s <= lag_max)

## Calculate the xcorr
xcorr_max = []
xcorr_lag = []
for subj_resp in EEG:
    ## zscore and normalize both audio and EEG signals
    audio_z = zscore_and_normalize(audio)
    resp_z = zscore_and_normalize(subj_resp)
    
    ## Calculate the xcorr
    xcorr = np.abs(signal.correlate(audio_z, resp_z, mode="full"))
    xcorr_win = xcorr[lag_mask]
    lag_win = lags_s[lag_mask]
    idx = np.argmax(xcorr_win)
    xcorr_max.append(xcorr_win[idx])
    xcorr_lag.append(lag_win[idx])

print(np.mean(xcorr_max))
print(np.mean(xcorr_lag))