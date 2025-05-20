#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 10:08:19 2025

CB acoustic analysis demo

@author: tzcheng
"""

import numpy as np
from numpy.linalg import norm
from scipy.stats import pearsonr
from scipy import signal
from scipy.io import wavfile
import pandas as pd
import copy
import random
import mne
from mne_connectivity import spectral_connectivity_epochs, spectral_connectivity_time
from mne_connectivity.viz import plot_connectivity_circle
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pandas

#%%######################################## load the wav file 
Wn = 50
fs_new = 1000
hop_length = 100

root_path = '/media/tzcheng/storage/babbles/babbles/'
fs, audio = wavfile.read(root_path + 'babble009.wav') 

## Get the envelope
# Hilbert 
target_env = np.abs(signal.hilbert(audio))
    
## Low-pass filtering 
b, a = signal.butter(4, Wn, fs = fs, btype='lowpass')
env_lp =1.3* signal.filtfilt(b, a, target_env)
plt.figure()
plt.plot(np.linspace(0,len(audio)/fs, len(audio)),audio,'k',linewidth=0.5)
plt.plot(np.linspace(0,len(env_lp)/fs, len(env_lp)),env_lp,'orange',linewidth=2)
plt.legend(['Raw waveform','Envelope'])

## AM spectrum
fmin = 0
fmax = 10
psds, freqs = mne.time_frequency.psd_array_welch(
env_lp,fs, 
# n_fft=len(env_lp),
n_fft=30000,
n_overlap=0,
n_per_seg=None,
fmin=fmin,
fmax=fmax,)

plt.figure()
plt.plot(freqs,psds)
plt.xlim([fmin,fmax])

## spectrogram
audio, fs = librosa.load(root_path + 'babble009.wav')
D = librosa.stft(audio)  # STFT of y
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# show the time
fig, ax = plt.subplots()
img = librosa.display.specshow(S_db,x_axis='time', y_axis='log', ax=ax)
fig.colorbar(img, ax=ax, format="%+2.f dB")

# show the tempo
fig, ax = plt.subplots()
img = librosa.display.specshow(S_db, x_axis='fourier_tempo', y_axis='log', ax=ax)
fig.colorbar(img, ax=ax, format="%+2.f dB")