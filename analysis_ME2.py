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

#%%####################################### Load the files
fs, random_audio = wavfile.read('/media/tzcheng/storage/ME2_MEG/Zoe_analyses/Stimuli/Random.wav')
fs, duple_audio = wavfile.read('/media/tzcheng/storage/ME2_MEG/Zoe_analyses/Stimuli/Duple300.wav')
fs, triple_audio = wavfile.read('/media/tzcheng/storage/ME2_MEG/Zoe_analyses/Stimuli/Triple300.wav')


#%%####################################### Frequency analysis: Are there peak amplitude in the beat and meter rates? which ROI, which source?
fmin = 0.5
fmax = 5

psds, freqs = mne.time_frequency.psd_array_welch(
random_audio,fs, # could replace with label time series
n_fft=len(random_audio),
n_overlap=0,
n_per_seg=None,
fmin=fmin,
fmax=fmax,)

# Downsample
fs_new = 1000
num_random = int((len(random_audio)*fs_new)/fs)
num_duple = int((len(duple_audio)*fs_new)/fs) 
num_triple = int((len(triple_audio)*fs_new)/fs)  
            
random_audio = signal.resample(random_audio, num_random, t=None, axis=0, window=None)
duple_audio = signal.resample(duple_audio, num_duple, t=None, axis=0, window=None)
triple_audio = signal.resample(triple_audio, num_triple, t=None, axis=0, window=None)

#%%####################################### Connectivity analysis
#%%##### 

#%%####################################### ML sliding estimator