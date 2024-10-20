#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 11:37:59 2024
Analyze the preprocessed EEG files.

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

#%%######################################## load EEG data and the wav file
root_path = '/media/tzcheng/storage/RASP/'

fs, audio = wavfile.read(root_path + 'Stimuli/Random.wav') # Random, Duple300, Triple300
epochs = mne.read_epochs(root_path + '')

#%%######################################## Extract the envelope
# Get the envelope

# Low-pass filtering 

# Downsample
fs_new = 1000
num_audio = int((len(audio)*fs_new)/fs)
audio_rs = signal.resample(audio, num_audio, t=None, axis=0, window=None)

#%%######################################## Calculate cortical tracking
# 1. create a fake channel for speech envelope
# 2. implement calculaion by myself
