#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 13:03:30 2023
Used for analyzing Curry EEG system. 
 
@author: tzcheng
"""

import mne 
import matplotlib.pyplot as plt
import numpy as np

raw = mne.io.read_raw_curry('/home/tzcheng/Downloads/Acquisition_cz01.cdt', preload=True)
raw = mne.io.read_raw_fif('/home/tzcheng/Downloads/irregular_cz.fif', preload=True)

events,event_id = mne.events_from_annotations(raw)
raw_avg_ref = raw.copy().set_eeg_reference(ref_channels="average")
raw_avg_ref.filter(l_freq=0,h_freq=50,method='iir',iir_params=dict(order=4,ftype='butter'))
epoch = mne.Epochs(raw_avg_ref,events, event_id=1,tmin=-0.1,tmax=0.5,baseline=(-0.1,0))
evoked = epoch.average()
evoked.plot()
