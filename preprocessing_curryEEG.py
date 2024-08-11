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

raw0 = mne.io.read_raw_curry('/media/tzcheng/storage/RASP/20240730/Acquisition_soundmod_Pilot3.cdt', preload=True)
raw0.set_montage("standard_1020", match_case=False,on_missing='warn')
raw0.plot_sensors(show_names = True,sphere="eeglab")
# raw0.plot()
raw = raw0.copy()
#%% Do some artifact removal
raw.drop_channels(["FT11","Cz","CPz"])

## Repairing with ICA
filt_raw = raw.copy().filter(l_freq=0.1, h_freq=None)

ica = mne.preprocessing.ICA(n_components=15, max_iter="auto", random_state=97)
ica.fit(filt_raw)
ica
explained_var_ratio = ica.get_explained_variance_ratio(filt_raw)
raw.load_data()

# Using an EOG channel to select ICA components
ica.exclude = []
# find which ICs match the EOG pattern
eog_indices, eog_scores = ica.find_bads_eog(raw)
ica.exclude = eog_indices
ecg_indices, ecg_scores = ica.find_bads_ecg(raw, method="correlation", threshold="auto")
ica.exclude = ecg_indices
reconst_raw = raw.copy()
ica.apply(reconst_raw)

# manual IC selection 
ica.plot_sources(raw, show_scrollbars=False)
ica.plot_components()
ica.plot_properties(raw, picks=[0, 1])
ica.exclude = [0, 1]
reconst_raw = raw.copy()
ica.apply(reconst_raw)

## Repairing with SSP
ecg_epochs = mne.preprocessing.create_ecg_epochs(raw,ch_name='EMG1').average()
ecg_epochs.plot_joint(times=[-0.25, -0.025, 0, 0.025, 0.25])
eog_epochs = mne.preprocessing.create_eog_epochs(raw,ch_name=['VEOG','HEOG']).average()
eog_epochs.plot_joint(times=[-0.25, -0.025, 0, 0.025, 0.25])
ecg_projs, ecg_events = mne.preprocessing.compute_proj_ecg(raw, ch_name='EMG1', reject=None)
eog_projs, eog_events = mne.preprocessing.compute_proj_eog(raw, ch_name=['VEOG','HEOG'], reject=None)

raw.add_proj(ecg_projs)
raw.add_proj(eog_projs)

#%% Get events
events,event_id = mne.events_from_annotations(raw)

#%% Referencing
raw_avg_ref = raw.copy().set_eeg_reference(ref_channels="average",ch_type='eeg')

#%% Filtering
raw_avg_ref.filter(l_freq=0.1,h_freq=50,method='iir',iir_params=dict(order=4,ftype='butter'))

#%% Do epoch based on the events
epoch = mne.Epochs(raw_avg_ref,events, event_id=2,tmin=-0.1,tmax=0.5,baseline=(-0.1,0))
evoked = epoch.average()

#%% check the stim2 and stimtracker timing
events_test = events[events[:,2] != 1]
diff = np.diff(events_test[:,0])/2000*1000 # difference showed in ms
all_diff = diff[diff<=50]
plt.hist(all_diff, bins =5)

#%% Visualization
epoch.plot_image(picks="eeg", combine="mean")
evoked.plot_joint()