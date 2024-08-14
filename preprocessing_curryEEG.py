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

root_path = '/media/tzcheng/storage/RASP/'
raw0 = mne.io.read_raw_curry(root_path + 'Acquisition_pilot2_clicks2.cdt', preload=True)
montage = mne.channels.read_custom_montage(root_path + 'SynAmps2_Quik-Cap64.loc')
# raw0.set_montage("standard_1020", match_case=False,on_missing='warn')
# raw0.plot_sensors(show_names = True,sphere="eeglab")
# raw0.plot()
raw = raw0.copy()
raw.drop_channels(['Trigger','F11','F12','Cb1','Cb2'])
raw.rename_channels({'Fp1':'FP1','Fp2':'FP2','Fpz':'FPZ',
                     'Fz':'FZ','FCz':'FCZ','Cz':'CZ','CPz':'CPZ','Pz':'PZ',
                     'POz':'POZ','Oz':'OZ','FT11':'FT7','FT12':'FT8'})
raw.set_montage(montage)

#%% Get events
events,event_id = mne.events_from_annotations(raw)

#%% Referencing
raw_avg_ref = raw.copy().set_eeg_reference(ref_channels="average",ch_type='eeg')

#%% Filtering
raw_avg_ref.filter(l_freq=0.1,h_freq=50,method='iir',iir_params=dict(order=4,ftype='butter'))

#%% Do some artifact removal
## Repairing with ICA
# filt_raw = raw.copy().filter(l_freq=1, h_freq=None)
# ica = mne.preprocessing.ICA(n_components=32, max_iter="auto", random_state=97)
# ica.fit(filt_raw)
# ica.plot_components()
# ica.plot_overlay(raw, exclude=[0], picks="eeg")

# # Using an EOG channel to select ICA components
# ica.exclude = []
# find which ICs match the EOG pattern
# eog_indices, eog_scores = ica.find_bads_eog(raw,ch_name='EMG1')
# ica.exclude = eog_indices
# ecg_indices, ecg_scores = ica.find_bads_ecg(raw,ch_name='VEOG', method="correlation", threshold="auto")
# ica.exclude = ecg_indices
# ecg_indices, ecg_scores = ica.find_bads_ecg(raw,ch_name='HEOG', method="correlation", threshold="auto")
# ica.exclude = ecg_indices
# reconst_raw = raw.copy()
# ica.apply(reconst_raw)

# # manual IC selection 
# ica.plot_sources(raw, show_scrollbars=False)
# ica.plot_components()
# ica.plot_properties(raw, picks=[0, 1])
# ica.exclude = [0, 1]
# reconst_raw = raw.copy()
# ica.apply(reconst_raw)

## Repairing with SSP
# ecg_epochs = mne.preprocessing.create_ecg_epochs(raw_avg_ref,ch_name='EMG1').average()
# ecg_epochs.plot_joint(times=[-0.25, -0.025, 0, 0.025, 0.25])
# eog_epochs = mne.preprocessing.create_eog_epochs(raw_avg_ref,ch_name=['VEOG','HEOG']).average()
# eog_epochs.plot_joint(times=[-0.25, -0.025, 0, 0.025, 0.25])
# ecg_projs, ecg_events = mne.preprocessing.compute_proj_ecg(raw_avg_ref, ch_name='EMG1', reject=None)
# eog_projs, eog_events = mne.preprocessing.compute_proj_eog(raw_avg_ref, ch_name=['VEOG','HEOG'], reject=None)

# raw_avg_ref.add_proj(ecg_projs)
# raw_avg_ref.add_proj(eog_projs)

#%% Do epoch based on the events
# epoch = mne.Epochs(raw_avg_ref,events, event_id=1,tmin=-0.1,tmax=0.3,baseline=(-0.05,0), proj='delayed',reject=dict(eeg=100e-6))
# evoked = epoch.average()
# evoked.plot_topomap(proj='interactive')
epoch = mne.Epochs(raw_avg_ref,events, event_id=1,tmin=-0.1,tmax=0.3,baseline=(-0.05,0), proj=True,reject=dict(eeg=100e-6))
evoked = epoch.average()
#%% check the stim2 and stimtracker timing
# events_test = events[events[:,2] != 1]
# diff = np.diff(events_test[:,0])/2000*1000 # difference showed in ms
# all_diff = diff[diff<=50]
# plt.hist(all_diff, bins =5)

#%% Visualization
evoked.plot_joint()