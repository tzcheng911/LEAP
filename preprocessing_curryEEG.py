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
from scipy.spatial.transform import Rotation as R
# import curryreader as cr

root_path = '/media/tzcheng/storage/RASP/'

## load the data with cr_read()
# currydata = cr.read(root_path + '20240815/Acquisition_Pilot4_clicks.cdt')

## load the data with read_raw_curry()
raw0 = mne.io.read_raw_curry(root_path + '20240730/Acquisition_soundmod_Pilot3.cdt', preload=True) # Random and soundmod_Pilot3 showed more clear AEP

## rotate the layout 180 degrees
r = R.from_euler('z', 180, degrees=True)
pos = np.zeros([64,3])
pos_180 = np.zeros([64,3])
for nch in np.arange(0,64,1):
    tmp_pos = raw0.info['chs'][nch]['loc'][:3]
    pos[nch,:] = tmp_pos
    pos_180[nch,:] = r.apply(tmp_pos)
    raw0.info['chs'][nch]['loc'][:3] = r.apply(tmp_pos)
    
# montage = mne.channels.read_custom_montage(root_path + 'SynAmps2_Quik-Cap64_curryreader.loc')
# raw0.set_montage("standard_1020", match_case=False,on_missing='warn')
# raw0.plot_sensors(show_names = True,sphere="eeglab")
# raw0.plot()
raw = raw0.copy()
# bad_channels = []
# raw.rename_channels({'Fp1':'FP1','Fp2':'FP2','Fpz':'FPZ','Cb1':'CB1','Cb2':'CB2',
#                      'Fz':'FZ','FCz':'FCZ','Cz':'CZ','CPz':'CPZ','Pz':'PZ',
#                      'POz':'POZ','Oz':'OZ','FT11':'FT7','FT12':'FT8'})
# raw.drop_channels(['Trigger','F11','F12','CB1','CB2'] + bad_channels)
# raw.set_montage(montage)


#%% Get events
events,event_id = mne.events_from_annotations(raw)

#%% Do some artifact removal
## Repairing with ICA
# raw.filter(l_freq=1,h_freq=None,method='iir',iir_params=dict(order=4,ftype='butter'))
# ica = mne.preprocessing.ICA(n_components=15, max_iter="auto", random_state=97)
# ica.fit(raw)

# # ica.plot_overlay(raw, exclude=[0], picks="eeg")

# # Using an EOG channel to select ICA components
# ica.exclude = []
# # find which ICs match the EOG pattern
# ecg_indices, ecg_scores = ica.find_bads_ecg(raw,ch_name='EMG1', method="correlation", threshold="auto")
# ica.exclude.append(ecg_indices[0])
# eog_indices, eog_scores = ica.find_bads_eog(raw,ch_name='VEOG')
# ica.exclude.append(eog_indices[0])
# eog_indices, eog_scores = ica.find_bads_eog(raw,ch_name='HEOG')
# ica.exclude.append(eog_indices[0])
# raw_ica = raw.copy()
# ica.apply(raw_ica)

# # manual IC selection 
# ica.plot_scores(eog_scores)
# ica.plot_sources(raw, show_scrollbars=False)
# ica.plot_components()
# ica.plot_properties(raw, picks=[11])
# ica.exclude = [0, 1]
# reconst_raw = raw.copy()
# ica.apply(reconst_raw)

## Repairing with SSP
# ecg_epochs = mne.preprocessing.create_ecg_epochs(raw,ch_name='EMG1').average()
# ecg_epochs.plot_joint(times=[-0.25, -0.025, 0, 0.025, 0.25])
# eog_epochs = mne.preprocessing.create_eog_epochs(raw,ch_name=['VEOG','HEOG']).average()
# eog_epochs.plot_joint(times=[-0.25, -0.025, 0, 0.025, 0.25])
# ecg_projs, ecg_events = mne.preprocessing.compute_proj_ecg(raw, ch_name='EMG1', reject=None)
# eog_projs, eog_events = mne.preprocessing.compute_proj_eog(raw, ch_name=['VEOG','HEOG'], reject=None)

# raw.add_proj(ecg_projs)
# raw.add_proj(eog_projs)

#%% Referencing
raw_avg_ref = raw.copy().set_eeg_reference(ref_channels="average",ch_type='eeg', projection=True)

#%% Filtering
raw_avg_ref.filter(l_freq=1,h_freq=50,method='iir',iir_params=dict(order=4,ftype='butter'))

#%% Do epoch based on the events
# epoch = mne.Epochs(raw_avg_ref,events, event_id=1,tmin=-0.1,tmax=0.3,baseline=(-0.05,0), proj='delayed',reject=dict(eeg=100e-6))
# evoked = epoch.average()
# evoked.plot_topomap(proj='interactive')
epoch = mne.Epochs(raw_avg_ref,events, event_id=1,tmin=-0.1,tmax=0.5,baseline=(-0.05,0), proj=True,reject=dict(eeg=35-6)) # reject=dict(eeg=100e-6)
evoked = epoch.average()
#%% check the stim2 and stimtracker timing
# events_test = events[events[:,2] != 1]
# diff = np.diff(events_test[:,0])/2000*1000 # difference showed in ms
# all_diff = diff[diff<=50]
# plt.hist(all_diff, bins =5)

#%% Visualization
times = np.arange(-0.1, 0.5, 0.05)
evoked.plot(ylim = dict(eeg=[-2, 2]))
evoked.plot_topomap(times,vlim=(-1,1),show_names=False)
plt.figure()
plt.plot(evoked.times,evoked.get_data().mean(0))