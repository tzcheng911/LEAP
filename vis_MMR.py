#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 11:31:16 2023

Used to visualize MMR subplots 

@author: tzcheng
"""

import mne
import matplotlib
import matplotlib.pyplot as plt 
import numpy as np
import os
from nilearn.plotting import plot_glass_brain
from scipy import stats,signal

def plot_err(group_stc,color,t):
    group_avg=np.mean(group_stc,axis=0)
   #plt.figure()
    err=np.std(group_stc,axis=0)/np.sqrt(group_stc.shape[0])
    up=group_avg+err
    lw=group_avg-err
    t=np.linspace(-100,600,3501)
    plt.plot(t,group_avg,color=color)
    plt.fill_between(t,up,lw,color=color,alpha=0.5)

#%%#######################################   
root_path='/media/tzcheng/storage/CBS/'
subjects_dir = '/media/tzcheng/storage2/subjects/'

## Load vertex
stc1 = mne.read_source_estimate(root_path + 'cbs_A101/sss_fif/cbs_A101_mmr2_morph-vl.stc')
MEG_mmr1_v = np.load(root_path + 'cbsA_meeg_analysis/group_mmr1_vector_morph.npy') # with the mag or vector method
MEG_mmr2_v = np.load(root_path + 'cbsA_meeg_analysis/group_mmr2_vector_morph.npy') # with the mag or vector method
MEG_mmr1_m = np.load(root_path + 'cbsA_meeg_analysis/group_mmr1_None_morph.npy') # with the mag or vector method
MEG_mmr2_m = np.load(root_path + 'cbsA_meeg_analysis/group_mmr2_None_morph.npy') # with the mag or vector method
EEG_mmr1 = np.load(root_path + 'cbsA_meeg_analysis/group_mmr1_eeg.npy')
EEG_mmr2 = np.load(root_path + 'cbsA_meeg_analysis/group_mmr2_eeg.npy')

## Load ROIs
MEG_mmr1_roi_v = np.load(root_path + 'cbsA_meeg_analysis/group_mmr1_vector_morph_roi.npy') # with the mag or vector method
MEG_mmr2_roi_v = np.load(root_path + 'cbsA_meeg_analysis/group_mmr2_vector_morph_roi.npy') # with the mag or vector method
MEG_mmr1_roi_m = np.load(root_path + 'cbsA_meeg_analysis/group_mmr1_None_morph_roi.npy') # with the mag or vector method
MEG_mmr2_roi_m = np.load(root_path + 'cbsA_meeg_analysis/group_mmr2_None_morph_roi.npy') # with the mag or vector method


subject = 'fsaverage'
src = mne.read_source_spaces(subjects_dir + subject + '/bem/fsaverage-vol-5-src.fif')
fname_aseg = subjects_dir + subject + '/mri/aparc+aseg.mgz'
label_names = np.asarray(mne.get_volume_labels_from_aseg(fname_aseg))
label_names
lh_ROI_label = [60,61,62,72] # STG and IFG (parsopercularis, parsorbitalis, parstriangularis)
rh_ROI_label = [96,97,98,108] # STG and IFG (parsopercularis, parsorbitalis, parstriangularis)

## visualization average
times = stc1.times
plt.figure()
plt.subplot(311)
plot_err(EEG_mmr1,'grey',stc1.times)
plot_err(EEG_mmr2,'k',stc1.times)
plt.title('Traditional direction')
plt.xlim([-100,600])

plt.subplot(312)
plot_err(MEG_mmr1_m.mean(axis=1),'c',stc1.times)
plot_err(MEG_mmr2_m.mean(axis=1),'b',stc1.times)
plt.xlim([-100,600])
plt.ylabel('Amplitude')

plt.subplot(313)
plot_err(MEG_mmr1_v.mean(axis=1),'m',stc1.times)
plot_err(MEG_mmr2_v.mean(axis=1),'r',stc1.times)
plt.xlim([-100,600])
plt.xlabel('Time (ms)')

## visualization ROI
plt.figure()
plt.subplot(311)
plot_err(EEG_mmr1,'grey',stc1.times)
plot_err(EEG_mmr2,'k',stc1.times)
plt.title('Traditional direction')
plt.xlim([-100,600])

plt.subplot(312)
plot_err(MEG_mmr1_roi_m[:,lh_ROI_label,:].mean(axis=1),'c',stc1.times)
plot_err(MEG_mmr2_roi_m[:,lh_ROI_label,:].mean(axis=1),'b',stc1.times)
plt.xlim([-100,600])
plt.ylabel('Amplitude')

plt.subplot(313)
plot_err(MEG_mmr1_roi_v[:,lh_ROI_label,:].mean(axis=1),'m',stc1.times)
plot_err(MEG_mmr2_roi_v[:,lh_ROI_label,:].mean(axis=1),'r',stc1.times)
plt.xlim([-100,600])
plt.xlabel('Time (ms)')

#%%####################################### check the evoked for first /ba/, /pa/ and /mba/
## Note that the result could be slightly different because /ba/ was random sampled
root_path='/media/tzcheng/storage/CBS/'
stc1 = mne.read_source_estimate(root_path + 'cbs_A101/sss_fif/cbs_A101_mmr2_morph-vl.stc')
times = stc1.times

EEG_1pa = np.load(root_path + 'cbsA_meeg_analysis/group_dev2_eeg.npy')
EEG_1mba = np.load(root_path + 'cbsA_meeg_analysis/group_dev1_eeg.npy')
EEG_1ba = np.load(root_path + 'cbsA_meeg_analysis/group_dev_reverse_eeg.npy')

EEG_endpa = np.load(root_path + 'cbsA_meeg_analysis/group_std2_reverse_eeg.npy')
EEG_endmba = np.load(root_path + 'cbsA_meeg_analysis/group_std1_reverse_eeg.npy')
EEG_endba = np.load(root_path + 'cbsA_meeg_analysis/group_std_eeg.npy')

plt.figure()
plot_err(EEG_1ba,'r',stc1.times)
plot_err(EEG_1mba,'g',stc1.times)
plot_err(EEG_1pa,'b',stc1.times)
plt.title('EEG Evoked response for ba, mba and pa')
plt.legend(['ba','','mba','','pa',''])
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.xlim([-100,600])

plt.figure()
plot_err(EEG_endba,'r',stc1.times)
plot_err(EEG_endmba,'g',stc1.times)
plot_err(EEG_endpa,'b',stc1.times)
plt.title('EEG Evoked response for ba, mba and pa')
plt.legend(['ba','','mba','','pa',''])
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.xlim([-100,600])

plt.figure()
plot_err(EEG_endmba,'r',stc1.times)
plot_err(EEG_1ba,'b',stc1.times)
plot_err(EEG_1ba-EEG_endmba,'k',stc1.times)
plt.legend(['std','','dev','','MMR',''])
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.xlim([-100,600])