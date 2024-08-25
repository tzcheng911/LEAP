#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 11:31:16 2023

Used to visualize MMR Figure 3 - 5. 

@author: tzcheng
"""

import mne
import matplotlib.pyplot as plt 
import numpy as np
from scipy.stats import pearsonr
import pandas as pd

def plot_err(group_stc,color,t):
    group_avg=np.mean(group_stc,axis=0)
    err=np.std(group_stc,axis=0)/np.sqrt(group_stc.shape[0])
    up=group_avg+err
    lw=group_avg-err
    plt.plot(t,group_avg,color=color)
    plt.fill_between(t,up,lw,color=color,alpha=0.5)

root_path='/home/tzcheng/Documents/GitHub/Paper1_MMR/data/'
subjects_dir = '/home/tzcheng/Documents/GitHub/Paper1_MMR/subjects/'
subject = 'fsaverage'
src = mne.read_source_spaces(subjects_dir + 'fsaverage/bem/fsaverage-vol-5-src.fif')
times = np.linspace(-0.1,0.6,3501)

#%%####################################### Load the files
MEG_mmr = np.load(root_path + 'adults/adult_group_mmr2_vector_morph.npy') # with the mag or vector method
EEG_mmr = np.load(root_path + 'adults/adult_group_mmr2_eeg.npy')

#%%####################################### Correlation analysis Figure 3ab
mean_MEG_mmr = MEG_mmr.mean(axis=1)
r_all_t = []
for t in np.arange(0,len(times),1):
    r,p = pearsonr(mean_MEG_mmr[:,t],EEG_mmr[:,t])
    r_all_t.append(r)

fig, ax = plt.subplots(1)

ax.plot(times, r_all_t)
ax.axhline(0, color="k", linestyle="--")
ax.axvline(0, color="k")
plt.title('MMR')
plt.legend(['MEG'])
plt.xlabel('Time (s)')
plt.ylabel('Pearson r')
plt.xlim([-0.05,0.45])

plt.figure()
plt.scatter(mean_MEG_mmr[:,1300],EEG_mmr[:,1300])  # look up the corresponding sample points from times
plt.xlabel('MEG')
plt.ylabel('EEG')
plt.title('t = 0.16 s')

#%%####################################### Correlation analysis Figure 3c
xcorr = pd.read_pickle(root_path + 'adults/df_xcorr_MEGEEG_mmr2.pkl')
xcorr_mean = xcorr.groupby('Vertno').mean()

# Visualization
v_hack = pd.concat([xcorr_mean["XCorr MEG & EEG"],xcorr_mean["XCorr MEG & EEG"]],axis=1)
stc = mne.read_source_estimate(subjects_dir + 'cbs_A101_mmr2_vector_morph-vl.stc')
stc.data = v_hack
stc.plot(src,clim=dict(kind="percent",lims=[90,95,99]),subject=subject, subjects_dir=subjects_dir)

#%%####################################### Decoding analysis Figure 4abef
## Figure 4ab Conventional MMR: ba to mba vs. ba to pa
scores_observed = np.load(root_path + 'adults/adult_scores_conv_morph_kall.npy')
patterns = np.load(root_path +'adults/adult_patterns_conv_morph_kall.npy')
scores_permute = np.load(root_path +'adults/adult_vector_scores_100perm_kall_conv.npz')

## Figure 4ef Controlled MMR: first - last mba vs. first pa - last pa
scores_observed = np.load(root_path + 'adults/adult_scores_conv_morph_kall.npy')
patterns = np.load(root_path + 'adults/adult_patterns_conv_morph_kall.npy')
scores_permute = np.load(root_path + 'adults/adult_vector_scores_100perm_kall_conv.npz')

#%%####################################### Decoding analysis Figure 5abef
## Figure 5ab Conventional MMR: ba to mba vs. ba to pa
scores_observed = np.load(root_path + 'infants/baby_scores_conv_morph_kall.npy')
patterns = np.load(root_path +'infants/baby_patterns_conv_morph_kall.npy')
scores_permute = np.load(root_path +'infants/baby_vector_scores_100perm_kall_conv.npz')

## Figure 5ef Controlled MMR: first - last mba vs. first pa - last pa
scores_observed = np.load(root_path + 'infants/baby_scores_conv_morph_kall.npy')
patterns = np.load(root_path + 'infants/baby_patterns_conv_morph_kall.npy')
scores_permute = np.load(root_path + 'infants/baby_vector_scores_100perm_kall_conv.npz')

#%% Plot decoding accuracy across time
fig, ax = plt.subplots(1)
ax.plot(times[250:2750], scores_observed.mean(0), label="score")
ax.plot(scores_permute['peaks_time'],np.percentile(scores_permute['scores_perm_array'],95,axis=0),'g.')
ax.axhline(0.5, color="k", linestyle="--", label="chance")
ax.axhline(np.percentile(scores_observed.mean(0),q = 95), color="grey", linestyle="--", label="95 percentile")
ax.axvline(0, color="k")
plt.xlabel('Time (s)')
plt.xlim([-0.05,0.45])
plt.ylim([0,1])

#%% Plot patterns across time
src = mne.read_source_spaces(subjects_dir + 'fsaverage/bem/fsaverage-vol-5-src.fif')
stc = mne.read_source_estimate(subjects_dir + 'cbs_A101_mmr2_vector_morph-vl.stc')
stc_crop = stc.copy().crop(tmin= -0.05, tmax=0.45)
stc_crop.data = patterns
stc_crop.plot(src, clim=dict(kind="percent",pos_lims=[90,95,99]), subject='fsaverage', subjects_dir=subjects_dir)

#%%####################################### Decoding analysis Figure 4cdgh
## Conventional MMR: ba to mba vs. ba to pa
MEG_mmr1 = np.load(root_path + 'adults/adult_group_mmr1_vector_morph.npy')
MEG_mmr2 = np.load(root_path + 'adults/adult_group_mmr2_vector_morph.npy')

## Figrue 4c
plt.figure()
plot_err(np.squeeze(MEG_mmr1.mean(axis = 1)),'c',times)
plot_err(np.squeeze(MEG_mmr2.mean(axis = 1)),'b',times)
plt.legend(['MMR1','','MMR2',''])
plt.xlabel('Time (s)')
plt.title('Conventional MMR')
plt.xlim([-0.05, 0.45])

## Figrue 4d
v_ind = np.where(src[0]['vertno'] == 25056) # conv_adult 25056, cont_adult 20041, conv_infant 23669, cont_infant 19842 
plt.figure()
plot_err(np.squeeze(MEG_mmr1[:,v_ind,:]),'c',times)
plot_err(np.squeeze(MEG_mmr2[:,v_ind,:]),'b',times)
plt.legend(['MMR1','','MMR2',''])
plt.xlabel('Time (s)')
plt.title('Conventional MMR v25056')
plt.xlim([-0.05, 0.45])

## Controlled MMR: first - last mba vs. first pa - last pa
MEG_mmr1 = np.load(root_path + 'adults/adult_group_mmr1_mba_vector_morph.npy')
MEG_mmr2 = np.load(root_path + 'adults/adult_group_mmr2_pa_vector_morph.npy')

## Figrue 4g
plt.figure()
plot_err(np.squeeze(MEG_mmr1.mean(axis = 1)),'c',times)
plot_err(np.squeeze(MEG_mmr2.mean(axis = 1)),'b',times)
plt.legend(['MMR1','','MMR2',''])
plt.xlabel('Time (s)')
plt.title('Controlled MMR')
plt.xlim([-0.05, 0.45])

## Figrue 4h
v_ind = np.where(src[0]['vertno'] == 20041) # conv_adult 25056, cont_adult 20041, conv_infant 23669, cont_infant 19842 
plt.figure()
plot_err(np.squeeze(MEG_mmr1[:,v_ind,:]),'c',times)
plot_err(np.squeeze(MEG_mmr2[:,v_ind,:]),'b',times)
plt.legend(['MMR1','','MMR2',''])
plt.xlabel('Time (s)')
plt.title('Controlled MMR v20041')
plt.xlim([-0.05, 0.45])

#%%####################################### Decoding analysis Figure 5cdgh
## Conventional MMR: ba to mba vs. ba to pa
MEG_mmr1 = np.load(root_path + 'infants/baby_group_mmr1_vector_morph.npy')
MEG_mmr2 = np.load(root_path + 'infants/baby_group_mmr2_vector_morph.npy')

## Figrue 5c
plt.figure()
plot_err(np.squeeze(MEG_mmr1.mean(axis = 1)),'c',times)
plot_err(np.squeeze(MEG_mmr2.mean(axis = 1)),'b',times)
plt.legend(['MMR1','','MMR2',''])
plt.xlabel('Time (s)')
plt.title('Conventional MMR')
plt.xlim([-0.05, 0.45])

## Figrue 5d
v_ind = np.where(src[0]['vertno'] == 23669) # conv_adult 25056, cont_adult 20041, conv_infant 23669, cont_infant 19842 
plt.figure()
plot_err(np.squeeze(MEG_mmr1[:,v_ind,:]),'c',times)
plot_err(np.squeeze(MEG_mmr2[:,v_ind,:]),'b',times)
plt.legend(['MMR1','','MMR2',''])
plt.xlabel('Time (s)')
plt.title('Conventional MMR v23669')
plt.xlim([-0.05, 0.45])

## Controlled MMR: first - last mba vs. first pa - last pa
MEG_mmr1 = np.load(root_path + 'infants/baby_group_mmr1_mba_vector_morph.npy')
MEG_mmr2 = np.load(root_path + 'infants/baby_group_mmr2_pa_vector_morph.npy')

## Figrue 5g
plt.figure()
plot_err(np.squeeze(MEG_mmr1.mean(axis = 1)),'c',times)
plot_err(np.squeeze(MEG_mmr2.mean(axis = 1)),'b',times)
plt.legend(['MMR1','','MMR2',''])
plt.xlabel('Time (s)')
plt.title('Controlled MMR')
plt.xlim([-0.05, 0.45])

## Figrue 5h
v_ind = np.where(src[0]['vertno'] == 22417) # conv_adult 25056, cont_adult 20041, conv_infant 23669, cont_infant 19842 
plt.figure()
plot_err(np.squeeze(MEG_mmr1[:,v_ind,:]),'c',times)
plot_err(np.squeeze(MEG_mmr2[:,v_ind,:]),'b',times)
plt.legend(['MMR1','','MMR2',''])
plt.xlabel('Time (s)')
plt.title('Controlled MMR v22417')
plt.xlim([-0.05, 0.45])

#%%####################################### sensor level MEG
from mne.viz import plot_evoked_topo
MEG_mmr1_sensor = np.load('/media/tzcheng/storage2/CBS/cbsA_meeg_analysis/MEG/MMR/sensor/group_mmr1_sensor.npy',allow_pickle=True)
MEG_mmr2_sensor = np.load('/media/tzcheng/storage2/CBS/cbsA_meeg_analysis/MEG/MMR/sensor/group_mmr2_sensor.npy',allow_pickle=True)
MEG_dev1_sensor = np.load('/media/tzcheng/storage2/CBS/cbsA_meeg_analysis/MEG/MMR/sensor/group_dev1_sensor.npy',allow_pickle=True)
MEG_dev2_sensor = np.load('/media/tzcheng/storage2/CBS/cbsA_meeg_analysis/MEG/MMR/sensor/group_dev2_sensor.npy',allow_pickle=True)
MEG_std_sensor = np.load('/media/tzcheng/storage2/CBS/cbsA_meeg_analysis/MEG/MMR/sensor/group_std_sensor.npy',allow_pickle=True)

evoked = mne.read_evokeds('/media/tzcheng/storage2/CBS/cbs_A108/sss_fif/cbs_A108_01_otp_raw_sss_proj_fil50_evoked_dev1_mmr.fif')[0]
mmr1 = evoked.copy()
mmr1.data = MEG_mmr1_sensor.mean(0)
dev1 = evoked.copy()
dev1.data = MEG_dev1_sensor.mean(0)
mmr2 = evoked.copy()
mmr2.data = MEG_mmr2_sensor.mean(0)
dev2 = evoked.copy()
dev2.data = MEG_dev2_sensor.mean(0)
std = evoked.copy()
std.data = MEG_std_sensor.mean(0)

evokeds = [mmr1, dev1, std]
plot_evoked_topo(evokeds,background_color="w")
plt.show()
plt.legend(['','mmr','dev','std'])

evokeds = [mmr2, dev2, std]
plot_evoked_topo(evokeds,background_color="w")
plt.show()
plt.legend(['','mmr','dev','std'])