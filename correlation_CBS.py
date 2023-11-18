#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 13:23:58 2023
Visualize and calculate EEG & MEG correlation in adults for MMR and Standard.

Correlation between EEG & MEG time series for...
1. averaged across all subjects and all vertices
2. averaged across all vertices for each subject (mean, max of window; point-by-point correlation)
3. averaged for each ROI
4. averaged for each vertice

Correlation methods: pearson r, xcorr, cosine similarity

@author: tzcheng
"""

import mne
import numpy as np
from numpy import dot
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy import stats,signal
import scipy as sp
import os
import seaborn as sns
import pandas as pd
import scipy.stats as stats
from scipy.io import wavfile

root_path='/media/tzcheng/storage/CBS/'
subjects_dir = '/media/tzcheng/storage2/subjects/'

## load source
subject = 'fsaverage'
src = mne.read_source_spaces(subjects_dir + 'fsaverage/bem/fsaverage-vol-5-src.fif')

## load ROI label 
fname_aseg = subjects_dir + subject + '/mri/aparc+aseg.mgz'
label_names = np.asarray(mne.get_volume_labels_from_aseg(fname_aseg))
lh_ROI_label = [60,61,62,72] # STG and IFG (parsopercularis, parsorbitalis, parstriangularis)
rh_ROI_label = [96,97,98,108] # STG and IFG (parsopercularis, parsorbitalis, parstriangularis)

def plot_err(group_data,color,t):
    group_avg=np.mean(group_data,axis=0)
   #plt.figure()
    err=np.std(group_data,axis=0)/np.sqrt(group_data.shape[0])
    up=group_avg+err
    lw=group_avg-err
    t=np.linspace(-100,600,3501)
    plt.plot(t,group_avg,color=color)
    plt.fill_between(t,up,lw,color=color,alpha=0.5)
    
#%%######################################## load data MMR
# MMR
stc1 = mne.read_source_estimate(root_path + 'cbs_A101/sss_fif/cbs_A101_mmr2_morph-vl.stc')
stc2 = mne.read_source_estimate(root_path + 'cbs_A101/sss_fif/cbs_A101_mmr2_morph-vl.stc')
MEG_mmr1_v = np.load(root_path + 'cbsA_meeg_analysis/MEG/vector_method/group_mmr1_vector_morph.npy') # with the mag or vector method
MEG_mmr2_v = np.load(root_path + 'cbsA_meeg_analysis/MEG/vector_method/group_mmr2_vector_morph.npy') # with the mag or vector method
MEG_mmr1_m = np.load(root_path + 'cbsA_meeg_analysis/MEG/magnitude_method/group_mmr1_None_morph.npy') # with the mag or vector method
MEG_mmr2_m = np.load(root_path + 'cbsA_meeg_analysis/MEG/magnitude_method/group_mmr2_None_morph.npy') # with the mag or vector method
MEG_mmr1_roi_v = np.load(root_path + 'cbsA_meeg_analysis/MEG/vector_method/group_mmr1_vector_morph_roi.npy') # with the mag or vector method
MEG_mmr2_roi_v = np.load(root_path + 'cbsA_meeg_analysis/MEG/vector_method/group_mmr2_vector_morph_roi.npy') # with the mag or vector method
MEG_mmr1_roi_m = np.load(root_path + 'cbsA_meeg_analysis/MEG/magnitude_method/group_mmr1_None_morph_roi.npy') # with the mag or vector method
MEG_mmr2_roi_m = np.load(root_path + 'cbsA_meeg_analysis/MEG/magnitude_method/group_mmr2_None_morph_roi.npy') # with the mag or vector method
EEG_mmr1 = np.load(root_path + 'cbsA_meeg_analysis/EEG/group_mmr1_eeg.npy')
EEG_mmr2 = np.load(root_path + 'cbsA_meeg_analysis/EEG/group_mmr2_eeg.npy')

## std
# Hypothesis: N1 is more sensory, so should observe the highest correlation in the auditory area
# stc1 = mne.read_source_estimate(root_path + 'cbs_A101/sss_fif/cbs_A101_std_vector_morph-vl.stc')
# MEG_std_v = np.load(root_path + 'cbsA_meeg_analysis/group_std_vector_morph.npy') # with the mag or vector method
# MEG_std_roi_v = np.load(root_path + 'cbsA_meeg_analysis/group_std_vector_morph_roi.npy') # with the mag or vector method
# EEG_std = np.load(root_path + 'cbsA_meeg_analysis/group_std_eeg.npy',allow_pickle=True)

#%%######################################## MEG Grand average across sources and subjects
ts = 1000 # 100 ms
te = 1750 # 250 ms

## averaged across all sources
mean_EEG_mmr2 = EEG_mmr2.mean(axis =0)
stc_m = MEG_mmr2_m.mean(axis =0).mean(axis=0)
stc_v = MEG_mmr2_v.mean(axis =0).mean(axis=0)
mean_MEG_mmr2_v = MEG_mmr2_v.mean(axis =0).mean(axis=0)
mean_MEG_mmr2_m = MEG_mmr2_m.mean(axis =0).mean(axis=0)

# pearson r 
# Tradtional direction: TOI [100 250] 0.89 for MEG_m & EEG and -0.63 for MEG_v & EEG 
# New direction: TOI [100 300] -0.40 for MEG_m & EEG and -0.26 for MEG_v & EEG 
corr_p = pearsonr(mean_EEG_mmr2[ts:te], mean_MEG_mmr2_m[ts:te])

# normalized cross correlation [-1 1]
# based on https://stackoverflow.com/questions/53436231/normalized-cross-correlation-in-python
# Tradtional direction: TOI [100 250] 0.91 (lag=0) for MEG_m & EEG and 0.33 (lage = 81.2 ms) for MEG_v & EEG
a = mean_EEG_mmr2[ts:te]
b = stc_v[ts:te]

norm_a = np.linalg.norm(a)
a = a / norm_a
norm_b = np.linalg.norm(b)
b = b / norm_b

xcorr = signal.correlate(a,b)
print(max(abs(xcorr)))
lags = signal.correlation_lags(len(a),len(b))
lags_time = lags/5000
print(lags_time[np.argmax(abs(xcorr))]) # if negative then the b is shifted left, if positive b is shifted right

# cosine similarity
cos_sim = dot(mean_EEG_mmr2[ts:te],mean_MEG_mmr2_m[ts:te]) / (norm(mean_EEG_mmr2[ts:te])*norm(mean_MEG_mmr2_m[ts:te]))
print(cos_sim)
cos_sim = dot(mean_EEG_mmr2[ts:te],stc_v[ts:te]) / (norm(mean_EEG_mmr2[ts:te])*norm(stc_v[ts:te]))
print(cos_sim)

#%%######################################## Done within each subject: correlate one value (e.g. mean or max in a window) between EEG & MEG
## pearson correlation (corr) between EEG and MEG
#%% [poor method] get the mean
## no correlation observed between EEG and MEG_m (better in window [100 200] ms)
stc_m = MEG_mmr2_m[:,:,ts:te].mean(axis=-1).mean(axis=-1)
stc_v = MEG_mmr2_v[:,:,ts:te].mean(axis=-1).mean(axis=-1)
mean_EEG_mmr2 = EEG_mmr2[:,ts:te].mean(axis=-1)

r,p = pearsonr(stc_m,mean_EEG_mmr2)
print(r,p)
r,p = pearsonr(stc_v,mean_EEG_mmr2)
print(r,p)
r,p = pearsonr(stc_v,stc_m)
print(r,p)

#%% [poor method] get the max (peak) or min (trough) point
## no correlation observed between EEG and MEG_m (better in window [100 200] ms)
max_MEG_mmr2_m = MEG_mmr2_m[:,:,ts:te].mean(axis=1).max(axis=-1)
max_MEG_mmr2_v = MEG_mmr2_v[:,:,ts:te].mean(axis=1).max(axis=-1)
max_EEG_mmr2 = EEG_mmr2[:,ts:te].max(axis=-1)

min_MEG_mmr2_m = MEG_mmr2_m[:,:,ts:te].mean(axis=1).min(axis=-1)
min_MEG_mmr2_v = MEG_mmr2_v[:,:,ts:te].mean(axis=1).min(axis=-1)
min_EEG_mmr2 = EEG_mmr2[:,ts:te].min(axis=-1)

ext_MEG_mmr2_m = []
ext_MEG_mmr2_v = []
ext_EEG_mmr2 = []
for i in np.arange(0,len(max_MEG_mmr2_m),1):
    if max_MEG_mmr2_m[i] > abs(min_MEG_mmr2_m[i]):
        ext_MEG_mmr2_m.append(max_MEG_mmr2_m[i])
    else:
        ext_MEG_mmr2_m.append(min_MEG_mmr2_m[i])

for i in np.arange(0,len(max_MEG_mmr2_m),1):
    if max_MEG_mmr2_v[i] > abs(min_MEG_mmr2_v[i]):
        ext_MEG_mmr2_v.append(max_MEG_mmr2_v[i])
    else:
        ext_MEG_mmr2_v.append(min_MEG_mmr2_v[i])
        
for i in np.arange(0,len(min_EEG_mmr2),1):
    if max_EEG_mmr2[i] > abs(min_EEG_mmr2[i]):
        ext_EEG_mmr2.append(max_EEG_mmr2[i])
    else:
        ext_EEG_mmr2.append(min_EEG_mmr2[i])

r,p = pearsonr(ext_MEG_mmr2_m,ext_EEG_mmr2)
print(r,p)
r,p = pearsonr(ext_MEG_mmr2_v,ext_EEG_mmr2)
print(r,p)

#%% [ok method] Point-by-point correlation between EEG & MEG
stc_m = MEG_mmr2_m.mean(axis=1) # change the name to std or other data
stc_v = MEG_mmr2_v.mean(axis=1)
EEG = EEG_mmr2

r_m_all_t = []
r_v_all_t = []

for t in np.arange(0,len(stc1.times),1):
    r,p = pearsonr(stc_m[:,t],EEG[:,t])
    r_m_all_t.append(r)
    r,p = pearsonr(stc_v[:,t],EEG[:,t])
    r_v_all_t.append(r)

fig, ax = plt.subplots(1)
ax.plot(stc1.times, r_m_all_t)
ax.plot(stc1.times, r_v_all_t)
ax.axhline(0, color="k", linestyle="--")
ax.axvline(0, color="k")
plt.title('MMR2')
plt.legend(['MEG_m','MEG_v'])
plt.xlabel('Time (s)')
plt.ylabel('Pearson r')
plt.xlim([-0.1,0.6])

plt.figure()
plt.scatter(stc_m[:,1300],EEG[:,1300])
plt.xlabel('MEG')
plt.ylabel('EEG')
plt.title('t = 160ms')

#%%######################################## Done within each subject: correlate time series in a window between EEG & MEG
## pearson correlation (corr) between EEG and MEG
stc_m = MEG_mmr2_m[:,:,ts:te].mean(axis=1)
stc_v = MEG_mmr2_v[:,:,ts:te].mean(axis=1)
EEG = EEG_mmr2

# [good method] get the pearson correlation of the whole time window
r_m_all_s = []
r_v_all_s = []

for s in np.arange(0,len(MEG_mmr2_m),1):
    r,p = pearsonr(stc_m[s,:],EEG[s,ts:te])
    r_m_all_s.append(r)
    r,p = pearsonr(stc_v[s,:],EEG[s,ts:te])
    r_v_all_s.append(r)

print('abs corr between MEG_m & EEG:' + str(np.abs(r_m_all_s).mean()))
print('abs corr between MEG_v & EEG:' + str(np.abs(r_v_all_s).mean()))
X = np.abs(r_v_all_s) - np.abs(r_m_all_s)
stats.ttest_1samp(X,0)

## corr between EEG and MEG ROI
r_all_s = []

for s in np.arange(0,len(MEG_mmr2_m),1):
    for nROI in np.arange(0,len(label_names),1):
        rm,p = pearsonr(MEG_mmr2_roi_m[s,nROI,ts:te],EEG[s,ts:te])
        rv,p = pearsonr(MEG_mmr2_roi_v[s,nROI,ts:te],EEG[s,ts:te])
        r_all_s.append([s,label_names[nROI],rm,rv])
    
df_ROI = pd.DataFrame(columns = ["Subject", "ROI", "Corr MEG_m & EEG", "Corr MEG_v & EEG"], data = r_all_s)
df_ROI.to_pickle('/media/tzcheng/storage/CBS/cbsA_meeg_analysis/correlation/df_corr_MEGEEG_mmr2_ROI.pkl')

## corr between EEG and each vertice: slower, just load the pickled file
r_v_all_s = []

for s in np.arange(0,len(MEG_mmr2_m),1):
    print('Now starting sub' + str(s))
    for v in np.arange(0,np.shape(MEG_mmr2_m)[1],1):
        rm,p = pearsonr(MEG_mmr2_m[s,v,ts:te],EEG[s,ts:te])
        rv,p = pearsonr(MEG_mmr2_v[s,v,ts:te],EEG[s,ts:te])
        r_v_all_s.append([s,v,rm,rv])    
df_v = pd.DataFrame(columns = ["Subject", "Vertno", "Corr MEG_m & EEG", "Corr MEG_v & EEG"], data = r_v_all_s)
df_v.to_pickle('df_corr_MEGEEG_mmr2_v.pkl')

#%%######################################## Done within each subject: cross-correlate (xcorr) time series in a window between EEG & MEG
# Cross-correlation (xcorr) between EEG and MEG
xcorr_m_all_s = []
xcorr_v_all_s = []

for s in np.arange(0,len(MEG_mmr2_m),1):
    a = (EEG[s,ts:te] - np.mean(EEG[s,ts:te]))/np.std(EEG[s,ts:te])
    b = (stc_m[s,:] - np.mean(stc_m[s,:]))/np.std(stc_m[s,:])
    c = (stc_v[s,:] - np.mean(stc_v[s,:]))/np.std(stc_v[s,:])
    
    ## the way matlab do xcorr normalization: the max is 1 if do a zero lag autocorrealtoin
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    c = c / np.linalg.norm(c)

    xcorr = signal.correlate(a,b)
    xcorr_m_all_s.append(xcorr)
    
    xcorr = signal.correlate(a,c)
    xcorr_v_all_s.append(xcorr)

lags = signal.correlation_lags(len(a),len(b))
lags_time = lags/5000

np.argmax(abs(xcorr_m_all_s),axis=1)

print('abs xcorr between MEG_m & EEG:' + str(np.max(np.abs(xcorr_m_all_s),axis=1).mean()))
print('abs xcorr between MEG_v & EEG:' + str(np.max(np.abs(xcorr_v_all_s),axis=1).mean()))
print('max lag of abs xcorr between MEG_m & EEG:' + str(lags_time[np.argmax(np.abs(xcorr_m_all_s),axis=1)].mean()))
print('max lag of abs xcorr between MEG_v & EEG:' + str(lags_time[np.argmax(np.abs(xcorr_v_all_s),axis=1)].mean()))
X = np.max(np.abs(xcorr_v_all_s),axis=1) - np.max(np.abs(xcorr_m_all_s),axis=1)
stats.ttest_1samp(X,0)

## xcorr between EEG and MEG ROI: only store the max xcorr and the lag
xcorr_all_s = []
lag_all_s = []
lags = signal.correlation_lags(len(EEG[0,ts:te]),len(EEG[0,ts:te]))
lags_time = lags/5000

for s in np.arange(0,len(MEG_mmr2_m),1):
    for nROI in np.arange(0,len(label_names),1):
        a = (EEG[s,ts:te] - np.mean(EEG[s,ts:te]))/np.std(EEG[s,ts:te])
        b = (MEG_mmr2_roi_m[s,nROI,ts:te] - np.mean(MEG_mmr2_roi_m[s,nROI,ts:te]))/np.std(MEG_mmr2_roi_m[s,nROI,ts:te])
        c = (MEG_mmr2_roi_v[s,nROI,ts:te] - np.mean(MEG_mmr2_roi_v[s,nROI,ts:te]))/np.std(MEG_mmr2_roi_v[s,nROI,ts:te])
        
        a = a /  np.linalg.norm(a)
        b = b /  np.linalg.norm(b)
        c = c /  np.linalg.norm(c)
        
        xcorr_m = signal.correlate(a,b)
        xcorr_v = signal.correlate(a,c)
        
        xcorr_all_s.append([s,label_names[nROI],max(abs(xcorr_m)),max(abs(xcorr_v))])
        lag_all_s.append([s,label_names[nROI],lags_time[np.argmax(abs(xcorr_m))],lags_time[np.argmax(abs(xcorr_v))]])

df_ROI = pd.DataFrame(columns = ["Subject", "ROI", "XCorr MEG_m & EEG", "XCorr MEG_v & EEG"], data = xcorr_all_s)
df_ROI.to_pickle('/media/tzcheng/storage/CBS/cbsA_meeg_analysis/correlation/df_xcorr_MEGEEG_mmr2_ROI.pkl')
df_ROI_lag = pd.DataFrame(columns = ["Subject", "ROI", "Lag XCorr MEG_m & EEG", "Lag XCorr MEG_v & EEG"], data = lag_all_s)
df_ROI_lag.to_pickle('/media/tzcheng/storage/CBS/cbsA_meeg_analysis/correlation/df_xcorr_lag_MEGEEG_mmr2_ROI.pkl')

# xcorr between EEG and each vertice: only store the max xcorr and the lag
xcorr_v_all_s = []
lag_v_all_s = []

for s in np.arange(0,len(MEG_mmr2_m),1):
    print('Now starting sub' + str(s))
    for v in np.arange(0,np.shape(MEG_mmr2_m)[1],1):
        a = (EEG[s,ts:te] - np.mean(EEG[s,ts:te]))/np.std(EEG[s,ts:te])
        b = (MEG_mmr2_m[s,v,ts:te] - np.mean(MEG_mmr2_m[s,v,ts:te]))/np.std(MEG_mmr2_m[s,v,ts:te])     
        c = (MEG_mmr2_v[s,v,ts:te] - np.mean(MEG_mmr2_v[s,v,ts:te]))/np.std(MEG_mmr2_v[s,v,ts:te])

        ## the way matlab do xcorr normalization: the max is 1 if do a zero lag autocorrealtoin
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        c = c / np.linalg.norm(c)
        
        xcorr_m = signal.correlate(a,b)
        xcorr_v = signal.correlate(a,c)
        xcorr_v_all_s.append([s,v,max(abs(xcorr_m)),max(abs(xcorr_v))])
        lag_v_all_s.append([s,v,lags_time[np.argmax(abs(xcorr_m))],lags_time[np.argmax(abs(xcorr_v))]])
    
df_v = pd.DataFrame(columns = ["Subject", "Vertno", "XCorr MEG_m & EEG", "XCorr MEG_v & EEG"], data = xcorr_v_all_s)
df_v.to_pickle('/media/tzcheng/storage/CBS/cbsA_meeg_analysis/correlation/df_xcorr_MEGEEG_mmr2_v.pkl')
df_lag = pd.DataFrame(columns = ["Subject", "Vertno", "Lag XCorr MEG_m & EEG", "Lag XCorr MEG_v & EEG"], data = lag_v_all_s)
df_v.to_pickle('/media/tzcheng/storage/CBS/cbsA_meeg_analysis/correlation/df_xcorr_lag_MEGEEG_mmr2_v.pkl')

#%%######################################## Analyze pre-saved pickle files
## corr 
corr_ROI =pd.read_pickle('/media/tzcheng/storage/CBS/cbsA_meeg_analysis/correlation/df_corr_MEGEEG_mmr2_ROI.pkl')
corr_v =pd.read_pickle('/media/tzcheng/storage/CBS/cbsA_meeg_analysis/correlation/df_corr_MEGEEG_mmr2_v.pkl')
corr_ROI["Corr MEG_v & EEG"] = pd.DataFrame.abs(corr_ROI["Corr MEG_v & EEG"]) # need to get abs
corr_ROI["Corr MEG_m & EEG"] = pd.DataFrame.abs(corr_ROI["Corr MEG_m & EEG"])
corr_v["Corr MEG_v & EEG"] = pd.DataFrame.abs(corr_v["Corr MEG_v & EEG"])
corr_v["Corr MEG_m & EEG"] = pd.DataFrame.abs(corr_v["Corr MEG_m & EEG"])
corr_ROI_mean = corr_ROI.groupby('ROI').mean()
corr_v_mean = corr_v.groupby('Vertno').mean()

# Visualization
subject = 'fsaverage'
src = mne.read_source_spaces(subjects_dir + 'fsaverage/bem/fsaverage-vol-5-src.fif')
stc1 = mne.read_source_estimate(root_path + 'cbs_A101/sss_fif/cbs_A101_mmr2_morph-vl.stc')

stc_corr = stc1.copy()
v_hack = pd.concat([corr_v_mean["Corr MEG_v & EEG"],corr_v_mean["Corr MEG_m & EEG"]],axis=1) # the first time point is MEG_v & EEG, the second is MEG_m & EEG
stc_corr.data = v_hack
stc_corr.plot(src,subject=subject, subjects_dir=subjects_dir)

## xcorr
xcorr_ROI =pd.read_pickle('/media/tzcheng/storage/CBS/cbsA_meeg_analysis/correlation/df_xcorr_MEGEEG_mmr2_ROI.pkl') # already have abs
xcorr_v =pd.read_pickle('/media/tzcheng/storage/CBS/cbsA_meeg_analysis/correlation/df_xcorr_MEGEEG_mmr2_v.pkl')
xcorr_ROI_mean = xcorr_ROI.groupby('ROI').mean()
xcorr_v_mean = xcorr_v.groupby('Vertno').mean()

# Visualization
subject = 'fsaverage'
src = mne.read_source_spaces(subjects_dir + 'fsaverage/bem/fsaverage-vol-5-src.fif')
stc1 = mne.read_source_estimate(root_path + 'cbs_A101/sss_fif/cbs_A101_mmr2_morph-vl.stc')

stc_corr = stc1.copy()
v_hack = pd.concat([xcorr_v_mean["XCorr MEG_v & EEG"],xcorr_v_mean["XCorr MEG_m & EEG"]],axis=1) # the first time point is MEG_v & EEG, the second is MEG_m & EEG
stc_corr.data = v_hack
stc_corr.plot(src,clim=dict(kind="percent",lims=[90,95,99]),subject=subject, subjects_dir=subjects_dir)

#%%######################################## load EEG & MEG FFR
EEG = mne.read_evokeds(root_path + s + '/eeg/cbs_' + s + '_01_evoked_substd_cabr.fif',allow_maxshield=True)
EEG[0].crop(-0.01,0.18)
MEG = mne.read_evokeds(root_path + s + '/sss_fif/' + s + '_01_evoked_substd_cabr.fif',allow_maxshield=True) # with the mag or vector method
EEG_data = EEG[0].get_data()
MEG_data = MEG[0].get_data()

# small correlation for some subjects this one subject
r_all= []
for i in np.arange(0,len(MEG_data),1):
    r,p = pearsonr(np.squeeze(EEG_data), MEG_data[i,])
    r_all.append(r)
print(max(r_all))
print(min(r_all))

#%%######################################## load EEG FFR
## Load FFR
std = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_std_cabr_eeg_200.npy')
dev1 = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_dev1_cabr_eeg_200.npy')
dev2 = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_dev2_cabr_eeg_200.npy')

## Load recorded  audio
std_audio = np.load(root_path + 'stimuli/cbs_A123_ba.npy')
dev1_audio = np.load(root_path + 'stimuli/cbs_A123_mba.npy')
dev2_audio = np.load(root_path + 'stimuli/cbs_A123_pa.npy')

times = np.linspace(-0.02,0.2,1101)
## Load real audio
# fs, std_audio = wavfile.read('/media/tzcheng/storage/CBS/stimuli/-10.wav')
# fs, dev1_audio = wavfile.read('/media/tzcheng/storage/CBS/stimuli/+40.wav')
# fs, dev2_audio = wavfile.read('/media/tzcheng/storage/CBS/stimuli/-40.wav')
# Downsample


## Initialize 
xcorr_all_s = []
xcorr_lag_all_s = []
times = np.linspace(-0.02,0.2,1101)
audio = std_audio
EEG = std

a = (audio - np.mean(audio))/np.std(audio)
a = a / np.linalg.norm(a)

for s in np.arange(0,len(std),1):
    b = (EEG[s,:] - np.mean(EEG[s,:]))/np.std(EEG[s,:])
    
    ## the way matlab do xcorr normalization: the max is 1 if do a zero lag autocorrealtoin
    b = b / np.linalg.norm(b)

    xcorr = signal.correlate(a,b)
    xcorr = abs(xcorr)
    xcorr_all_s.append(np.max(xcorr))
    xcorr_lag_all_s.append(np.argmax(xcorr))
lags = signal.correlation_lags(len(a),len(b))
lags_time = lags/5000
    
print('abs xcorr between FFR & audio:' + str(np.array(xcorr_all_s).mean()))
print('abs xcorr lag between FFR & audio:' + str(np.array(lags_time[xcorr_lag_all_s]).mean()))

dev1_xcorr
dev1_xcorr_lag
dev2_xcorr
dev2_xcorr_lag
std_xcorr
std_xcorr_lag

X = np.array(dev1_xcorr_lag) - np.array(dev2_xcorr_lag)
stats.ttest_1samp(X,0)
