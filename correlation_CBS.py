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

root_path='/media/tzcheng/storage/CBS/'
subjects_dir = '/media/tzcheng/storage2/subjects/'

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
# mmr
stc1 = mne.read_source_estimate(root_path + 'cbs_A101/sss_fif/cbs_A101_mmr2_morph-vl.stc')
stc2 = mne.read_source_estimate(root_path + 'cbs_A101/sss_fif/cbs_A101_mmr2_morph-vl.stc')
MEG_mmr1_v = np.load(root_path + 'cbsA_meeg_analysis/MEG/vector_method/group_mmr1_mba_vector_morph.npy') # with the mag or vector method
MEG_mmr2_v = np.load(root_path + 'cbsA_meeg_analysis/MEG/vector_method/group_mmr2_pa_vector_morph.npy') # with the mag or vector method
MEG_mmr1_m = np.load(root_path + 'cbsA_meeg_analysis/MEG/magnitude_method/group_mmr1_mba_None_morph.npy') # with the mag or vector method
MEG_mmr2_m = np.load(root_path + 'cbsA_meeg_analysis/MEG/magnitude_method/group_mmr2_pa_None_morph.npy') # with the mag or vector method
MEG_mmr1_roi_v = np.load(root_path + 'cbsA_meeg_analysis/MEG/vector_method/group_mmr1_mba_vector_morph_roi.npy') # with the mag or vector method
MEG_mmr2_roi_v = np.load(root_path + 'cbsA_meeg_analysis/MEG/vector_method/group_mmr2_pa_vector_morph_roi.npy') # with the mag or vector method
MEG_mmr1_roi_m = np.load(root_path + 'cbsA_meeg_analysis/MEG/magnitude_method/group_mmr1_mba_morph_roi.npy') # with the mag or vector method
MEG_mmr2_roi_m = np.load(root_path + 'cbsA_meeg_analysis/MEG/magnitude_method/group_mmr2_pa_morph_roi.npy') # with the mag or vector method

EEG_mmr1 = np.load(root_path + 'cbsA_meeg_analysis/EEG/group_mmr1_mba_eeg.npy')
EEG_mmr2 = np.load(root_path + 'cbsA_meeg_analysis/EEG/group_mmr2_pa_eeg.npy')

times = stc1.times

# load source
subject = 'fsaverage'
src = mne.read_source_spaces(subjects_dir + 'fsaverage/bem/fsaverage-vol-5-src.fif')

# load ROI label 
fname_aseg = subjects_dir + subject + '/mri/aparc+aseg.mgz'
label_names = np.asarray(mne.get_volume_labels_from_aseg(fname_aseg))

lh_ROI_label = [60,61,62,72] # STG and IFG (parsopercularis, parsorbitalis, parstriangularis)
rh_ROI_label = [96,97,98,108] # STG and IFG (parsopercularis, parsorbitalis, parstriangularis)

#%%######################################## visualization
# brain heat map for the averaged result
stc1.data = MEG_mmr1_v.mean(axis=0)
stc2.data = MEG_mmr2_m.mean(axis=0)

# stc.plot(src,clim=dict(kind="value",pos_lims=[0,2,5]),subject=subject, subjects_dir=subjects_dir)
stc1.plot(src,subject=subject, subjects_dir=subjects_dir,clim=dict(kind="value",pos_lims=[0,2,8]))
stc2.plot(src,subject=subject, subjects_dir=subjects_dir,clim=dict(kind="value",pos_lims=[0,4,10]))
stc2.plot(src,subject=subject, subjects_dir=subjects_dir)

# time series of EEG, MEG_v, MEG_m averaged across all sources
plt.figure()
plot_err(stats.zscore(EEG_mmr2,axis=1),'k',stc1.times)
plot_err(stats.zscore(MEG_mmr2_m.mean(axis=1),axis=1),'b',stc1.times)
plot_err(stats.zscore(MEG_mmr2_v.mean(axis=1),axis=1),'r',stc1.times)
plt.legend(['EEG','','MEG mag','','MEG vector',''])
plt.xlabel('Time (ms)')
plt.ylabel('zscore')
plt.xlim([-100, 600])
plt.ylim([-2, 1.5])

# time series of EEG, MEG_v, MEG_m averaged across all sources for each individual
fig,axs = plt.subplots(1,len(MEG_mmr1_m),sharex=True,sharey=True)
fig.suptitle('MMR2')

for s in np.arange(0,len(MEG_mmr1_m),1):
    axs[s].plot(times,stats.zscore(EEG_mmr2[s,:],axis=0),'k')
    axs[s].plot(times,stats.zscore(MEG_mmr2_m[s,:,:].mean(axis =0),axis=0),'b')
    axs[s].plot(times,stats.zscore(MEG_mmr2_v[s,:,:].mean(axis =0),axis=0),'r')
    axs[s].set_xlim([0.05,0.3])
    axs[s].set_title('subj' + str(s+1))
    
#%%######################################## averaged across subjects
ts = 1000 # 100 ms
te = 2000 # 300 ms

## averaged across all sources
mean_EEG_mmr2 = EEG_mmr2.mean(axis =0)
stc_m = MEG_mmr2_m.mean(axis =0).mean(axis=0)

mean_MEG_mmr2_v = MEG_mmr2_v.mean(axis =0).mean(axis=0)
mean_MEG_mmr2_m = MEG_mmr2_m.mean(axis =0).mean(axis=0)

# pearson r 
# Tradtional direction: TOI [100 250] 0.89 for MEG_m & EEG and -0.63 for MEG_v & EEG 
# New direction: TOI [100 300] -0.40 for MEG_m & EEG and -0.26 for MEG_v & EEG 
corr_p = pearsonr(mean_EEG_mmr2[ts:te], mean_MEG_mmr2_m[ts:te])

# normalized cross correlation [-1 1]
# based on https://stackoverflow.com/questions/53436231/normalized-cross-correlation-in-python
a = mean_EEG_mmr2
b = stc_m

norm_a = np.linalg.norm(a)
a = a / norm_a
norm_b = np.linalg.norm(b)
b = b / norm_b

xcorr = signal.correlate(a,b)
print(max(xcorr))
print(min(xcorr))
print(np.argmax(xcorr))
print(np.argmin(xcorr))
lags = signal.correlation_lags(len(mean_EEG_mmr2),len(mean_MEG_mmr2_m))
lags_time = lags/5000

#xcorr = np.abs(xcorr)
max_xcorr = np.max(xcorr)
max_xcorr_ind = np.argmax(xcorr)
times[np.argmax(xcorr)] # the time when the max corr happened

# cosine similarity
cos_sim = dot(mean_EEG_mmr2[ts:te],mean_MEG_mmr2_m[ts:te]) / (norm(mean_EEG_mmr2[ts:te])*norm(mean_MEG_mmr2_m[ts:te]))
print(cos_sim)
cos_sim = dot(mean_EEG_mmr2[ts:te],stc_v[ts:te]) / (norm(mean_EEG_mmr2[ts:te])*norm(stc_v[ts:te]))
print(cos_sim)

## for each ROI

## for each vertice

#%%######################################## for each subject: get mean or max in a window

## corr between EEG and averaged source MEG
#%% [poor method] get the mean: no correlation observed between EEG and MEG_m (better in window [100 200] ms)
stc_m = MEG_mmr2_m[:,:,ts:te].mean(axis=-1).mean(axis=-1)
stc_v = MEG_mmr2_v[:,:,ts:te].mean(axis=-1).mean(axis=-1)
mean_EEG_mmr2 = EEG_mmr2[:,ts:te].mean(axis=-1)

r,p = pearsonr(stc_m,mean_EEG_mmr2)
print(r,p)
r,p = pearsonr(stc_v,mean_EEG_mmr2)
print(r,p)
r,p = pearsonr(stc_v,stc_m)
print(r,p)

#%% [poor method] get the max (peak) or min (trough) point: no correlation observed between EEG and MEG_m (better in window [100 200] ms)
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

#%% [ok method] get the correlation between each point
stc_m = MEG_mmr1_m.mean(axis=1) # change the name to std or other data
stc_v = MEG_mmr1_v.mean(axis=1)
EEG = EEG_mmr1


r_m_all_t = []
r_v_all_t = []

for t in np.arange(0,len(times),1):
    r,p = pearsonr(stc_m[:,t],EEG[:,t])
    r_m_all_t.append(r)
    r,p = pearsonr(stc_v[:,t],EEG[:,t])
    r_v_all_t.append(r)

fig, ax = plt.subplots(1)
ax.plot(times, r_m_all_t)
ax.plot(times, r_v_all_t)
ax.axhline(0, color="k", linestyle="--")
ax.axvline(0, color="k")
plt.title('MMR1')
plt.legend(['MEG_m','MEG_v'])
plt.xlabel('Time (s)')
plt.ylabel('Pearson r')
plt.xlim([-0.1,0.6])

plt.figure()
plt.scatter(stc_v[:,1300],EEG[:,1300])
plt.xlabel('MEG')
plt.ylabel('EEG')
plt.title('t = 150ms')

#%%######################################## for each subject: correlate time series in a window with pearson r
stc_m = MEG_mmr2_m[:,:,ts:te].mean(axis=1)
stc_v = MEG_mmr2_v[:,:,ts:te].mean(axis=1)
EEG = EEG_mmr2

# [good method] get the pearson correlation of the whole time window
r_m_all_s = []
r_v_all_s = []

for s in np.arange(0,len(MEG_mmr1_m),1):
    r,p = pearsonr(stc_m[s,:],EEG[s,ts:te])
    r_m_all_s.append(r)
    r,p = pearsonr(stc_v[s,:],EEG[s,ts:te])
    r_v_all_s.append(r)

print('abs corr between MEG_m & EEG:' + str(np.abs(r_m_all_s).mean()))
print('abs corr between MEG_v & EEG:' + str(np.abs(r_v_all_s).mean()))

## corr between EEG and MEG ROI
r_all_s = []

for s in np.arange(0,len(MEG_mmr1_m),1):
    for nROI in np.arange(0,len(label_names),1):
        rm,p = pearsonr(MEG_mmr1_roi_m[s,nROI,ts:te],EEG[s,ts:te])
        rv,p = pearsonr(MEG_mmr1_roi_v[s,nROI,ts:te],EEG[s,ts:te])
        r_all_s.append([s,label_names[nROI],rm,rv])
    
df_ROI = pd.DataFrame(columns = ["Subject", "ROI", "Corr MEG_m & EEG", "Corr MEG_v & EEG"], data = r_all_s)
df_ROI.groupby('ROI').mean()

## corr between EEG and each vertice: slower, just load the pickled file
r_all_s = []

for s in np.arange(0,len(MEG_mmr1_m),1):
    print('Now starting sub' + str(s))
    for v in np.arange(0,np.shape(MEG_mmr1_m)[1],1):
        rm,p = pearsonr(MEG_mmr1_m[s,v,ts:te],EEG[s,ts:te])
        rv,p = pearsonr(MEG_mmr1_v[s,v,ts:te],EEG[s,ts:te])
        r_all_s.append([s,v,rm,rv])    
df_v = pd.DataFrame(columns = ["Subject", "Vertno", "Corr MEG_m & EEG", "Corr MEG_v & EEG"], data = r_all_s)
df_v.to_pickle('df_corr_MEGEEG_mmr1_v.pkl')
df_v_mean = df_v.groupby('Vertno').mean()
v_hack = df_v_mean["Corr MEG_v & EEG"]
v_hack = pd.concat([v_hack,v_hack],axis=1)
stc_corr = stc1.copy()
stc_corr.data = v_hack
stc_corr.plot(src,subject=subject, subjects_dir=subjects_dir,clim=dict(kind="value",pos_lims=[0,0.16,0.32]))

#%%######################################## for each subject: correlate time series in a window with xcorr
# get the cross correlation of the whole time windowr
xcorr_m_all_s = []
xcorr_v_all_s = []

for s in np.arange(0,len(MEG_mmr1_m),1):
    a = (EEG[s,ts:te] - np.mean(EEG[s,ts:te]))/np.std(EEG[s,ts:te])
    b = (stc_m[s,:] - np.mean(stc_m[s,:]))/np.std(stc_m[s,:])
    c = (stc_v[s,:] - np.mean(stc_v[s,:]))/np.std(stc_v[s,:])
    
    ## the way matlab do xcorr normalization: the max is 1 if do a zero lag autocorrealtoin
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    c = c / np.linalg.norm(c)

    xcorr = signal.correlate(a,b)
    xcorr = abs(xcorr)
    xcorr_m_all_s.append(xcorr)
    
    xcorr = signal.correlate(a,c)
    xcorr = abs(xcorr)
    xcorr_v_all_s.append(xcorr)

print('abs xcorr between MEG_m & EEG:' + str(np.array(xcorr_m_all_s).mean()))
print('abs xcorr between MEG_v & EEG:' + str(np.array(xcorr_v_all_s).mean()))

## xcorr between EEG and MEG ROI
xcorr_all_s = []

for s in np.arange(0,len(MEG_mmr1_m),1):
    for nROI in np.arange(0,len(label_names),1):
        a = (EEG[s,ts:te] - np.mean(EEG[s,ts:te]))/np.std(EEG[s,ts:te])
        b = (MEG_mmr2_roi_m[s,nROI,ts:te] - np.mean(MEG_mmr2_roi_m[s,nROI,ts:te]))/np.std(MEG_mmr2_roi_m[s,nROI,ts:te])
        c = (MEG_mmr2_roi_v[s,nROI,ts:te] - np.mean(MEG_mmr2_roi_v[s,nROI,ts:te]))/np.std(MEG_mmr2_roi_v[s,nROI,ts:te])
        
        a = a /  np.linalg.norm(a)
        b = b /  np.linalg.norm(b)
        c = c /  np.linalg.norm(c)
        
        xcorr_m = signal.correlate(a,b)
        xcorr_m = abs(xcorr_m)
        xcorr_v = signal.correlate(a,b)
        xcorr_v = abs(xcorr_v)
        
        xcorr_all_s.append([s,label_names[nROI],xcorr_m,xcorr_v])

df_ROI = pd.DataFrame(columns = ["Subject", "ROI", "Corr MEG_m & EEG", "Corr MEG_v & EEG"], data = r_all_s)
df_ROI.groupby('ROI').mean()

#%%######################################## for each subject: correlate time series in a window with cosine similarity (still working)
cos_sim_all_s = [] 

for s in np.arange(0,len(MEG_mmr1_m),1):
    a = EEG[s,ts:te]
    b = stc_v[s,:]
     
    cos_sim = dot(a,b) / (norm(a)*norm(b))
    cos_sim_all_s.append(cos_sim)

print('abs cos sim between MEG_m & EEG:' + str(np.array(xcorr_m_all_s).mean()))
print('abs cos sim between MEG_v & EEG:' + str(np.array(xcorr_v_all_s).mean()))

#%%######################################## load data Standard
# Hypothesis: N1 is more sensory, so should observe the highest correlation in the auditory area
stc1 = mne.read_source_estimate(root_path + 'cbs_A101/sss_fif/cbs_A101_std_vector_morph-vl.stc')

MEG_std_v = np.load(root_path + 'cbsA_meeg_analysis/group_std_vector_morph.npy') # with the mag or vector method
MEG_std_roi_v = np.load(root_path + 'cbsA_meeg_analysis/group_std_vector_morph_roi.npy') # with the mag or vector method

EEG_std = np.load(root_path + 'cbsA_meeg_analysis/group_std_eeg.npy',allow_pickle=True)

# brain heat map for the averaged result
stc1.data = MEG_std_v.mean(axis=0)
#%%######################################## visualization
# stc.plot(src,clim=dict(kind="value",pos_lims=[0,2,5]),subject=subject, subjects_dir=subjects_dir)
stc1.plot(src,subject=subject, subjects_dir=subjects_dir,clim=dict(kind="value",pos_lims=[0,2,8]))
stc1.plot(src,subject=subject, subjects_dir=subjects_dir)

# time series of EEG, MEG_v, MEG_m averaged across all sources
plt.figure()
plot_err(stats.zscore(EEG_std,axis=1),'k',stc1.times)
plot_err(stats.zscore(MEG_std_v.mean(axis=1),axis=1),'b',stc1.times)
plt.legend(['EEG','','MEG mag',''])
plt.xlabel('Time (ms)')
plt.ylabel('zscore')
plt.xlim([-100, 600])

# time series of EEG, MEG_v, MEG_m averaged across all sources for each individual
fig,axs = plt.subplots(1,len(MEG_std_v),sharex=True,sharey=True)
fig.suptitle('Standard')

for s in np.arange(0,len(MEG_std_v),1):
    axs[s].plot(times,stats.zscore(EEG_std[s,:],axis=0),'k')
    axs[s].plot(times,stats.zscore(MEG_std_v[s,:,:].mean(axis =0),axis=0),'b')
    axs[s].set_xlim([0.05,0.15])
    axs[s].set_title('subj' + str(s+1))

#%%######################################## correlation
ts = 750 
te = 1500

# point-by-point
stc_v = MEG_std_v.mean(axis=1)

r_v_all_t = []

for t in np.arange(0,len(times),1):
    r,p = pearsonr(stc_v[:,t],EEG_std[:,t])
    r_v_all_t.append(r)
    
plt.figure()
plt.plot(times,r_v_all_t)
plt.xlim([-0.1,0.6])
plt.xlabel('Time (s)')
plt.ylabel('Pearson r')
plt.title('Standard N1 component')

# pearson across time series
stc_v = MEG_std_v[:,:,ts:te].mean(axis=1)

# [good method] get the pearson correlation of the whole time window
r_v_all_s = []

for s in np.arange(0,len(stc_v),1):
    r,p = pearsonr(stc_v[s,:],EEG_std[s,ts:te])
    r_v_all_s.append(r)

print('abs corr between MEG_v & EEG:' + str(np.abs(r_v_all_s).mean()))

## corr between EEG and MEG ROI
# r_roi_s = []

# for s in np.arange(0,len(MEG_std_roi_v),1):
#     for nROI in np.arange(0,len(label_names),1):
#         rv,p = pearsonr(MEG_std_roi_v[s,nROI,ts:te],EEG_std[s,ts:te])
#         r_roi_s.append([s,label_names[nROI],rv])
    
# df_ROI = pd.DataFrame(columns = ["Subject", "ROI", "Corr MEG_v & EEG"], data = r_roi_s)
# df_ROI.groupby('ROI').mean()
# df_ROI.to_pickle('df_corr_MEGEEG_std_ROI.pkl')
df_ROI_corr = pd.read_pickle('df_corr_MEGEEG_std_ROI.pkl')
df_ROI_corr["Corr MEG_v & EEG"] =df_ROI_corr["Corr MEG_v & EEG"].abs()
df_ROI_corr = df_ROI_corr.groupby('ROI').mean()

## corr between EEG and each vertice: slower, just load the pickled file
# r_all_s = []
# for s in np.arange(0,len(stc_v),1):
#     print('Now starting sub' + str(s))
#     for v in np.arange(0,np.shape(MEG_std_v)[1],1):
#         rv,p = pearsonr(MEG_std_v[s,v,ts:te],EEG_std[s,ts:te])
#         r_all_s.append([s,v,rv])    
# df_v = pd.DataFrame(columns = ["Subject", "Vertno", "Corr MEG_v & EEG"], data = r_all_s)
# df_v.to_pickle('df_corr_MEGEEG_std.pkl')
df_v_corr = pd.read_pickle('df_corr_MEGEEG_std.pkl')
df_v_corr["Corr MEG_v & EEG"] =df_v_corr["Corr MEG_v & EEG"].abs()
df_v_mean = df_v_corr.groupby('Vertno').mean()
v_hack = df_v_mean["Corr MEG_v & EEG"]
v_hack = pd.concat([v_hack,v_hack],axis=1)
stc_corr = stc1.copy()
stc_corr.data = v_hack
stc_corr.plot(src,subject=subject, subjects_dir=subjects_dir)
stc_corr.plot(src,subject=subject, subjects_dir=subjects_dir,clim=dict(kind="value",pos_lims=[0,0.2,0.39]))

# xcorr get the cross correlation of the whole time windowr
xcorr_v_all_s = []
xcorr_lag_v_all_s = []

for s in np.arange(0,len(stc_v),1):
    a = (EEG_std[s,ts:te] - np.mean(EEG_std[s,ts:te]))/np.std(EEG_std[s,ts:te])
    b = (stc_v[s,:] - np.mean(stc_v[s,:]))/np.std(stc_v[s,:])
    
    ## the way matlab do xcorr normalization: the max is 1 if do a zero lag autocorrealtoin
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    xcorr = signal.correlate(a,b)
    xcorr = abs(xcorr)
    xcorr_v_all_s.append(np.max(xcorr))
    xcorr_lag_v_all_s.append(np.argmax(xcorr))
    
print('abs xcorr between MEG_v & EEG:' + str(np.array(xcorr_v_all_s).mean()))

## xcorr between EEG and MEG ROI
# xcorr_v_roi_s = []

# for s in np.arange(0,len(MEG_std_roi_v),1):
#     for nROI in np.arange(0,len(label_names),1):
#         a = (EEG_std[s,ts:te] - np.mean(EEG_std[s,ts:te]))/np.std(EEG_std[s,ts:te])
#         b = (MEG_std_roi_v[s,nROI,ts:te] - np.mean(MEG_std_roi_v[s,nROI,ts:te]))/np.std(MEG_std_roi_v[s,nROI,ts:te])
        
#         ## the way matlab do xcorr normalization: the max is 1 if do a zero lag autocorrealtoin
#         a = a / np.linalg.norm(a)
#         b = b / np.linalg.norm(b)
        
#         xcorr = signal.correlate(a,b)
#         xcorr = abs(xcorr)
#         xcorr_v_roi_s.append([s,label_names[nROI],np.max(xcorr)])
    
# df_ROI = pd.DataFrame(columns = ["Subject", "ROI", "XCorr MEG_v & EEG"], data = xcorr_v_roi_s)
# df_ROI.groupby('ROI').mean()
# df_ROI.to_pickle('df_xcorr_MEGEEG_std_ROI.pkl')
df_ROI_xcorr = pd.read_pickle('df_xcorr_MEGEEG_std_ROI.pkl')
df_ROI_xcorr = df_ROI_xcorr.groupby('ROI').mean()
# xcorr between EEG and each vertice
# xcorr_v_all_s = []

# for s in np.arange(0,len(MEG_std_v),1):
#     print('Now starting sub' + str(s))
#     for v in np.arange(0,np.shape(MEG_std_v)[1],1):
#         a = (EEG_std[s,ts:te] - np.mean(EEG_std[s,ts:te]))/np.std(EEG_std[s,ts:te])
#         b = (MEG_std_v[s,v,ts:te] - np.mean(MEG_std_v[s,v,ts:te]))/np.std(MEG_std_v[s,v,ts:te])
        
#         ## the way matlab do xcorr normalization: the max is 1 if do a zero lag autocorrealtoin
#         a = a / np.linalg.norm(a)
#         b = b / np.linalg.norm(b)
        
#         xcorr = signal.correlate(a,b)
#         xcorr = abs(xcorr)
#         xcorr_v_all_s.append([s,v,np.max(xcorr)])
    
# df_v = pd.DataFrame(columns = ["Subject", "Vertno", "XCorr MEG_v & EEG"], data = xcorr_v_all_s)
# df_v.to_pickle('df_xcorr_MEGEEG_std.pkl')
# del df_v
df_v_xcorr = pd.read_pickle('df_xcorr_MEGEEG_std.pkl')
df_v_mean = df_v_xcorr.groupby('Vertno').mean()
v_hack = df_v_mean["XCorr MEG_v & EEG"]
v_hack = pd.concat([v_hack,v_hack],axis=1)
stc_corr = stc1.copy()
stc_corr.data = v_hack
stc_corr.plot(src,subject=subject, subjects_dir=subjects_dir)
stc_corr.plot(src,subject=subject, subjects_dir=subjects_dir,clim=dict(kind="value",pos_lims=[0,0.2,0.39]))

#%%######################################## load data FFR
s = 'cbs_A109'
raw_file = mne.io.read_raw_fif(root_path + s +'/raw_fif/' + s + '_01_raw.fif',allow_maxshield=True)
cabr_event = mne.read_events(root_path + s + '/events/' + s + '_01_events_cabr-eve.fif')

audio = raw_file.pick_channels(['MISC001'])
event_id = {'Standardp':1,'Standardn':2, 'Deviant1p':3,'Deviant1n':5, 'Deviant2p':6,'Deviant2n':7}
epochs = mne.Epochs(audio, cabr_event, event_id,tmin =-0.01, tmax=0.18,baseline=(-0.01,0))
evoked_substd=epochs['Standardp'].average(picks=('misc'))
evoked_dev1=epochs['Deviant1p'].average(picks=('misc'))
evoked_dev2=epochs['Deviant2p'].average(picks=('misc'))
evoked_substd.plot()
evoked_dev1.plot()
evoked_dev2.plot()

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