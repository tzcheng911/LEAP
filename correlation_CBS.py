#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 13:23:58 2023
calculate EEG & MEG correlation in adults for 
1. TOI 150 - 250 ms across whole brain
2. ROI
3. each vertice

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

np.printoptions(precision=3)
sp.printoptions(precision=3)

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
    
#%%######################################## load data
stc1 = mne.read_source_estimate(root_path + 'cbs_A101/sss_fif/cbs_A101_mmr2_morph-vl.stc')
stc2 = mne.read_source_estimate(root_path + 'cbs_A101/sss_fif/cbs_A101_mmr2_morph-vl.stc')
MEG_mmr1_v = np.load(root_path + 'cbsA_meeg_analysis/group_mmr1_vector_morph.npy') # with the mag or vector method
MEG_mmr2_v = np.load(root_path + 'cbsA_meeg_analysis/group_mmr2_vector_morph.npy') # with the mag or vector method
MEG_mmr1_m = np.load(root_path + 'cbsA_meeg_analysis/group_mmr1_morph.npy') # with the mag or vector method
MEG_mmr2_m = np.load(root_path + 'cbsA_meeg_analysis/group_mmr2_morph.npy') # with the mag or vector method
MEG_mmr1_roi_v = np.load(root_path + 'cbsA_meeg_analysis/group_mmr1_vector_morph_roi.npy') # with the mag or vector method
MEG_mmr2_roi_v = np.load(root_path + 'cbsA_meeg_analysis/group_mmr2_vector_morph_roi.npy') # with the mag or vector method
MEG_mmr1_roi_m = np.load(root_path + 'cbsA_meeg_analysis/group_mmr1_morph_roi.npy') # with the mag or vector method
MEG_mmr2_roi_m = np.load(root_path + 'cbsA_meeg_analysis/group_mmr2_morph_roi.npy') # with the mag or vector method

EEG_mmr1 = np.load(root_path + 'cbsA_meeg_analysis/group_mmr1_eeg.npy')
EEG_mmr2 = np.load(root_path + 'cbsA_meeg_analysis/group_mmr2_eeg.npy')
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
stc1.data = MEG_mmr1_m.mean(axis=0)
stc2.data = MEG_mmr2_m.mean(axis=0)

# stc.plot(src,clim=dict(kind="value",pos_lims=[0,2,5]),subject=subject, subjects_dir=subjects_dir)
stc1.plot(src,subject=subject, subjects_dir=subjects_dir)
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
te = 1750 # 250 ms

## averaged across all sources
mean_EEG_mmr2 = EEG_mmr1.mean(axis =0)
mean_MEG_mmr2_m = MEG_mmr1_m.mean(axis =0).mean(axis=0)
mean_MEG_mmr2_v = MEG_mmr1_v.mean(axis =0).mean(axis=0)

# pearson r 
# TOI [100 250] 0.89 for MEG_m & EEG and -0.63 for MEG_v & EEG 
corr_p = pearsonr(mean_EEG_mmr2[ts:te], mean_MEG_mmr2_v[ts:te])

# normalized cross correlation [-1 1]
# based on https://stackoverflow.com/questions/53436231/normalized-cross-correlation-in-python
a = mean_EEG_mmr2
b = mean_MEG_mmr2_m

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
cos_sim = dot(mean_EEG_mmr2[ts:te],mean_MEG_mmr2_v[ts:te]) / (norm(mean_EEG_mmr2[ts:te])*norm(mean_MEG_mmr2_v[ts:te]))
print(cos_sim)

## for each ROI

## for each vertice

#%%######################################## for each subject: get mean or max in a window
ts = 1000
te = 1500

## corr between EEG and averaged source MEG
#%% [poor method] get the mean: no correlation observed between EEG and MEG_m (better in window [100 200] ms)
mean_MEG_mmr2_m = MEG_mmr2_m[:,:,ts:te].mean(axis=-1).mean(axis=-1)
mean_MEG_mmr2_v = MEG_mmr2_v[:,:,ts:te].mean(axis=-1).mean(axis=-1)
mean_EEG_mmr2 = EEG_mmr2[:,ts:te].mean(axis=-1)

r,p = pearsonr(mean_MEG_mmr2_m,mean_EEG_mmr2)
print(r,p)
r,p = pearsonr(mean_MEG_mmr2_v,mean_EEG_mmr2)
print(r,p)
r,p = pearsonr(mean_MEG_mmr2_v,mean_MEG_mmr2_m)
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

#%% [ok method] get the correlation between each point: no correlation observed between EEG and MEG_m (better in window [100 200] ms)
mean_MEG_mmr2_m = MEG_mmr2_m.mean(axis=1)
mean_MEG_mmr2_v = MEG_mmr2_v.mean(axis=1)

r_m_all_t = []
r_v_all_t = []

for t in np.arange(0,len(times),1):
    r,p = pearsonr(mean_MEG_mmr2_m[:,t],EEG_mmr2[:,t])
    r_m_all_t.append(r)
    r,p = pearsonr(mean_MEG_mmr2_v[:,t],EEG_mmr2[:,t])
    r_v_all_t.append(r)
    
plt.figure()
plt.plot(times,r_m_all_t)
plt.plot(times,r_v_all_t)
plt.xlim([-0.1,0.6])
plt.xlabel('Time (s)')
plt.legend(['MEG_m','MEG_v'])
plt.ylabel('Pearson r')

#%%######################################## for each subject: correlate time series in a window with pearson r
ts = 1000 # 100 ms
te = 2000 # 300 ms

mean_MEG_mmr1_m = MEG_mmr1_m[:,:,ts:te].mean(axis=1)
mean_MEG_mmr1_v = MEG_mmr1_v[:,:,ts:te].mean(axis=1)
mean_MEG_mmr2_m = MEG_mmr2_m[:,:,ts:te].mean(axis=1)
mean_MEG_mmr2_v = MEG_mmr2_v[:,:,ts:te].mean(axis=1)

# [good method] get the pearson correlation of the whole time window
r_m_all_s = []
r_v_all_s = []

for s in np.arange(0,len(MEG_mmr1_m),1):
    r,p = pearsonr(mean_MEG_mmr2_m[s,:],EEG_mmr2[s,ts:te])
    r_m_all_s.append(r)
    r,p = pearsonr(mean_MEG_mmr2_v[s,:],EEG_mmr2[s,ts:te])
    r_v_all_s.append(r)

print('abs corr between MEG_m & EEG:' + str(np.abs(r_m_all_s).mean()))
print('abs corr between MEG_v & EEG:' + str(np.abs(r_v_all_s).mean()))

## corr between EEG and MEG ROI
mean_MEG_mmr2_m = MEG_mmr2_roi_m[:,lh_ROI_label,ts:te].mean(axis=1)
mean_MEG_mmr2_v = MEG_mmr2_roi_v[:,lh_ROI_label,ts:te].mean(axis=1)
r_all_s = []
p_all_s = []
for s in np.arange(0,len(MEG_mmr1_m),1):
    r,p = pearsonr(mean_MEG_mmr2_v[s,:],EEG_mmr2[s,ts:te])
    r_all_s.append(r)
    p_all_s.append(p)
    
## corr between EEG and each vertice

#%%######################################## for each subject: correlate time series in a window with xcorr
# get the cross correlation of the whole time windowr
xcorr_m_all_s = []
xcorr_v_all_s = []

for s in np.arange(0,len(MEG_mmr1_m),1):
    a = EEG_mmr2[s,ts:te]
    b = mean_MEG_mmr2_m[s,:]
    c = mean_MEG_mmr2_v[s,:]
    
    ## the way matlab do xcorr normalization: the max is 1 if do a zero lag autocorrealtoin
    norm_a = np.linalg.norm(a)
    a = a / norm_a
    norm_b = np.linalg.norm(b)
    b = b / norm_b
    norm_c = np.linalg.norm(c)
    c = c / norm_c

    xcorr = signal.correlate(a,b)
    xcorr = abs(xcorr)
    xcorr_m_all_s.append(xcorr)
    
    xcorr = signal.correlate(a,c)
    xcorr = abs(xcorr)
    xcorr_v_all_s.append(xcorr)

print('abs xcorr between MEG_m & EEG:' + str(np.array(xcorr_m_all_s).mean()))
print('abs xcorr between MEG_v & EEG:' + str(np.array(xcorr_v_all_s).mean()))

#%%######################################## for each subject: correlate time series in a window with cosine similarity (still working)
cos_sim_all_s = [] 

for s in np.arange(0,len(MEG_mmr1_m),1):
    a = EEG_mmr2[s,ts:te]
    b = mean_MEG_mmr2_v[s,:]
     
    cos_sim = dot(a,b) / (norm(a)*norm(b))
    cos_sim_all_s.append(cos_sim)

print('abs cos sim between MEG_m & EEG:' + str(np.array(xcorr_m_all_s).mean()))
print('abs cos sim between MEG_v & EEG:' + str(np.array(xcorr_v_all_s).mean()))