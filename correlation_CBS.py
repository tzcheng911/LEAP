#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 13:23:58 2023
1. calculate EEG & MEG correlation in adults
2. calculate correlation between adult and baby MEG
@author: tzcheng
"""

import mne
import numpy as np
from numpy import dot
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy import stats,signal
import os

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
    
#%%######################################## load MEG
stc1 = mne.read_source_estimate(root_path + 'cbs_A101/sss_fif/cbs_A101_mmr2_morph-vl.stc')
stc2 = mne.read_source_estimate(root_path + 'cbs_A101/sss_fif/cbs_A101_mmr2_morph-vl.stc')
MEG_mmr1_v = np.load(root_path + 'meeg_mmr_analysis/group_mmr1_vector_morph.npy') # with the mag or vector method
MEG_mmr2_v = np.load(root_path + 'meeg_mmr_analysis/group_mmr2_vector_morph.npy') # with the mag or vector method
MEG_mmr1_m = np.load(root_path + 'meeg_mmr_analysis/group_mmr1_morph.npy') # with the mag or vector method
MEG_mmr2_m = np.load(root_path + 'meeg_mmr_analysis/group_mmr2_morph.npy') # with the mag or vector method

stc1.data = MEG_mmr1_m.mean(axis=0)
stc2.data = MEG_mmr2_m.mean(axis=0)

subject = 'fsaverage'
src = mne.read_source_spaces(subjects_dir + 'fsaverage/bem/fsaverage-vol-5-src.fif')

# stc.plot(src,clim=dict(kind="value",pos_lims=[0,2,5]),subject=subject, subjects_dir=subjects_dir)
stc1.plot(src,subject=subject, subjects_dir=subjects_dir)
stc2.plot(src,subject=subject, subjects_dir=subjects_dir)

#%%######################################## load EEG
EEG_mmr1 = np.load(root_path + 'meeg_mmr_analysis/group_mmr1_eeg.npy')
EEG_mmr2 = np.load(root_path + 'meeg_mmr_analysis/group_mmr2_eeg.npy')

plot_err(stats.zscore(EEG_mmr2,axis=1),'k',stc1.times)
plot_err(stats.zscore(MEG_mmr2_m.mean(axis=1),axis=1),'r',stc1.times)
plot_err(stats.zscore(MEG_mmr2_v.mean(axis=1),axis=1),'r',stc1.times)

plt.legend(['EEG','','MEG mag','','MEG vector',''])
plt.xlabel('Time (ms)')
plt.ylabel('zscore')

#%%######################################## averaged across subjects
ts = 501 # 0ms
te = 2000# 200 ms
mean_EEG_mmr2 = EEG_mmr2.mean(axis =0)
mean_MEG_mmr2_m = MEG_mmr2_m.mean(axis =0).mean(axis=0)
mean_MEG_mmr2_v = MEG_mmr2_v.mean(axis =0).mean(axis=0)

# pearson r
corr_p = pearsonr(mean_EEG_mmr2[ts:te], mean_MEG_mmr2_v[ts:te]) # 0 ms to 200 ms

# cross correlation
xcorr = signal.correlate(mean_EEG_mmr2,mean_MEG_mmr2_m)
lags = signal.correlation_lags(len(mean_EEG_mmr2),len(mean_MEG_mmr2_m))
xcorr = np.abs(xcorr)

fig,(ax1,ax2,ax3) = plt.subplots(3,1)
ax1.plot(stc1.times,mean_EEG_mmr2)
ax2.plot(stc1.times,mean_MEG_mmr2_m)
ax3.plot(lags,xcorr)

max_xcorr = np.max(xcorr)
max_xcorr_ind = np.argmax(xcorr)
stc1.times[max_xcorr_ind - len(stc1.times)] # the time when the max corr happened

# cosine similarity
cos_sim = dot(mean_EEG_mmr2[ts:te],mean_MEG_mmr2_m[ts:te]) / (norm(mean_EEG_mmr2[ts:te])*norm(mean_MEG_mmr2_m[ts:te]))
print(cos_sim)

cos_sim = dot(mean_EEG_mmr2[ts:te],mean_MEG_mmr2_v[ts:te]) / (norm(mean_EEG_mmr2[ts:te])*norm(mean_MEG_mmr2_v[ts:te]))
print(cos_sim)

#%%######################################## for each subject
allv_MEG_mmr2 = MEG_mmr2_m.mean(axis = 1)
r_all = []
for nsub in np.arange(0,len(allv_MEG_mmr2),1):
    r,p = pearsonr(EEG_mmr2[nsub,],allv_MEG_mmr2[nsub,])
    r_all.append(r)
    
#%%######################################## for each vertice or the hot spot [20000-30000]
