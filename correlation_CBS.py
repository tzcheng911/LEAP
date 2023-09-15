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
    
#%%######################################## load MEG
stc1 = mne.read_source_estimate(root_path + 'cbs_A101/sss_fif/cbs_A101_mmr2_morph-vl.stc')
stc2 = mne.read_source_estimate(root_path + 'cbs_A101/sss_fif/cbs_A101_mmr2_morph-vl.stc')
MEG_mmr1_v = np.load(root_path + 'cbsA_meeg_analysis/group_mmr1_vector_morph.npy') # with the mag or vector method
MEG_mmr2_v = np.load(root_path + 'cbsA_meeg_analysis/group_mmr2_vector_morph.npy') # with the mag or vector method
MEG_mmr1_m = np.load(root_path + 'cbsA_meeg_analysis/group_mmr1_morph.npy') # with the mag or vector method
MEG_mmr2_m = np.load(root_path + 'cbsA_meeg_analysis/group_mmr2_morph.npy') # with the mag or vector method

stc1.data = MEG_mmr1_m.mean(axis=0)
stc2.data = MEG_mmr2_m.mean(axis=0)

subject = 'fsaverage'
src = mne.read_source_spaces(subjects_dir + 'fsaverage/bem/fsaverage-vol-5-src.fif')

# stc.plot(src,clim=dict(kind="value",pos_lims=[0,2,5]),subject=subject, subjects_dir=subjects_dir)
stc1.plot(src,subject=subject, subjects_dir=subjects_dir)
stc2.plot(src,subject=subject, subjects_dir=subjects_dir)

#%%######################################## load EEG
EEG_mmr1 = np.load(root_path + 'cbsA_meeg_analysis/group_mmr1_eeg.npy')
EEG_mmr2 = np.load(root_path + 'cbsA_meeg_analysis/group_mmr2_eeg.npy')

plt.figure()
plot_err(stats.zscore(EEG_mmr2,axis=1),'k',stc1.times)
plot_err(stats.zscore(MEG_mmr2_m.mean(axis=1),axis=1),'b',stc1.times)
plot_err(stats.zscore(MEG_mmr2_v.mean(axis=1),axis=1),'r',stc1.times)
plt.legend(['EEG','','MEG mag','','MEG vector',''])
plt.xlabel('Time (ms)')
plt.ylabel('zscore')
plt.xlim([-100, 600])

mean_EEG_mmr2 = EEG_mmr2.mean(axis =0)
mean_MEG_mmr2_m = MEG_mmr2_m.mean(axis =0).mean(axis=0)
mean_MEG_mmr2_v = MEG_mmr2_v.mean(axis =0).mean(axis=0)

fig,(ax1,ax2,ax3) = plt.subplots(3,1)
ax1.plot(stc1.times,mean_EEG_mmr2)
ax2.plot(stc1.times,mean_MEG_mmr2_m)
ax3.plot(stc1.times,mean_MEG_mmr2_v)

#%%######################################## averaged across subjects
ts = 501 # 0ms
te = 2000# 200 ms

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

#%%######################################## for each subject: window pearson R
ts = 1000 # 100 ms
te = 1750 # 250 ms
times = stc1.times

## corr between EEG and averaged source MEG
# get the mean
mean_MEG_mmr2_m = MEG_mmr2_m[:,:,ts:te].mean(axis=-1).mean(axis=-1)
mean_MEG_mmr2_v = MEG_mmr2_v[:,:,ts:te].mean(axis=-1).mean(axis=-1)
mean_EEG_mmr2 = EEG_mmr2[:,ts:te].mean(axis=-1)

r,p = pearsonr(mean_MEG_mmr2_m,mean_EEG_mmr2)
print(r,p)
r,p = pearsonr(mean_MEG_mmr2_v,mean_EEG_mmr2)
print(r,p)

# get the max (peak) or min (trough)
max_MEG_mmr2_m = MEG_mmr2_m[:,:,ts:te].mean(axis=1).max(axis=-1)
max_MEG_mmr2_v = MEG_mmr2_v[:,:,ts:te].mean(axis=1).max(axis=-1)
max_EEG_mmr2 = EEG_mmr2[:,ts:te].max(axis=-1)

min_MEG_mmr2_m = MEG_mmr2_m[:,:,ts:te].mean(axis=1).min(axis=-1)
min_MEG_mmr2_v = MEG_mmr2_v[:,:,ts:te].mean(axis=1).min(axis=-1)
min_EEG_mmr2 = EEG_mmr2[:,ts:te].min(axis=-1)

r,p = pearsonr(max_MEG_mmr2_m,max_EEG_mmr2)
print(r,p)
r,p = pearsonr(max_MEG_mmr2_v,max_EEG_mmr2)
print(r,p)

# get the point-by-point
mean_MEG_mmr2_m = MEG_mmr2_m.mean(axis=1)
mean_MEG_mmr2_v = MEG_mmr2_v.mean(axis=1)

r_all_t = []
p_all_t = []
for t in np.arange(0,len(times),1):
    r,p = pearsonr(mean_MEG_mmr2_v[:,t],EEG_mmr2[:,t])
    r_all_t.append(r)
    p_all_t.append(p)

fig,(ax1,ax2,ax3,ax4,ax5) = plt.subplots(5,1)
ax1.plot(stc1.times,mean_EEG_mmr2)
ax2.plot(stc1.times,mean_MEG_mmr2_m.mean(axis=0).transpose())
ax3.plot(stc1.times,mean_MEG_mmr2_v.mean(axis=0).transpose())
ax4.plot(stc1.times,r_all_t)
ax5.plot(stc1.times,-np.log(p_all_t))

## corr between EEG and MEG ROI

## corr between EEG and each vertice -> plug back in to the stc for visualization

#%%######################################## for each subject: xcorr
for nsub in np.arange(0,len(mean_EEG_mmr2),1):
    xcorr = signal.correlate(mean_EEG_mmr2,mean_MEG_mmr2_m)
    lags = signal.correlation_lags(len(mean_EEG_mmr2),len(mean_MEG_mmr2_m))
    xcorr = np.abs(xcorr)

#%%######################################## for each subject: cosine similarity