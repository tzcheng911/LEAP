#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 12:00:00 2023
Used to get the normalized peak of the MMR vector method as in SSEP method. 
Use with vis_MMR.py to visualize the results.

@author: tzcheng
"""

import mne 
import numpy as np
import itertools
import matplotlib.pyplot as plt 

def plot_err(group_stc,color,t):
    group_avg=np.mean(group_stc,axis=0)
   #plt.figure()
    err=np.std(group_stc,axis=0)/np.sqrt(group_stc.shape[0])
    up=group_avg+err
    lw=group_avg-err
    plt.plot(t,group_avg,color=color)
    plt.fill_between(t,up,lw,color=color,alpha=0.5)

#%%#######################################   
root_path='/media/tzcheng/storage/CBS/'
subjects_dir = '/media/tzcheng/storage2/subjects/'

## Load the vectors time series 
stc1 = mne.read_source_estimate(root_path + 'cbs_A101/sss_fif/cbs_A101_mmr2_morph-vl.stc')
MEG_mmr1_v = np.load(root_path + 'cbsA_meeg_analysis/group_mmr1_vector_morph.npy') # with the mag or vector method
MEG_mmr2_v = np.load(root_path + 'cbsA_meeg_analysis/group_mmr2_vector_morph.npy') # with the mag or vector method
MEG_mmr1_roi_v = np.load(root_path + 'cbsA_meeg_analysis/group_mmr1_vector_morph_roi.npy') # with the mag or vector method
MEG_mmr2_roi_v = np.load(root_path + 'cbsA_meeg_analysis/group_mmr2_vector_morph_roi.npy') # with the mag or vector method
times = stc1.times

## Get the mean across vertex
mean_MEG_mmr1_v = np.mean(MEG_mmr1_v,axis=1)
mean_MEG_mmr2_v = np.mean(MEG_mmr2_v,axis=1)

#%%#######################################
## "Normalize" the peak by neighboring bins as in Nozaradan
sub_bin = 3; # critical parameter follow Nozaradan 2011 parameter n - mean(n-5, n-4, n-3, n+3, n+4, n+5)
nt = np.shape(mean_MEG_mmr1_v)[1]

mean_MEG_mmr1_v_subt = []
mean_MEG_mmr2_v_subt = []
t_ind = []
for t in np.arange(sub_bin,nt-sub_bin,1):
    t_ind.append(t)
    subt_ind = [t-sub_bin,t-sub_bin+1,t+sub_bin-1,t+sub_bin] # t-2, t-1, t+1, t+2
    subt = mean_MEG_mmr1_v[:,t] - np.mean(mean_MEG_mmr1_v[:,subt_ind],axis=1)
    mean_MEG_mmr1_v_subt.append(subt)
    subt = mean_MEG_mmr2_v[:,t] - np.mean(mean_MEG_mmr2_v[:,subt_ind],axis=1)
    mean_MEG_mmr2_v_subt.append(subt)

mean_MEG_mmr1_v_subt = np.transpose(mean_MEG_mmr1_v_subt)
mean_MEG_mmr2_v_subt = np.transpose(mean_MEG_mmr2_v_subt)

new_times = times[t_ind]

#%%#######################################   
plt.figure()
plot_err(mean_MEG_mmr1_v,'m',times)
plot_err(mean_MEG_mmr2_v,'r',times)

plt.figure()
plot_err(mean_MEG_mmr1_v_subt,'b',new_times)
plot_err(mean_MEG_mmr2_v_subt,'r',new_times)