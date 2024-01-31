#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 11:13:47 2023
Statistical test on MMR1 vs. MMR2
non-parametric cluster-based permutation t-test

@author: tzcheng
"""

import numpy as np
import mne
import scipy.stats as stats

#%% non-paramatric permutation test on EEG
root_path='/media/tzcheng/storage/CBS/'

## MMR
times = np.linspace(-0.1,0.6,3501)
std = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_std_eeg.npy')
dev1 = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_dev1_eeg.npy')
dev2 = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_dev2_eeg.npy')

ts = 500 # 0 s
te = 1750 # 0.25 s

MMR1 = dev1 - std
MMR2 = dev2 - std

X = MMR1-MMR2
X = X[:,ts:te]

T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(X, seed=0)
good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
for i in np.arange(0,len(good_cluster_inds),1):
    print("The " + str(i+1) + "st significant cluster")
    print(times[ts:te][clusters[good_cluster_inds[i]]])