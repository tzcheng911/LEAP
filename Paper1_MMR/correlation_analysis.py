#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 13:23:58 2023
Visualize and calculate EEG & MEG-measured MMR correlation in adults.

Correlation methods: pearson r, xcorr

@author: tzcheng
"""

import numpy as np
from numpy.linalg import norm
from scipy.stats import pearsonr
from scipy import signal
import pandas as pd
import copy
import random

#%%######################################## load data MMR
# MMR
root_path='/home/tzcheng/Documents/GitHub/Paper1_MMR/data/'
times = np.linspace(-0.1,0.6,3501)

MEG_mmr = np.load(root_path + 'adults/adult_group_mmr1_vector_morph.npy') 
EEG_mmr = np.load(root_path + 'adults/adult_group_mmr1_eeg.npy')

#%%######################################## Group level
ts = 1000 # 100 ms
te = 1750 # 250 ms

## MEG grand average across source points and subjects
mean_EEG_mmr = EEG_mmr.mean(axis =0)
mean_MEG_mmr = MEG_mmr.mean(axis =0).mean(axis=0)

#%%%% pearson r
corr_p = pearsonr(mean_EEG_mmr[ts:te], mean_MEG_mmr[ts:te])

#%%%% normalized cross correlation [-1 1]
a = (mean_EEG_mmr[ts:te]- np.mean(mean_EEG_mmr[ts:te]))/np.std(mean_EEG_mmr[ts:te])
b = (mean_MEG_mmr[ts:te]- np.mean(mean_MEG_mmr[ts:te]))/np.std(mean_MEG_mmr[ts:te])
a = a / np.linalg.norm(a)
b = b / np.linalg.norm(b)

xcorr = signal.correlate(a,b)
print(min(xcorr))
lags = signal.correlation_lags(len(a),len(b))
lags_time = lags/5000
print(lags_time[np.argmax(abs(xcorr))]) # if negative then the b is shifted left, if positive b is shifted right

#%%%% Sample-by-Sample pearson correlation between EEG & MEG
EEG = EEG_mmr
stc = MEG_mmr.mean(axis=1)
r_all_t = []
r_all_t_v = np.zeros([np.shape(MEG_mmr)[1],np.shape(MEG_mmr)[2]])

for t in np.arange(0,len(times),1):
    r,p = pearsonr(stc[:,t],EEG[:,t])
    r_all_t.append(r)

## whole-brain ttcorr
for v in np.arange(0,np.shape(MEG_mmr)[1],1):
    print('Vertex ' + str(v))
    for t in np.arange(0,len(times),1):
        r,p = pearsonr(EEG[:,t],MEG_mmr[:,v,t])
        r_all_t_v[v,t] = r
r_all_t_v = np.asarray(r_all_t_v)
np.save(root_path + '/cbsA_meeg_analysis/correlation/ttcorr_MMR1_conv_v.npy',r_all_t_v)

## permutation
n_perm=1000
r_all_t_perm = np.zeros([n_perm,len(times)])
for i in range(n_perm):
    print('Iteration' + str(i))
    EEG_p = copy.deepcopy(EEG).transpose() # transpose to shuffle the first dimension
    stc_p = copy.deepcopy(stc).transpose()
    np.random.shuffle(EEG_p)
    np.random.shuffle(stc_p)
    
    for t in np.arange(0,len(times),1):
        r,p = pearsonr(stc_p[t,:],EEG_p[t,:])
        r_all_t_perm[i,t] = r
r_all_t_perm = np.asarray(r_all_t_perm)
np.save('/media/tzcheng/storage2/CBS/cbsA_meeg_analysis/correlation/ttcorr_MMR1_conv_perm1000.npy',r_all_t_perm)

#%%######################################## Individual level
#%%%% Waveform-by-Waveform pearson correlation between EEG & MEG
EEG = EEG_mmr
stc = MEG_mmr[:,:,ts:te].mean(axis=1)

r_all_s = []

for s in np.arange(0,len(stc),1):
    r,p = pearsonr(stc[s,:],EEG[s,ts:te])
    r_all_s.append(r)

print('mean abs corr between MEG_v & EEG:' + str(np.abs(r_all_s).mean()))
print('std abs corr between MEG_v & EEG:' + str(np.abs(r_all_s).std()))

#%%%% Waveform-by-Waveform xcorr between EEG & MEG
xcorr_all_s = []

for s in np.arange(0,len(MEG_mmr),1):
    a = (EEG[s,ts:te] - np.mean(EEG[s,ts:te]))/np.std(EEG[s,ts:te])
    c = (stc[s,:] - np.mean(stc[s,:]))/np.std(stc[s,:])
    
    ## the way matlab do xcorr normalization: the max is 1 if do a zero lag autocorrealtoin
    a = a / np.linalg.norm(a)
    c = c / np.linalg.norm(c)
    
    xcorr = signal.correlate(a,c)
    xcorr_all_s.append(xcorr)

lags = signal.correlation_lags(len(a),len(c))
lags_time = lags/5000

print('mean max abs xcorr between MEG & EEG:' + str(np.max(np.abs(xcorr_all_s),axis=1).mean()))
print('mean max lag of abs xcorr between MEG & EEG:' + str(lags_time[np.argmax(np.abs(xcorr_all_s),axis=1)].mean()))
print('std max abs xcorr between MEG & EEG:' + str(np.max(np.abs(xcorr_all_s),axis=1).std()))
print('std max lag of abs xcorr between MEG & EEG:' + str(lags_time[np.argmax(np.abs(xcorr_all_s),axis=1)].std()))

#%% xcorr between EEG and each vertice: a bit slow, consider save/load the pickled file
# stc = MEG_mmr
# EEG = EEG_mmr
# xcorr_all_s = []
# lag_all_s = []

# for s in np.arange(0,len(MEG_mmr),1):
#     print('Now starting sub' + str(s))
#     for v in np.arange(0,np.shape(MEG_mmr)[1],1):
#         a = (EEG[s,ts:te] - np.mean(EEG[s,ts:te]))/np.std(EEG[s,ts:te])
#         c = (stc[s,v,ts:te] - np.mean(stc[s,v,ts:te]))/np.std(stc[s,v,ts:te])

#         ## the way matlab do xcorr normalization: the max is 1 if do a zero lag autocorrealtoin
#         a = a / np.linalg.norm(a)
#         c = c / np.linalg.norm(c)
        
#         xcorr = signal.correlate(a,c)
#         xcorr_all_s.append([s,v,max(abs(xcorr))])
#         lag_all_s.append([s,v,lags_time[np.argmax(abs(xcorr))]])
    
# df_v = pd.DataFrame(columns = ["Subject", "Vertno","XCorr MEG & EEG"], data = xcorr_all_s)
# df_v.to_pickle(root_path + 'adults/df_xcorr_MEGEEG_cont_mmr2.pkl')
# df_lag = pd.DataFrame(columns = ["Subject", "Vertno", "Lag XCorr MEG & EEG"], data = lag_all_s)
# df_v.to_pickle(root_path + 'adults/df_xcorr_lag_MEGEEG_cont_mmr2.pkl')