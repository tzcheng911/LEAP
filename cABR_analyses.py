#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 16:49:00 2024
cABR analysis

Analysis
1. cross-correlation
2. decoding
3. PCA

Level
1. sensor
2. ROI 
3. vertices

@author: tzcheng
"""

import mne
import matplotlib.pyplot as plt 
import numpy as np
import os
from scipy import stats,signal
from numpy import dot
from numpy.linalg import norm
from scipy.stats import pearsonr
import scipy as sp
import os
import seaborn as sns
import pandas as pd
import scipy.stats as stats
from scipy.io import wavfile
import time


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
import random
import mne
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne.preprocessing import Xdawn
from mne.decoding import (
    SlidingEstimator,
    GeneralizingEstimator,
    Scaler,
    cross_val_multiscore,
    LinearModel,
    get_coef,
    Vectorizer,
    CSP,
)

def plot_err(group_stc,color,t):
    group_avg=np.mean(group_stc,axis=0)
   #plt.figure()
    err=np.std(group_stc,axis=0)/np.sqrt(group_stc.shape[0])
    up=group_avg+err
    lw=group_avg-err
    plt.plot(t,group_avg,color=color)
    plt.fill_between(t,up,lw,color=color,alpha=0.5)

root_path='/media/tzcheng/storage2/CBS/'
subjects_dir = '/media/tzcheng/storage2/subjects/'
stc1 = mne.read_source_estimate(root_path + 'cbs_A101/sss_fif/cbs_A101_pa_cabr_morph-vl.stc')

#%%####################################### Load audio and cABR
## audio 
fs, ba_audio = wavfile.read('/media/tzcheng/storage2/CBS/stimuli/+10.wav')
fs, mba_audio = wavfile.read('/media/tzcheng/storage2/CBS/stimuli/-40.wav')
fs, pa_audio = wavfile.read('/media/tzcheng/storage2/CBS/stimuli/+40.wav')

# Downsample
fs_new = 5000
num_std = int((len(ba_audio)*fs_new)/fs)
num_dev = int((len(pa_audio)*fs_new)/fs)  # #sample_new/fs_new=#sample/fs find number of samples in the resampled data
audio_ba = signal.resample(ba_audio, num_std, t=None, axis=0, window=None)
audio_mba = signal.resample(mba_audio, num_dev, t=None, axis=0, window=None)
audio_pa = signal.resample(pa_audio, num_dev, t=None, axis=0, window=None)

## EEG
EEG_ba_cABR = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_std_cabr_eeg_200.npy')
EEG_mba_cABR = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_dev1_cabr_eeg_200.npy')
EEG_pa_cABR = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_dev2_cabr_eeg_200.npy')

## MEG: more than 200 trials for now
# adults
MEG_ba_cABR = np.load(root_path + 'cbsA_meeg_analysis/MEG/' + 'group_ba_cabr_morph.npy')
MEG_mba_cABR = np.load(root_path + 'cbsA_meeg_analysis/MEG/' + 'group_mba_cabr_morph.npy')
MEG_pa_cABR = np.load(root_path + 'cbsA_meeg_analysis/MEG/' + 'group_pa_cabr_morph.npy')

# infants 
MEG_ba_cABR = np.load(root_path + 'cbsb_meg_analysis/MEG/' + 'group_ba_cabr_morph.npy')
MEG_mba_cABR = np.load(root_path + 'cbsb_meg_analysis/MEG/' + 'group_mba_cabr_morph.npy')
MEG_pa_cABR = np.load(root_path + 'cbsb_meg_analysis/MEG/' + 'group_pa_cabr_morph.npy')

#%%####################################### visualize cABR of ba, mba and pa in EEG
plt.figure()
plt.subplot(311)
plot_err(EEG_ba_cABR,'k',stc1.times)
plt.xlim([-0.02,0.2])

plt.subplot(312)
plot_err(EEG_mba_cABR,'b',stc1.times)
plt.xlim([-0.02,0.2])
plt.ylabel('Amplitude')

plt.subplot(313)
plot_err(EEG_pa_cABR,'r',stc1.times)
plt.xlim([-0.02,0.2])
plt.xlabel('Time (s)')

## visualize cABR of ba, mba and pa in MEG
plt.figure()
plt.subplot(311)
plot_err(MEG_ba_cABR.mean(axis=1),'k',stc1.times)
plt.xlim([-0.02,0.2])

plt.subplot(312)
plot_err(MEG_mba_cABR.mean(axis=1),'b',stc1.times)
plt.xlim([-0.02,0.2])
plt.ylabel('Amplitude')

plt.subplot(313)
plot_err(MEG_pa_cABR.mean(axis=1),'r',stc1.times)
plt.xlim([-0.02,0.2])
plt.xlabel('Time (s)')

#%%####################################### Sliding estimator decoding
tic = time.time()
root_path='/media/tzcheng/storage2/CBS/'
subjects_dir = '/media/tzcheng/storage2/subjects/'
os.chdir(root_path)
stc1 = mne.read_source_estimate(root_path + 'cbs_A101/sss_fif/cbs_A101_ba_cabr_morph-vl.stc')
times = stc1.times

## parameter
ROI_wholebrain = 'wholebrain' # ROI or wholebrain or sensor
k_feature = 'all' # ROI: 'all' features; whole brain: 500 features

filename = 'cabr'
filename_cabr_ba = 'group_ba_cabr_morph'
filename_cabr_mba = 'group_mba_cabr_morph'
filename_cabr_pa = 'group_pa_cabr_morph'

fname_aseg = subjects_dir + 'fsaverage/mri/aparc+aseg.mgz'
label_names = np.asarray(mne.get_volume_labels_from_aseg(fname_aseg))
lh_ROI_label = [72,60,61,62] # Simport random
rh_ROI_label = [108,96,97,98] # STG and IFG (parsopercularis, parsorbitalis, parstriangularis)

if ROI_wholebrain == 'ROI':
    cabr_ba = np.load(root_path + 'cbsA_meeg_analysis/MEG/' + filename_cabr_ba + '_roi.npy',allow_pickle=True)
    cabr_mba = np.load(root_path + 'cbsA_meeg_analysis/MEG/' + filename_cabr_mba + '_roi.npy',allow_pickle=True)
    cabr_pa = np.load(root_path + 'cbsA_meeg_analysis/MEG/' + filename_cabr_pa + '_roi.npy',allow_pickle=True)
elif ROI_wholebrain == 'wholebrain':
    cabr_ba = np.load(root_path + 'cbsA_meeg_analysis/MEG/' + filename_cabr_ba + '.npy',allow_pickle=True)
    cabr_mba = np.load(root_path + 'cbsA_meeg_analysis/MEG/' + filename_cabr_mba + '.npy',allow_pickle=True)
    cabr_pa = np.load(root_path + 'cbsA_meeg_analysis/MEG/' + filename_cabr_pa + '.npy',allow_pickle=True)
else:
    print("Need to decide whether to use ROI or whole brain as feature.")
X = np.concatenate((cabr_ba,cabr_mba,cabr_pa),axis=0)
# X = X[:,:,ts:te] 
y = np.concatenate((np.repeat(0,len(cabr_ba)),np.repeat(1,len(cabr_ba)),np.repeat(2,len(cabr_ba)))) #0 is for mmr1 and 1 is for mmr2

# prepare a series of classifier applied at each time sample
clf = make_pipeline(
    StandardScaler(),  # z-score normalization
    SelectKBest(f_classif, k=k_feature),  # select features for speed
    LinearModel(),
    )
time_decod = SlidingEstimator(clf)

# Run cross-validated decoding analyses
scores_observed = cross_val_multiscore(time_decod, X, y, cv=5, n_jobs=None) # leave one out
score = np.mean(scores_observed, axis=0)

#Plot average decoding scores of 5 splits
# TOI = np.linspace(0,450,num=2250)
# fig, ax = plt.subplots(1)
# ax.plot(TOI, scores_observed.mean(0), label="score")
# ax.axhline(0.5, color="k", linestyle="--", label="chance")
# ax.axvline(0, color="k")
# plt.legend()

# The fitting needs not be cross validated because the weights are based on
# the training sets
time_decod.fit(X, y) # not changed after shuffling the initial
# Retrieve patterns after inversing the z-score normalization step:
patterns = get_coef(time_decod, "patterns_", inverse_transform=True)

toc = time.time()
print('It takes ' + str((toc - tic)/60) + 'min to run decoding')

np.save(root_path + 'cbsA_meeg_analysis/decoding/roc_auc_kall_' + filename + '.npy',scores_observed)
np.save(root_path + 'cbsA_meeg_analysis/decoding/patterns_kall_' + filename + '.npy',patterns)

#%%####################################### Cross-correlation
root_path='/media/tzcheng/storage2/CBS/'
subjects_dir = '/media/tzcheng/storage2/subjects/'
os.chdir(root_path)
stc1 = mne.read_source_estimate(root_path + 'cbs_A101/sss_fif/cbs_A101_ba_cabr_morph-vl.stc')
times = stc1.times

## parameter
ROI_wholebrain = 'wholebrain' # ROI or wholebrain or sensor
k_feature = 'all' # ROI: 'all' features; whole brain: 500 features
ts = 100
te = 1100

filename = 'cabr'
filename_cabr_ba = 'group_ba_cabr_morph'
filename_cabr_mba = 'group_mba_cabr_morph'
filename_cabr_pa = 'group_pa_cabr_morph'

fname_aseg = subjects_dir + 'fsaverage/mri/aparc+aseg.mgz'
label_names = np.asarray(mne.get_volume_labels_from_aseg(fname_aseg))

if ROI_wholebrain == 'ROI':
    cabr_ba = np.load(root_path + 'cbsA_meeg_analysis/MEG/' + filename_cabr_ba + '_roi.npy',allow_pickle=True)
    cabr_mba = np.load(root_path + 'cbsA_meeg_analysis/MEG/' + filename_cabr_mba + '_roi.npy',allow_pickle=True)
    cabr_pa = np.load(root_path + 'cbsA_meeg_analysis/MEG/' + filename_cabr_pa + '_roi.npy',allow_pickle=True)
elif ROI_wholebrain == 'wholebrain':
    cabr_ba = np.load(root_path + 'cbsA_meeg_analysis/MEG/' + filename_cabr_ba + '.npy',allow_pickle=True)
    cabr_mba = np.load(root_path + 'cbsA_meeg_analysis/MEG/' + filename_cabr_mba + '.npy',allow_pickle=True)
    cabr_pa = np.load(root_path + 'cbsA_meeg_analysis/MEG/' + filename_cabr_pa + '.npy',allow_pickle=True)
else:
    print("Need to decide whether to use ROI or whole brain as feature.")

## Only need to calculate audio once 
a_ba = (audio_ba - np.mean(audio_ba))/np.std(audio_ba)
a_ba = a_ba / np.linalg.norm(a_ba)
a_mba = (audio_mba - np.mean(audio_mba))/np.std(audio_mba)
a_mba = a_mba / np.linalg.norm(a_mba)
a_pa = (audio_pa - np.mean(audio_pa))/np.std(audio_pa)
a_pa = a_pa / np.linalg.norm(a_pa)

## Xcorr between audio and each vertice in MEG for averaged across subjects
lags = signal.correlation_lags(len(a_ba),len(cabr_ba[0,0,ts:te]))
lags_time = lags/5000
xcorr_all_v = []

mean_cabr_ba = cabr_ba.mean(axis=0)
mean_cabr_mba = cabr_mba.mean(axis=0)
mean_cabr_pa = cabr_pa.mean(axis=0)

for v in np.arange(0,np.shape(cabr_ba)[1],1):
    b_ba = (mean_cabr_ba[v,ts:te] - np.mean(mean_cabr_ba[v,ts:te]))/np.std(mean_cabr_ba[v,ts:te])     
    b_ba = b_ba / np.linalg.norm(b_ba)
    b_mba = (mean_cabr_mba[v,ts:te] - np.mean(mean_cabr_mba[v,ts:te]))/np.std(mean_cabr_mba[v,ts:te])     
    b_mba = b_mba / np.linalg.norm(b_mba)
    b_pa = (mean_cabr_pa[v,ts:te] - np.mean(mean_cabr_pa[v,ts:te]))/np.std(mean_cabr_pa[v,ts:te])     
    b_pa = b_pa / np.linalg.norm(b_pa)
    
    xcorr_ba = signal.correlate(a_ba,b_ba)
    xcorr_mba = signal.correlate(a_mba,b_mba)
    xcorr_pa = signal.correlate(a_pa,b_pa)
    
    xcorr_all_v.append([v,max(abs(xcorr_ba)),lags_time[np.argmax(abs(xcorr_ba))],max(abs(xcorr_mba)),
                        lags_time[np.argmax(abs(xcorr_mba))], max(abs(xcorr_pa)),lags_time[np.argmax(abs(xcorr_pa))]])
df_v = pd.DataFrame(columns = ["Vertno", "abs XCorr MEG & audio_ba", "max Lag MEG & audio_ba", 
                               "abs XCorr MEG & audio_mba", "max Lag MEG & audio_mba", 
                               "abs XCorr MEG & audio_pa", "max Lag MEG & audio_pa"], data = xcorr_all_v)
if ROI_wholebrain == 'ROI':
    df_v.to_pickle(root_path + 'cbsA_meeg_analysis/' + filename + '_df_xcorr_roi.pkl')
elif ROI_wholebrain == 'wholebrain': 
    df_v.to_pickle(root_path + 'cbsA_meeg_analysis/' + filename + '_df_xcorr_v.pkl')

## Xcorr between audio and each vertice in MEG for each individual
xcorr_all_v_s = []
lag_all_v_s = []

for s in np.arange(0,len(cabr_ba),1):
    print('Now starting sub' + str(s))
    for v in np.arange(0,np.shape(cabr_ba)[1],1):
        b_ba = (cabr_ba[s,v,ts:te] - np.mean(cabr_ba[s,v,ts:te]))/np.std(cabr_ba[s,v,ts:te])     
        b_ba = b_ba / np.linalg.norm(b_ba)
        b_mba = (cabr_mba[s,v,ts:te] - np.mean(cabr_mba[s,v,ts:te]))/np.std(cabr_mba[s,v,ts:te])     
        b_mba = b_mba / np.linalg.norm(b_mba)
        b_pa = (cabr_pa[s,v,ts:te] - np.mean(cabr_pa[s,v,ts:te]))/np.std(cabr_pa[s,v,ts:te])     
        b_pa = b_pa / np.linalg.norm(b_pa)

        xcorr_ba = signal.correlate(a_ba,b_ba)
        xcorr_mba = signal.correlate(a_mba,b_mba)
        xcorr_pa = signal.correlate(a_pa,b_pa)
        
        xcorr_all_v_s.append([s,v,max(abs(xcorr_ba)),lags_time[np.argmax(abs(xcorr_ba))],max(abs(xcorr_mba)),
                            lags_time[np.argmax(abs(xcorr_mba))], max(abs(xcorr_pa)),lags_time[np.argmax(abs(xcorr_pa))]])
df_v_s = pd.DataFrame(columns = ["Subject","Vertno", "abs XCorr MEG & audio_ba", "max Lag MEG & audio_ba", 
                                   "abs XCorr MEG & audio_mba", "max Lag MEG & audio_mba", 
                                   "abs XCorr MEG & audio_pa", "max Lag MEG & audio_pa"], data = xcorr_all_v_s)
if ROI_wholebrain == 'ROI':
    df_v_s.to_pickle(root_path + 'cbsA_meeg_analysis/' + filename + '_df_xcorr_roi_s.pkl')
elif ROI_wholebrain == 'wholebrain': 
    df_v_s.to_pickle(root_path + 'cbsA_meeg_analysis/' + filename + '_df_xcorr_v_s.pkl')
