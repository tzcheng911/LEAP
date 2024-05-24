#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 16:49:00 2024
FFR analysis still under development

Analysis
1. Decoding
2. X-corr
3. PCA
4. spectrum

Level
1. sensor
2. ROI 
3. vertices

@author: tzcheng
"""

## Import library  
import mne
from mne.decoding import UnsupervisedSpatialFilter
from mne.preprocessing import ICA
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

from sklearn.decomposition import PCA, FastICA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
import random
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
src = mne.read_source_spaces(subjects_dir + 'fsaverage/bem/fsaverage-vol-5-src.fif')

#%%####################################### Load audio and FFR
## audio 
fs, ba_audio = wavfile.read(root_path + 'stimuli/+10.wav')
fs, mba_audio = wavfile.read(root_path + 'stimuli/-40.wav')
fs, pa_audio = wavfile.read(root_path + 'stimuli/+40.wav')

# Downsample
fs_new = 5000
num_std = int((len(ba_audio)*fs_new)/fs)
num_dev = int((len(pa_audio)*fs_new)/fs)  # #sample_new/fs_new=#sample/fs find number of samples in the resampled data
audio_ba = signal.resample(ba_audio, num_std, t=None, axis=0, window=None)
audio_mba = signal.resample(mba_audio, num_dev, t=None, axis=0, window=None)
audio_pa = signal.resample(pa_audio, num_dev, t=None, axis=0, window=None)

## EEG
EEG_ba_FFR = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_std_ffr_eeg_200.npy')
EEG_mba_FFR = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_dev1_ffr_eeg_200.npy')
EEG_pa_FFR = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_dev2_ffr_eeg_200.npy')

## MEG sensors
MEG_ba_FFR = np.load(root_path + 'cbsA_meeg_analysis/MEG/FFR/' + 'group_ba_ffr_sensor.npy')
MEG_mba_FFR = np.load(root_path + 'cbsA_meeg_analysis/MEG/FFR/' + 'group_mba_ffr_sensor.npy')
MEG_pa_FFR = np.load(root_path + 'cbsA_meeg_analysis/MEG/FFR/' + 'group_pa_ffr_sensor.npy')

## MEG source vertices: more than 200 trials for nowrand_ind = np.arange(0,len(X))
random.Random(0).shuffle(rand_ind)
X = X[rand_ind,:,:]
y = y[rand_ind]
# adults
MEG_ba_FFR = np.load(root_path + 'cbsA_meeg_analysis/MEG/FFR/' + 'group_ba_ffr_morph.npy')
MEG_mba_FFR = np.load(root_path + 'cbsA_meeg_analysis/MEG/FFR/' + 'group_mba_ffr_morph.npy')
MEG_pa_FFR = np.load(root_path + 'cbsA_meeg_analysis/MEG/FFR/' + 'group_pa_ffr_morph.npy')

# infants 
MEG_ba_FFR = np.load(root_path + 'cbsb_meg_analysis/MEG/FFR/' + 'group_ba_ffr_morph.npy')
MEG_mba_FFR = np.load(root_path + 'cbsb_meg_analysis/MEG/FFR/' + 'group_mba_ffr_morph.npy')
MEG_pa_FFR = np.load(root_path + 'cbsb_meg_analysis/MEG/FFR/' + 'group_pa_ffr_morph.npy')

## MEG source ROI: more than 200 trials for now
# adults
MEG_ba_FFR = np.load(root_path + 'cbsA_meeg_analysis/MEG/FFR/' + 'group_ba_ffr_f80450_morph_roi.npy')
MEG_mba_FFR = np.load(root_path + 'cbsA_meeg_analysis/MEG/FFR/' + 'group_mba_ffr_f80450_morph_roi.npy')
MEG_pa_FFR = np.load(root_path + 'cbsA_meeg_analysis/MEG/FFR/' + 'group_pa_ffr_f80450_morph_roi.npy')

## MEG sensor: more than 200 trials for now
# adults
MEG_ba_FFR = np.load(root_path + 'cbsA_meeg_analysis/MEG/FFR/' + 'group_ba_sensor.npy')
MEG_mba_FFR = np.load(root_path + 'cbsA_meeg_analysis/MEG/FFR/' + 'group_mba_sensor.npy')
MEG_pa_FFR = np.load(root_path + 'cbsA_meeg_analysis/MEG/FFR/' + 'group_pa_sensor.npy')

#%%####################################### Subject-by-subject MEG decoding for each condition 
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

filename = 'ffr'
filename_ffr_ba = 'group_ba_ffr_morph'
filename_ffr_mba = 'group_mba_ffr_morph'
filename_ffr_pa = 'group_pa_ffr_morph'

fname_aseg = subjects_dir + 'fsaverage/mri/aparc+aseg.mgz'
label_names = np.asarray(mne.get_volume_labels_from_aseg(fname_aseg))
lh_ROI_label = [12, 72,76,74] # [subcortical] brainstem,[AC] STG, transversetemporal, [controls] frontal pole
rh_ROI_label = [12, 108,112,110] # [subcortical] brainstem,[AC] STG, transversetemporal, [controls] frontal pole

if ROI_wholebrain == 'ROI':
    ffr_ba = np.load(root_path + 'cbsA_meeg_analysis/MEG/FFR/' + filename_ffr_ba + '_roi.npy',allow_pickle=True)
    ffr_mba = np.load(root_path + 'cbsA_meeg_analysis/MEG/FFR/' + filename_ffr_mba + '_roi.npy',allow_pickle=True)
    ffr_pa = np.load(root_path + 'cbsA_meeg_analysis/MEG/FFR/' + filename_ffr_pa + '_roi.npy',allow_pickle=True)
elif ROI_wholebrain == 'wholebrain':
    ffr_ba = np.load(root_path + 'cbsA_meeg_analysis/MEG/FFR/' + filename_ffr_ba + '.npy',allow_pickle=True)
    ffr_mba = np.load(root_path + 'cbsA_meeg_analysis/MEG/FFR/' + filename_ffr_mba + '.npy',allow_pickle=True)
    ffr_pa = np.load(root_path + 'cbsA_meeg_analysis/MEG/FFR/' + filename_ffr_pa + '.npy',allow_pickle=True)
else:
    print("Need to decide whether to use ROI or whole brain as feature.")
X = np.concatenate((ffr_ba,ffr_mba,ffr_pa),axis=0)
# X = X[:,:,ts:te] 
y = np.concatenate((np.repeat(0,len(ffr_ba)),np.repeat(1,len(ffr_ba)),np.repeat(2,len(ffr_ba)))) #0 is for mmr1 and 1 is for mmr2

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

np.save(root_path + 'cbsA_meeg_analysis/decoding/roc_auc_kall_' + filename + '.npy',scores_observed)
np.save(root_path + 'cbsA_meeg_analysis/decoding/patterns_kall_' + filename + '.npy',patterns)

#%%####################################### MEG decoding across time
root_path='/media/tzcheng/storage2/CBS/'
subjects_dir = '/media/tzcheng/storage2/subjects/'
os.chdir(root_path)

n_top = 3
n_trial = 'all' # 'ntrial_200/' or 'ntrial_all/' or ''
stc1 = mne.read_source_estimate(root_path + 'cbs_A101/sss_fif/cbs_A101_ba_cabr_morph-vl.stc')
times = stc1.times

did_pca = '_pcffr80450_'  # without or with pca "_pcffr"
filename_ffr_ba = 'group_ba' + did_pca
filename_ffr_mba = 'group_mba' + did_pca
filename_ffr_pa = 'group_pa' + did_pca

fname_aseg = subjects_dir + 'fsaverage/mri/aparc+aseg.mgz'
label_names = np.asarray(mne.get_volume_labels_from_aseg(fname_aseg))

## FFR relevant ROIs
lh_ROI_label = [12, 72,76,74] # [subcortical] brainstem,[AC] STG, transversetemporal, [controls] frontal pole
rh_ROI_label = [12, 108,112,110] # [subcortical] brainstem,[AC] STG, transversetemporal, [controls] frontal pole

baby_or_adult = 'cbsb_meg_analysis' # baby or adult
input_data = 'wholebrain' # ROI or wholebrain or sensor or pcffr
k_feature = 'all' # ROI: 'all' features; whole brain: 500 features

if input_data == 'sensor':
    ffr_ba = np.load(root_path + baby_or_adult + '/MEG/FFR/' + 'ntrial_' + str(n_trial)+ '/' + filename_ffr_ba + str(n_top) + '_' + str(n_trial) +'_sensor.npy',allow_pickle=True)
    ffr_mba = np.load(root_path + baby_or_adult + '/MEG/FFR/'+ 'ntrial_' + str(n_trial) + '/'  + filename_ffr_mba + str(n_top) + '_' + str(n_trial) + '_sensor.npy',allow_pickle=True)
    ffr_pa = np.load(root_path + baby_or_adult + '/MEG/FFR/'+ 'ntrial_' + str(n_trial) + '/'  + filename_ffr_pa + str(n_top) + '_' + str(n_trial) + '_sensor.npy',allow_pickle=True)
elif input_data == 'ROI':
    ffr_ba = np.load(root_path + baby_or_adult +'/MEG/FFR/'+ 'ntrial_' + str(n_trial) + '/'  + filename_ffr_ba + str(n_top) + '_' + str(n_trial) +'_morph_roi.npy',allow_pickle=True)
    ffr_mba = np.load(root_path + baby_or_adult +'/MEG/FFR/'+ 'ntrial_' + str(n_trial) + '/'  + filename_ffr_mba + str(n_top) + '_' + str(n_trial) + '_morph_roi.npy',allow_pickle=True)
    ffr_pa = np.load(root_path + baby_or_adult +'/MEG/FFR/'+ 'ntrial_' + str(n_trial) + '/'  + filename_ffr_pa + str(n_top) + '_' + str(n_trial) + '_morph_roi.npy',allow_pickle=True)
elif input_data == 'wholebrain':
    ffr_ba = np.load(root_path + baby_or_adult + '/MEG/FFR/' + 'ntrial_' + str(n_trial) + '/'  + filename_ffr_ba + str(n_top) + '_' + str(n_trial) + '_morph.npy',allow_pickle=True)
    ffr_mba = np.load(root_path + baby_or_adult + '/MEG/FFR/' + 'ntrial_' + str(n_trial) + '/'  + filename_ffr_mba + str(n_top) + '_' + str(n_trial) +'_morph.npy',allow_pickle=True)
    ffr_pa = np.load(root_path + baby_or_adult + '/MEG/FFR/' + 'ntrial_' + str(n_trial) + '/'  + filename_ffr_pa + str(n_top) + '_' + str(n_trial) + '_morph.npy',allow_pickle=True)
else:
    print("Need to decide whether to use ROI or whole brain as feature.")

all_score = []
## Three way classification using ovr
X = np.concatenate((ffr_ba,ffr_mba,ffr_pa),axis=0)
y = np.concatenate((np.repeat(0,len(ffr_ba)),np.repeat(1,len(ffr_mba)),np.repeat(2,len(ffr_pa))))

# X = np.concatenate((ffr_ba,ffr_mba),axis=0)
# y = np.concatenate((np.repeat(0,len(ffr_ba)),np.repeat(1,len(ffr_mba))))

# X = np.concatenate((ffr_ba,ffr_pa),axis=0)
# y = np.concatenate((np.repeat(0,len(ffr_ba)),np.repeat(2,len(ffr_pa))))

rand_ind = np.arange(0,len(X))
random.Random(15).shuffle(rand_ind)
X = X[rand_ind,:,:]

## 1st vs. 2nd half decoding
# X1 = X[rand_ind,:,:np.shape(X)[-1]//2]
# X2 = X[rand_ind,:,np.shape(X)[-1]//2:]

y = y[rand_ind]

clf = make_pipeline(
    StandardScaler(),  # z-score normalization
    LogisticRegression(solver="liblinear")  # liblinear is faster than lbfgs
)    
for n in np.arange(0,np.shape(X)[1],1):
        scores = cross_val_multiscore(clf, X[:,n,:], y, cv=5, n_jobs=4) # takes about 10 mins to run
        score = np.mean(scores, axis=0)
        print("Data " + str(n+1) + " Accuracy: %0.1f%%" % (100 * score,))
        all_score.append(score)
np.save(root_path + baby_or_adult +'/decoding/PCFFR80450_'+ str(n_top) + '_ntrial_' + str(n_trial) + '_decoding_accuracy_' + input_data +'_r15.npy',all_score)

#%%####################################### check acc for each sensor, ROI or vertice
acc_ind = np.where(np.array(all_score) >= 0.5)

## visualize sensor
evoked = mne.read_evokeds(root_path + 'cbs_A123/sss_fif/cbs_A123_01_otp_raw_sss_proj_f_evoked_substd_cabr.fif')[0]
ch_name = np.array(evoked.ch_names)
evoked.info['bads'] = ch_name[acc_ind[0]].tolist() # hack the evoked.info['bads'] to visualize the high decoding accuracy sensor
evoked.plot_sensors(ch_type='all',kind ='3d')

## visualize ROI
label_names[acc_ind] # ctx-rh-bankssts reached 0.46363636 decoding accuracy for adults, ctx-rh-middletemporal reached 0.48214286 for infants
np.sort(all_score) 
np.argsort(all_score)

## visualize vertice
stc1.data = np.array([all_score,all_score]).transpose()
stc1.plot(src, clim=dict(kind="percent",pos_lims=[90,95,99]), subject='fsaverage', subjects_dir=subjects_dir)

#%%####################################### Cross-correlation audio and MEG sensor and source
root_path='/media/tzcheng/storage2/CBS/'
subjects_dir = '/media/tzcheng/storage2/subjects/'
os.chdir(root_path)
stc1 = mne.read_source_estimate(root_path + 'cbs_A101/sss_fif/cbs_A101_ba_cabr_morph-vl.stc')
times = stc1.times

## parameter
ROI_wholebrain = 'wholebrain' # ROI or wholebrain or sensor
ts = 100
te = 1100

filename = 'ffr'
filename_ffr_ba = 'group_ba_ffr_morph'
filename_ffr_mba = 'group_mba_ffr_morph'
filename_ffr_pa = 'group_pa_ffr_morph'

fname_aseg = subjects_dir + 'fsaverage/mri/aparc+aseg.mgz'
label_names = np.asarray(mne.get_volume_labels_from_aseg(fname_aseg))

## FFR relevant ROIs
lh_ROI_label = [12, 72,76,74] # [subcortical] brainstem,[AC] STG, transversetemporal, [controls] frontal pole
rh_ROI_label = [12, 108,112,110] # [subcortical] brainstem,[AC] STG, transversetemporal, [controls] frontal pole

if ROI_wholebrain == 'sensor':
    FFR_ba = np.load(root_path + 'cbsA_meeg_analysis/MEG/FFR/' + filename_ffr_ba + '_sensor.npy',allow_pickle=True)
    FFR_mba = np.load(root_path + 'cbsA_meeg_analysis/MEG/FFR/' + filename_ffr_mba + '_sensor.npy',allow_pickle=True)
    FFR_pa = np.load(root_path + 'cbsA_meeg_analysis/MEG/FFR/' + filename_ffr_pa + '_sensor.npy',allow_pickle=True)
elif ROI_wholebrain == 'ROI':
    FFR_ba = np.load(root_path + 'cbsA_meeg_analysis/MEG/FFR/' + filename_ffr_ba + '_morph_roi.npy',allow_pickle=True)
    FFR_mba = np.load(root_path + 'cbsA_meeg_analysis/MEG/FFR/' + filename_ffr_mba + '_morph_roi.npy',allow_pickle=True)
    FFR_pa = np.load(root_path + 'cbsA_meeg_analysis/MEG/FFR/' + filename_ffr_pa + '_morph_roi.npy',allow_pickle=True)
elif ROI_wholebrain == 'wholebrain':
    FFR_ba = np.load(root_path + 'cbsA_meeg_analysis/MEG/FFR/' + filename_ffr_ba + '_morph.npy',allow_pickle=True)
    FFR_mba = np.load(root_path + 'cbsA_meeg_analysis/MEG/FFR/' + filename_ffr_mba + '_morph.npy',allow_pickle=True)
    FFR_pa = np.load(root_path + 'cbsA_meeg_analysis/MEG/FFR/' + filename_ffr_pa + '_morph.npy',allow_pickle=True)
else:
    print("Need to decide whether to use ROI or whole brain as feature.")

## Only need to calculate audio once 
# a_ba = (audio_ba - np.mean(audio_ba))/np.std(audio_ba)
# a_ba = a_ba / np.linalg.norm(a_ba)
# a_mba = (audio_mba - np.mean(audio_mba))/np.std(audio_mba)
# a_mba = a_mba / np.linalg.norm(a_mba)
# a_pa = (audio_pa - np.mean(audio_pa))/np.std(audio_pa)
# a_pa = a_pa / np.linalg.norm(a_pa)

## Get the EEG
mean_EEG_ba_FFR = EEG_ba_FFR.mean(axis=0)
mean_EEG_mba_FFR = EEG_mba_FFR.mean(axis=0)
mean_EEG_pa_FFR = EEG_pa_FFR.mean(axis=0)

a_ba = (mean_EEG_ba_FFR - np.mean(mean_EEG_ba_FFR))/np.std(mean_EEG_ba_FFR)
EEG_ba_FFR = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_std_ffr_eeg_200.npy')
EEG_mba_FFR = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_dev1_ffr_eeg_200.npy')
EEG_pa_FFR = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_dev2_ffr_eeg_200.npy')
a_ba = a_ba / np.linalg.norm(a_ba)
a_mba = (mean_EEG_mba_FFR - np.mean(mean_EEG_mba_FFR))/np.std(mean_EEG_mba_FFR)
a_mba = a_mba / np.linalg.norm(a_mba)
a_pa = (mean_EEG_pa_FFR - np.mean(mean_EEG_pa_FFR))/np.std(mean_EEG_pa_FFR)
a_pa = a_pa / np.linalg.norm(a_pa)

## Xcorr between audio and each vertice in MEG for averaged across subjects
lags = signal.correlation_lags(len(a_ba),len(FFR_ba[0,0,ts:te]))
lags_time = lags/5000
xcorr_all_v = []

mean_FFR_ba = FFR_ba.mean(axis=0)
mean_FFR_mba = FFR_mba.mean(axis=0)
mean_FFR_pa = FFR_pa.mean(axis=0)

for v in np.arange(0,np.shape(FFR_ba)[1],1):
    b_ba = (mean_FFR_ba[v,ts:te] - np.mean(mean_FFR_ba[v,ts:te]))/np.std(mean_FFR_ba[v,ts:te])     
    b_ba = b_ba / np.linalg.norm(b_ba)
    b_mba = (mean_FFR_mba[v,ts:te] - np.mean(mean_FFR_mba[v,ts:te]))/np.std(mean_FFR_mba[v,ts:te])     
    b_mba = b_mba / np.linalg.norm(b_mba)
    b_pa = (mean_FFR_pa[v,ts:te] - np.mean(mean_FFR_pa[v,ts:te]))/np.std(mean_FFR_pa[v,ts:te])     
    b_pa = b_pa / np.linalg.norm(b_pa)
    
    xcorr_ba = signal.correlate(a_ba,b_ba)
    xcorr_mba = signal.correlate(a_mba,b_mba)
    xcorr_pa = signal.correlate(a_pa,b_pa)
    
    xcorr_all_v.append([v,max(abs(xcorr_ba)),lags_time[np.argmax(abs(xcorr_ba))],max(abs(xcorr_mba)),
                        lags_time[np.argmax(abs(xcorr_mba))], max(abs(xcorr_pa)),lags_time[np.argmax(abs(xcorr_pa))]])
df_v = pd.DataFrame(columns = ["Vertno", "abs XCorr MEG & ba", "max Lag MEG & ba", 
                               "abs XCorr MEG & mba", "max Lag MEG & mba", 
                               "abs XCorr MEG & pa", "max Lag MEG & pa"], data = xcorr_all_v)
if ROI_wholebrain == 'ROI':
    df_v.to_pickle(root_path + 'cbsA_meeg_analysis/' + filename + '_df_xcorr_MEGEEG_roi.pkl')
elif ROI_wholebrain == 'wholebrain': 
    df_v.to_pickle(root_path + 'cbsA_meeg_analysis/' + filename + '_df_xcorr_MEGEEG_v.pkl')
elif ROI_wholebrain == 'sensor':
    df_v.to_pickle(root_path + 'cbsA_meeg_analysis/' + filename + '_df_xcorr_MEGEEG_sensor.pkl')


## Xcorr between audio and each vertice in MEG for each individual
xcorr_all_v_s = []
lag_all_v_s = []

for s in np.arange(0,len(FFR_ba),1):
    print('Now starting sub' + str(s))
    for v in np.arange(0,np.shape(FFR_ba)[1],1):
        b_ba = (FFR_ba[s,v,ts:te] - np.mean(FFR_ba[s,v,ts:te]))/np.std(FFR_ba[s,v,ts:te])     
        b_ba = b_ba / np.linalg.norm(b_ba)
        b_mba = (FFR_mba[s,v,ts:te] - np.mean(FFR_mba[s,v,ts:te]))/np.std(FFR_mba[s,v,ts:te])     
        b_mba = b_mba / np.linalg.norm(b_mba)
        b_pa = (FFR_pa[s,v,ts:te] - np.mean(FFR_pa[s,v,ts:te]))/np.std(FFR_pa[s,v,ts:te])     
        b_pa = b_pa / np.linalg.norm(b_pa)

        xcorr_ba = signal.correlate(a_ba,b_ba)
        xcorr_mba = signal.correlate(a_mba,b_mba)
        xcorr_pa = signal.correlate(a_pa,b_pa)
        
        xcorr_all_v_s.append([s,v,max(abs(xcorr_ba)),lags_time[np.argmax(abs(xcorr_ba))],max(abs(xcorr_mba)),
                            lags_time[np.argmax(abs(xcorr_mba))], max(abs(xcorr_pa)),lags_time[np.argmax(abs(xcorr_pa))]])
df_v_s = pd.DataFrame(columns = ["Subject","Vertno", "abs XCorr MEG & ba", "max Lag MEG & ba", 
                                   "abs XCorr MEG & mba", "max Lag MEG & mba", 
                                   "abs XCorr MEG & pa", "max Lag MEG & pa"], data = xcorr_all_v_s)
if ROI_wholebrain == 'ROI':
    df_v_s.to_pickle(root_path + 'cbsA_meeg_analysis/' + filename + '_df_xcorr_MEGEEG_roi_s.pkl')
elif ROI_wholebrain == 'wholebrain': 
    df_v_s.to_pickle(root_path + 'cbsA_meeg_analysis/' + filename + '_df_xcorr_MEGEEG_v_s.pkl')
elif ROI_wholebrain == 'sensor':
    df_v_s.to_pickle(root_path + 'cbsA_meeg_analysis/' + filename + '_df_xcorr_MEGEEG_sensor_s.pkl')

#%%####################################### quick pca test between the two methods
X = MEG_pa_FFR
X0 = np.reshape(X[0,:,:],[1,306,1101])
pca_m1 = UnsupervisedSpatialFilter(PCA(60))
pca_data_m1 = pca_m1.fit_transform(X0)
pca_data_m1 = np.squeeze(pca_data_m1)

pca_m2 = PCA(60)
pca_data_m2 = pca_m2.fit_transform(X[0,:,:].transpose()) # combine the fit and transform functions

#%%####################################### apply dimension reduction on the sensor, source level
## UnsupervisedSpatialFilter
X = MEG_pa_FFR
pca = UnsupervisedSpatialFilter(PCA(60))
pca_data = pca.fit_transform(X)
ica = UnsupervisedSpatialFilter(FastICA(30, whiten="unit-variance"), average=False)
ica_data = ica.fit_transform(X)

## PCA direct function
# Transpose first sample x feature
X = MEG_pa_FFR.mean(0).transpose()
pca = PCA(60)
pca.fit(X) # combine the fit and transform functions
pca_data = pca.fit_transform(X) # combine the fit and transform functions

## Inverse transform with selected PCs
# 1. zero out the unwanted PCs then use inverse_transform function
X_pcainv = pca.inverse_transform(pca_data)
# 2. perform by in-house script to select the wanted PCs
ind_components = [0,6,12]
Xhat = np.dot(pca.transform(X)[:,ind_components], pca.components_[ind_components,:])
Xhat += np.mean(X.mean(0), axis=0)        

## prove that Xhat and X_pcainv are the same
plt.figure()
plt.plot(X.mean(0)[0,:])
plt.plot(X_pcainv[0,:])
plt.plot(Xhat[0,:])

pca.explained_variance_ratio_.cumsum() # should be close to 1
plt.figure()
plt.plot(pca.explained_variance_ratio_)

plt.figure()
plt.subplot(211)
plot_err(EEG_ba_FFR,'k',stc1.times)
plt.title('ba')
plt.xlim([-0.02,0.2])
plt.subplot(212)
plot_err(pca_data[:,0,:],'k',stc1.times)
plt.xlim([-0.02,0.2])

plt.figure()
plt.subplot(211)
plot_err(EEG_ba_FFR,'k',stc1.times)
plt.title('ba')
plt.xlim([-0.02,0.2])
plt.subplot(212)
plot_err(ica_data[:,0,:],'k',stc1.times)
plt.xlim([-0.02,0.2])

#%%####################################### Spectrum analysis
tmin = 0
tmax = 0.13
fmin = 50
fmax = 150

subject = 'fsaverage'
src = mne.read_source_spaces(subjects_dir + subject + '/bem/fsaverage-vol-5-src.fif')
fname_aseg = subjects_dir + subject + '/mri/aparc+aseg.mgz'
label_names = np.asarray(mne.get_volume_labels_from_aseg(fname_aseg))

## For audio
psds, freqs = mne.time_frequency.psd_array_welch(
    ba_audio,fs, # could replace with label time series
    n_fft=len(ba_audio),
    n_overlap=0,
    n_per_seg=None,
    fmin=fmin,
    fmax=fmax,)
plt.plot(freqs,psds)

## For one subject
epochs = mne.read_epochs(root_path + 'cbs_A101/sss_fif/cbs_A101_01_otp_raw_sss_proj_f_cABR_e.fif')
evoked = mne.read_evokeds(root_path + 'cbs_A101/sss_fif/cbs_A101_01_otp_raw_sss_proj_f_evoked_substd_cabr.fif')[0]
sfreq = epochs.info["sfreq"]
evoked.compute_psd("welch",
   n_fft=int(sfreq * (tmax - tmin)),
   n_overlap=0,
   n_per_seg=None,
   tmin=tmin,
   tmax=tmax,
   fmin=fmin,
   fmax=fmax,
   window="boxcar",
   verbose=False,).plot(average=False,picks="data", exclude="bads")

## For group results
# EEG
psds, freqs = mne.time_frequency.psd_array_welch(
    EEG_pa_FFR.mean(0),sfreq, # could replace with label time series
    n_fft=int(sfreq * (tmax - tmin)),
    n_overlap=0,
    n_per_seg=None,
    fmin=fmin,
    fmax=fmax,)

# MEG sensor
psds, freqs = mne.time_frequency.psd_array_welch(
    MEG_ba_FFR.mean(0),sfreq, # could replace with label time series
    n_fft=int(sfreq * (tmax - tmin)),
    n_overlap=0,
    n_per_seg=None,
    fmin=fmin,
    fmax=fmax,)
evoked.data = MEG_ba_FFR.mean(0)
evk_spectrum = evoked.compute_psd("welch",
   n_fft=int(sfreq * (tmax - tmin)),
   n_overlap=0,
   n_per_seg=None,
   tmin=tmin,
   tmax=tmax,
   fmin=fmin,
   fmax=fmax,
   window="boxcar",
   verbose=False,)

evk_spectrum.plot_topo(color="k", fig_facecolor="w", axis_facecolor="w")

# MEG sensor PCA
psds, freqs = mne.time_frequency.psd_array_welch(
    pca_data.transpose(),sfreq, # could replace with label time series
    n_fft=int(sfreq * (tmax - tmin)),
    n_overlap=0,
    n_per_seg=None,
    fmin=fmin,
    fmax=fmax,)

# MEG source
psds, freqs = mne.time_frequency.psd_array_welch(
    MEG_pa_FFR.mean(0)[43],sfreq, # could replace with label time series
    n_fft=np.shape(MEG_pa_FFR)[2],
    n_overlap=0,
    n_per_seg=None,
    fmin=fmin,
    fmax=fmax,)
        
plt.figure()
plt.plot(freqs,psds.transpose())
