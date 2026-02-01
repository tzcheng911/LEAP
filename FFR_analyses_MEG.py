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

#%%####################################### Import library
import mne
import os
import math
import matplotlib.pyplot as plt
from scipy import stats, signal
from scipy.signal import butter, filtfilt, hilbert
from scipy.io import savemat
import numpy as np
from scipy.io import wavfile
import time
from mne.decoding import (
    cross_val_multiscore,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import random

#%%####################################### Define functions
def plot_err(group_stc,color,t):
    group_avg=np.mean(group_stc,axis=0)
   #plt.figure()
    err=np.std(group_stc,axis=0)/np.sqrt(group_stc.shape[0])
    up=group_avg+err
    lw=group_avg-err
    plt.plot(t,group_avg,color=color)
    plt.fill_between(t,up,lw,color=color,alpha=0.5)

def ff(input_arr,target): 
    ## find the idx of the closest freqeuncy (time) in freqs (times)
    delta = 1000000000
    idx = -1
    for i, val in enumerate(input_arr):
        if abs(input_arr[i]-target) < delta:
            idx = i
            delta = abs(input_arr[i]-target)
    return idx

def load_CBS_file(file_type, sound_type, subject_type):
    ## load and plot the time series of 'p10','n40','p40' for 'audio', 'eeg', 'sensor', 'ROI', 'morph' in "infants" or "adults"
    root_path = '/media/tzcheng/storage2/CBS/'
    fs = 5000
    # map sound names when needed
    sound_map = {
        'p10': 'ba',
        'p40': 'pa',
        'n40': 'mba'
    }
    if file_type == 'audio':
        fs, signal = wavfile.read(root_path + '/stimuli/' + sound_type + '.wav')
    elif file_type == 'EEG':
        signal = np.load(root_path + 'cbsA_meeg_analysis/EEG/group_' + sound_type + '_ffr_eeg_200.npy')
    elif file_type in ('sensor', 'morph_roi','morph'): ## for the MEG
        name = sound_map.get(sound_type, sound_type)
        if subject_type == 'adults':
            signal = np.load(root_path + 'cbsA_meeg_analysis/MEG/FFR/ntrial_200/group_' + name + '_pcffr80450_3_200_' + file_type + '.npy')
        elif subject_type == 'infants':   
            signal = np.load(root_path + 'cbsb_meg_analysis/MEG/FFR/ntrial_200/group_' + name + '_pcffr80450_3_200_' + file_type + '.npy')
    return fs, signal

def load_brainstem_file(file_type, ntrial):
    root_path = '/media/tzcheng/storage/Brainstem/'
    fs = 5000
    if file_type == 'EEG':
        p10_eng = np.load(root_path + 'EEG/p10_eng_eeg_ntr' + ntrial + '_01.npy')
        n40_eng = np.load(root_path + 'EEG/n40_eng_eeg_ntr' + ntrial + '_01.npy')
        p10_spa = np.load(root_path + 'EEG/p10_spa_eeg_ntr' + ntrial + '_01.npy')
        n40_spa = np.load(root_path + 'EEG/n40_spa_eeg_ntr' + ntrial + '_01.npy')
        return fs, p10_eng, n40_eng, p10_spa, n40_spa
    elif file_type in ('sensor', 'morph_roi','morph'): ## for the MEG
        p10_eng = np.load(root_path + 'cbsA_meeg_analysis/MEG/FFR/ntrial_200/group_pcffr80450_3_p10_01_' + file_type + '.npy')
        n40_eng = np.load(root_path + 'cbsA_meeg_analysis/MEG/FFR/ntrial_200/group_pcffr80450_3_n40_01_' + file_type + '.npy')
        return fs, p10_eng, n40_eng
    
def do_subject_by_subject_decoding(X_list,times,ts,te,ncv,shuffle,random_state):
    """
    X_list : list of ndarrays
        One array per condition/category
    """
    ## classifier
    clf = make_pipeline(
        StandardScaler(),  # z-score normalization
        SVC(kernel='rbf',gamma='auto',C=0.1,class_weight='balanced')  
        # SVC(kernel='linear', C=1)
    )
    tslice = slice(ff(times, ts), ff(times, te))
    X = []
    y = []
    
    ## shuffle the order across participants but keep the pair
    if shuffle == "keep pair":
        rng = np.random.default_rng(random_state)
        perm = rng.permutation(len(X_list[0]))
        for label, Xi in enumerate(X_list):
            X.append(Xi[perm, tslice])
            y.append(np.full(len(Xi), label))
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y)
    
    ## shuffle the order across participants fully
    elif shuffle == "full":
        for label, Xi in enumerate(X_list):
            X.append(Xi[:, tslice])
            y.append(np.full(len(Xi), label))
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y)
        rng = np.random.default_rng(random_state)
        perm = rng.permutation(len(y))
        X = X[perm]
        y = y[perm]
   
    scores = cross_val_multiscore(clf, X, y, cv=ncv, n_jobs=None)
    score = np.mean(scores, axis=0)
    print("Decoding Accuracy %0.1f%%" % (100 * score))
    return scores

def plot_group_ffr(data1, data2, label1,label2,
                   times,
                   ylim=(-5e-7, 5e-7),
                   n_times=1101):
    """
    Plot mean FFR responses and differential response for one group.
    """
    
    # Mean responses
    plt.figure()
    plt.plot(times, data1.mean(0), label=label1)
    plt.plot(times, data2.mean(0), label=label2)
    plt.xlim(np.min(times), np.max(times))
    plt.ylim(*ylim)
    plt.legend()

    # Differential response
    plt.figure()
    plt.title('Differential response')
    plt.plot(times, data1.mean(0) - data2.mean(0))
    plt.xlim(np.min(times), np.max(times))
    plt.ylim(*ylim)
    
#%%####################################### Set path   
root_path='/media/tzcheng/storage2/CBS/'
subjects_dir = '/media/tzcheng/storage2/subjects/'
stc1 = mne.read_source_estimate(root_path + 'cbs_A101/sss_fif/cbs_A101_pa_cabr_morph-vl.stc')
src = mne.read_source_spaces(subjects_dir + 'fsaverage/bem/fsaverage-vol-5-src.fif')
fname_aseg = subjects_dir + 'fsaverage/mri/aparc+aseg.mgz'
label_names = np.asarray(mne.get_volume_labels_from_aseg(fname_aseg))
times = np.linspace(-20,200,1101)

#%%####################################### load the data
file_type = 'morph_roi'
subject_type = 'adults'
fs,std = load_CBS_file(file_type, 'p10', subject_type)
fs,dev1 = load_CBS_file(file_type, 'n40', subject_type)
fs,dev2 = load_CBS_file(file_type, 'p40', subject_type)
    
## brainstem
file_type = 'EEG'
ntrial = 'all'
fs, p10_eng, n40_eng, p10_spa, n40_spa = load_brainstem_file(file_type, ntrial)
fs, p10_eng, n40_eng = load_brainstem_file(file_type, ntrial)

#%%####################################### Subject-by-subject MEG decoding for each condition 
#%%# permuation test to compare p10/n40 vs. p10/p40 decoding in CBS eng

## reduce dimension
# get the mean
std = std.mean(axis=1)
dev1 = dev1.mean(axis=1)
dev2 = dev2.mean(axis=1)

# get the mag mean because it is more sensitive to the deep source
epochs = mne.read_epochs(root_path + 'cbs_A101/sss_fif/cbs_A101_01_otp_raw_sss_proj_f_cABR_e.fif')
indices_by_type = mne.channel_indices_by_type(epochs.info)
std = std[:,indices_by_type['mag'],:]
dev1 = dev1[:,indices_by_type['mag'],:]
dev2 = dev2[:,indices_by_type['mag'],:]

# ROI
lh_ROI_label = [12, 72,76,74] # [subcortical] brainstem,[AC] STG, transversetemporal, [controls] frontal pole
rh_ROI_label = [12, 108,112,110] # [subcortical] brainstem,[AC] STG, transversetemporal, [controls] frontal pole
nROI = 2
std = std[:,rh_ROI_label[nROI],:] 
dev1 = dev1[:,rh_ROI_label[nROI],:] 
dev2 = dev2[:,rh_ROI_label[nROI],:]

## run decoding
ts = 0
te = 0.20
niter = 1000 # see how the random seed affects accuracy
shuffle = "keep pair"
randseed = 2
ncv = len(std)

## real difference between eng and spa decoding accuracy of p10 vs. n40 sounds
decoding_acc_p10n40 = do_subject_by_subject_decoding([std, dev1], times, ts, te, ncv, shuffle, randseed)
decoding_acc_p10p40 = do_subject_by_subject_decoding([std, dev2], times, ts, te, ncv, shuffle, randseed)
diff_acc = np.mean(decoding_acc_p10n40, axis=0) - np.mean(decoding_acc_p10p40, axis=0)

## permute between dev1 and dev2
diff_scores_perm = []
n40p40_all = np.vstack([dev1,dev2])
n_total = len(n40p40_all)
rng = np.random.default_rng(None)
 
for n_iter in np.arange(0,niter,1):
    print("iter " + str(n_iter))
    
    perm_ind = rng.permutation(n_total)
    group1_ind = perm_ind[:n_total//2]
    group2_ind = perm_ind[n_total//2:]
    dev1_perm = n40p40_all[group1_ind]
    dev2_perm = n40p40_all[group2_ind]
    
    decoding_acc_group1_perm = do_subject_by_subject_decoding([std, dev1_perm], times, ts, te, ncv, shuffle, randseed)
    decoding_acc_group2_perm = do_subject_by_subject_decoding([std, dev2_perm], times, ts, te, ncv, shuffle, randseed)

    diff_scores_perm.append(np.mean(decoding_acc_group1_perm, axis=0) - np.mean(decoding_acc_group2_perm, axis=0))
diff_scores_perm = np.array(diff_scores_perm)
print(f"Accuracy: {np.mean(diff_scores_perm):.3f}")

fig, ax = plt.subplots(1)
ax.hist(diff_scores_perm, bins=7, alpha=0.6)
ax.set_ylabel("Count", fontsize=20)
ax.set_xlabel("Accuracy Difference", fontsize=20)

# chance line
ax.axvline(np.mean(diff_scores_perm), color="grey", linestyle="--")
# 95% line
ax.axvline(np.percentile(diff_scores_perm,95),ymin=0,ymax=1000,color='grey',linewidth=2)
# mean lines
ax.axvline(diff_acc, color="red", linewidth=2)

#%%####################################### Sliding estimator decoding brainstem eng speakers
root_path='/media/tzcheng/storage/Brainstem/'
subjects_dir = '/media/tzcheng/storage2/subjects/'
os.chdir(root_path)
stc1 = mne.read_source_estimate(root_path + 'brainstem_113/sss_fif/brainstem_113_pcffr80200_3_n40_01_morph-vl.stc')
times = stc1.times

## parameter
ROI_wholebrain = 'wholebrain' # ROI or wholebrain or sensor
k_feature = 'all' # ROI: 'all' features; whole brain: 500 features

fname_aseg = subjects_dir + 'fsaverage/mri/aparc+aseg.mgz'
label_names = np.asarray(mne.get_volume_labels_from_aseg(fname_aseg))
lh_ROI_label = [12, 72,76,74] # [subcortical] brainstem,[AC] STG, transversetemporal, [controls] frontal pole
rh_ROI_label = [12, 108,112,110] # [subcortical] brainstem,[AC] STG, transversetemporal, [controls] frontal pole

if ROI_wholebrain == 'ROI':
    ffr_ba = np.load(root_path + 'MEG/FFR/group_pcffr80200_3_p10_01_roi.npy',allow_pickle=True)
    ffr_mba = np.load(root_path + 'MEG/FFR/group_pcffr80200_3_n40_01_roi.npy',allow_pickle=True)
elif ROI_wholebrain == 'wholebrain':
    ffr_ba = np.load(root_path + 'MEG/FFR/group_pcffr80200_3_p10_01_morph.npy',allow_pickle=True)
    ffr_mba = np.load(root_path + 'MEG/FFR/group_pcffr80200_3_n40_01_morph.npy',allow_pickle=True)
else:
    print("Need to decide whether to use ROI or whole brain as feature.")

X = np.concatenate((ffr_ba,ffr_mba),axis=0)
# X = X[:,:,ts:te] 
y = np.concatenate((np.repeat(0,len(ffr_ba)),np.repeat(1,len(ffr_ba)))) #0 is for mmr1 and 1 is for mmr2

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
TOI = np.linspace(-20,200,num=1101)
fig, ax = plt.subplots(1)
ax.plot(TOI, scores_observed.mean(0), label="score")
ax.axhline(1/3, color="k", linestyle="--", label="chance")
ax.axvline(0, color="k")
plt.legend()

# The fitting needs not be cross validated because the weights are based on
# the training sets
time_decod.fit(X, y) # not changed after shuffling the initial
# Retrieve patterns after inversing the z-score normalization step:
patterns = get_coef(time_decod, "patterns_", inverse_transform=True)

#%%####################################### Sliding estimator decoding CBS
tic = time.time()
root_path='/media/tzcheng/storage2/CBS/'
subjects_dir = '/media/tzcheng/storage2/subjects/'
os.chdir(root_path)
stc1 = mne.read_source_estimate(root_path + 'cbs_A101/sss_fif/cbs_A101_ba_cabr_morph-vl.stc')
times = stc1.times
n_top = '3' # could number of IC: 3 or 10, or dss: dss_f80450, dss
n_trial = '200' # 'ntrial_200/' or 'ntrial_all/' or ''

## parameter
ROI_wholebrain = 'wholebrain' # ROI or wholebrain or sensor
k_feature = 'all' # ROI: 'all' features; whole brain: 500 features

filename = 'pcffr80200'
filename_ffr_ba = 'group_ba_pcffr80200_'
filename_ffr_mba = 'group_mba_pcffr80200_'
filename_ffr_pa = 'group_pa_pcffr80200_'

fname_aseg = subjects_dir + 'fsaverage/mri/aparc+aseg.mgz'
label_names = np.asarray(mne.get_volume_labels_from_aseg(fname_aseg))
lh_ROI_label = [12, 72,76,74] # [subcortical] brainstem,[AC] STG, transversetemporal, [controls] frontal pole
rh_ROI_label = [12, 108,112,110] # [subcortical] brainstem,[AC] STG, transversetemporal, [controls] frontal pole

if ROI_wholebrain == 'ROI':
    ffr_ba = np.load(root_path + 'cbsA_meeg_analysis/MEG/FFR/ntrial_' + str(n_trial)+ '/' + filename_ffr_ba + str(n_top) + '_' + str(n_trial) + '_roi.npy',allow_pickle=True)
    ffr_mba = np.load(root_path + 'cbsA_meeg_analysis/MEG/FFR/ntrial_' + str(n_trial)+ '/' + filename_ffr_mba + str(n_top) + '_' + str(n_trial) + '_roi.npy',allow_pickle=True)
    ffr_pa = np.load(root_path + 'cbsA_meeg_analysis/MEG/FFR/ntrial_' + str(n_trial)+ '/' + filename_ffr_pa + str(n_top) + '_' + str(n_trial) + '_roi.npy',allow_pickle=True)
elif ROI_wholebrain == 'wholebrain':
    ffr_ba = np.load(root_path + 'cbsA_meeg_analysis/MEG/FFR/ntrial_' + str(n_trial)+ '/' + filename_ffr_ba + str(n_top) + '_' + str(n_trial) + '_morph.npy',allow_pickle=True)
    ffr_mba = np.load(root_path + 'cbsA_meeg_analysis/MEG/FFR/ntrial_' + str(n_trial)+ '/' + filename_ffr_mba + str(n_top) + '_' + str(n_trial) + '_morph.npy',allow_pickle=True)
    ffr_pa = np.load(root_path + 'cbsA_meeg_analysis/MEG/FFR/ntrial_' + str(n_trial)+ '/' + filename_ffr_pa + str(n_top) + '_' + str(n_trial) + '_morph.npy',allow_pickle=True)
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
TOI = np.linspace(-20,200,num=1101)
fig, ax = plt.subplots(1)
ax.plot(TOI, scores_observed.mean(0), label="score")
ax.axhline(1/3, color="k", linestyle="--", label="chance")
ax.axvline(0, color="k")
plt.legend()

# The fitting needs not be cross validated because the weights are based on
# the training sets
time_decod.fit(X, y) # not changed after shuffling the initial
# Retrieve patterns after inversing the z-score normalization step:
patterns = get_coef(time_decod, "patterns_", inverse_transform=True)

toc = time.time()

np.save(root_path + 'cbsA_meeg_analysis/decoding/roc_auc_kall_' + filename + '.npy',scores_observed)
np.save(root_path + 'cbsA_meeg_analysis/decoding/patterns_kall_' + filename + '.npy',patterns)

#%%####################################### MEG decoding CBS across time
root_path='/media/tzcheng/storage2/CBS/'
subjects_dir = '/media/tzcheng/storage2/subjects/'
os.chdir(root_path)

n_top = '3' # could number of IC: 3 or 10, or dss: dss_f80450, dss
n_trial = '200' # 'ntrial_200/' or 'ntrial_all/' or ''
stc1 = mne.read_source_estimate(root_path + 'cbs_A101/sss_fif/cbs_A101_ba_cabr_morph-vl.stc')
# times = stc1.times
did_pca = '_pcffr80200_'  # '_': without or with pca "_pcffr80450_" the filter between 80 and 450 Hz is applied
filename_ffr_ba = 'group_ba' + did_pca
filename_ffr_mba = 'group_mba' + did_pca
filename_ffr_pa = 'group_pa' + did_pca

fname_aseg = subjects_dir + 'fsaverage/mri/aparc+aseg.mgz'
label_names = np.asarray(mne.get_volume_labels_from_aseg(fname_aseg))

## FFR relevant ROIs
lh_ROI_label = [12, 72,76,74] # [subcortical] brainstem,[AC] STG, transversetemporal, [controls] frontal pole
rh_ROI_label = [12, 108,112,110] # [subcortical] brainstem,[AC] STG, transversetemporal, [controls] frontal pole

baby_or_adult = 'cbsA_meeg_analysis' # baby or adult
input_data = 'ROI' # ROI or wholebrain or sensor or pcffr
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

####################################### 3-way decoding: ba vs. pa vs. mba
all_score = []
## Three way classification using ovr
X = np.concatenate((ffr_ba,ffr_mba,ffr_pa),axis=0)
# X = np.concatenate((ffr_ba[:,:,ff(times,40):ff(times,130)],ffr_mba[:,:,ff(times,40):ff(times,130)],ffr_pa[:,:,ff(times,40):ff(times,130)]),axis=0) # use just the V section
y = np.concatenate((np.repeat(0,len(ffr_ba)),np.repeat(1,len(ffr_mba)),np.repeat(2,len(ffr_pa))))

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
# np.save(root_path + baby_or_adult +'/decoding/'+ str(n_top) + '_ntrial_' + str(n_trial) + '_3way_decoding_accuracy_V_section_' + input_data +'_r15.npy',all_score)
np.save(root_path + baby_or_adult +'/decoding/'+ str(n_top) + '_ntrial_' + str(n_trial) + '_3way_decoding_accuracy_' + input_data +'_r15.npy',all_score)

####################################### 2-way decoding: ba vs. pa, ba vs. mba, pa vs. mba
clf = make_pipeline(
    StandardScaler(),  # z-score normalization
    LogisticRegression(solver="liblinear")  # liblinear is faster than lbfgs
)  

all_score_ba_mba = []
X = np.concatenate((ffr_ba,ffr_mba),axis=0)
# X = np.concatenate((ffr_ba[:,:,ff(times,10):ff(times,100)],ffr_mba[:,:,ff(times,40):ff(times,130)]),axis=0)

y = np.concatenate((np.repeat(0,len(ffr_ba)),np.repeat(1,len(ffr_mba))))

rand_ind = np.arange(0,len(X))
random.Random(15).shuffle(rand_ind)

X = X[rand_ind,:,:]
y = y[rand_ind]
for n in np.arange(0,np.shape(X)[1],1):
        scores = cross_val_multiscore(clf, X[:,n,:], y, cv=5, n_jobs=4) # takes about 10 mins to run
        score = np.mean(scores, axis=0)
        print("Data " + str(n+1) + " Accuracy: %0.1f%%" % (100 * score,))
        all_score_ba_mba.append(score)
np.save(root_path + baby_or_adult +'/decoding/'+ str(n_top) + '_ntrial_' + str(n_trial) + '_ba_mba_decoding_accuracy_' + input_data +'_r15.npy',all_score_ba_mba)

all_score_ba_pa = []
X = []
y = []
X = np.concatenate((ffr_ba,ffr_pa),axis=0)
y = np.concatenate((np.repeat(0,len(ffr_ba)),np.repeat(1,len(ffr_pa))))
# X = np.concatenate((ffr_ba[:,:,ff(times,10):ff(times,100)],ffr_pa[:,:,ff(times,40):ff(times,130)]),axis=0)

X = X[rand_ind,:,:]
y = y[rand_ind]
for n in np.arange(0,np.shape(X)[1],1):
        scores = cross_val_multiscore(clf, X[:,n,:], y, cv=5, n_jobs=4) # takes about 10 mins to run
        score = np.mean(scores, axis=0)
        print("Data " + str(n+1) + " Accuracy: %0.1f%%" % (100 * score,))
        all_score_ba_pa.append(score)
np.save(root_path + baby_or_adult +'/decoding/'+ str(n_top) + '_ntrial_' + str(n_trial) + '_ba_pa_decoding_accuracy_' + input_data +'_r15.npy',all_score_ba_pa)

all_score_mba_pa = []
X = []
y = []
X = np.concatenate((ffr_mba,ffr_pa),axis=0)
y = np.concatenate((np.repeat(0,len(ffr_mba)),np.repeat(1,len(ffr_pa))))
X = np.concatenate((ffr_mba[:,:,ff(times,40):ff(times,130)],ffr_pa[:,:,ff(times,40):ff(times,130)]),axis=0)


X = X[rand_ind,:,:]
y = y[rand_ind]
for n in np.arange(0,np.shape(X)[1],1):
        scores = cross_val_multiscore(clf, X[:,n,:], y, cv=5, n_jobs=4) # takes about 10 mins to run
        score = np.mean(scores, axis=0)
        print("Data " + str(n+1) + " Accuracy: %0.1f%%" % (100 * score,))
        all_score_mba_pa.append(score)
np.save(root_path + baby_or_adult +'/decoding/'+ str(n_top) + '_ntrial_' + str(n_trial) + '_mba_pa_decoding_accuracy_' + input_data +'_r15.npy',all_score_mba_pa)

## C and V section decoding: for ba, 10 ms + 90 ms = 100 ms; for mba and pa, 40 ms + 90 ms = 130 ms
X1 = X[rand_ind,:,:np.shape(X)[-1]//2]
X2 = X[rand_ind,:,np.shape(X)[-1]//2:]

#%%####################################### check acc for each sensor, ROI or vertice
acc_ind = np.where(np.array(all_score) >= 1/3)

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

## list the hot spots
label_v_ind = np.load('/media/tzcheng/storage/scripts_zoe/ROI_lookup.npy', allow_pickle=True)
high_acc = np.where(np.array(all_score) > 0.5)
label_names = mne.get_volume_labels_from_aseg('/media/tzcheng/storage2/subjects/fsaverage/mri/aparc+aseg.mgz')
high_acc = np.array(high_acc[0])
ROIs = []
for i in np.arange(0,len(high_acc),1):
    for nlabel in np.arange(0,len(label_names),1):
        if high_acc[i] in label_v_ind[nlabel][0] and label_names[nlabel] not in ROIs:
            ROIs.append(label_names[nlabel])
print(ROIs)

#%%####################################### MEG decoding brainstem dataset across time
root_path='/media/tzcheng/storage/Brainstem/' # brainstem files
subjects_dir = '/media/tzcheng/storage2/subjects/'
os.chdir(root_path)

n_top = '3' # could number of IC: 3 or 10, or dss: dss_f80450, dss
stc1 = mne.read_source_estimate(root_path + 'brainstem_133/sss_fif/brainstem_133_pcffr80200_3_p10_01_morph-vl.stc')
# times = stc1.times
did_pca = '_pcffr80200_'  # '_': without or with pca "_pcffr80450_" the filter between 80 and 450 Hz is applied

fname_aseg = subjects_dir + 'fsaverage/mri/aparc+aseg.mgz'
label_names = np.asarray(mne.get_volume_labels_from_aseg(fname_aseg))

## FFR relevant ROIs
lh_ROI_label = [12, 72,76,74] # [subcortical] brainstem,[AC] STG, transversetemporal, [controls] frontal pole
rh_ROI_label = [12, 108,112,110] # [subcortical] brainstem,[AC] STG, transversetemporal, [controls] frontal pole

input_data = 'ROI' # ROI or wholebrain or sensor or pcffr
k_feature = 'all' # ROI: 'all' features; whole brain: 500 features

if input_data == 'sensor':
    ffr_ba = np.load(root_path + '/MEG/FFR/group_f80200_p10_01_sensor.npy',allow_pickle=True)
    ffr_mba = np.load(root_path + '/MEG/FFR/group_f80200_n40_01_sensor.npy',allow_pickle=True)
elif input_data == 'ROI':
    ffr_ba = np.load(root_path + '/MEG/FFR/group_f80200_p10_01_roi.npy',allow_pickle=True)
    ffr_mba = np.load(root_path + '/MEG/FFR/group_f80200_n40_01_roi.npy',allow_pickle=True)
elif input_data == 'wholebrain':
    ffr_ba = np.load(root_path + '/MEG/FFR/group_f80200_p10_01_morph.npy',allow_pickle=True)
    ffr_mba = np.load(root_path + '/MEG/FFR/group_f80200_n40_01_morph.npy',allow_pickle=True)
else:
    print("Need to decide whether to use ROI or whole brain as feature.")

all_score = []
X = np.concatenate((ffr_ba,ffr_mba),axis=0)
# X = np.concatenate((ffr_ba[:,:,ff(times,40):ff(times,130)],ffr_mba[:,:,ff(times,40):ff(times,130)]),axis=0) # use just the V section
y = np.concatenate((np.repeat(0,len(ffr_ba)),np.repeat(1,len(ffr_mba))))

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

# np.save(root_path +'/MEG/FFR/'+ str(n_top) + '_3way_decoding_accuracy_V_section_' + input_data +'_r15.npy',all_score)
np.save(root_path +'/MEG/FFR/'+ str(n_top) + '_decoding_accuracy_' + input_data +'_r15.npy',all_score)

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
audio = pa_audio
psds, freqs = mne.time_frequency.psd_array_welch(
    audio,fs, # could replace with label time series
    n_fft=len(audio),
    n_overlap=0,
    n_per_seg=None,
    fmin=fmin,
    fmax=fmax,)
plt.figure()
plt.title('pa audio spectrum')
plt.plot(freqs,psds)
plt.xlim([60, 140])

## For each individual
# MEG
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

# EEG
EEG = EEG_pa_FFR
psds, freqs = mne.time_frequency.psd_array_welch(
    EEG,sfreq, # could replace with label time series
    n_fft=len(EEG[1,:]),
    n_overlap=0,
    n_per_seg=None,
    fmin=fmin,
    fmax=fmax,)
plt.figure()
plt.title('ba spectrum')
plot_err(psds,'k',freqs)
plt.xlim([60, 140])

## For group results
# EEG
psds, freqs = mne.time_frequency.psd_array_welch(
    EEG_ba_FFR.mean(0),sfreq, # could replace with label time series
    n_fft=int(sfreq * (tmax - tmin)),
    n_overlap=0,
    n_per_seg=None,
    fmin=fmin,
    fmax=fmax,)
plt.figure()
plt.plot(freqs,psds)

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

#%%####################################### save the mat files of MEG sensor data for dss
random.seed(15)
root_path='/media/tzcheng/storage2/CBS/'
os.chdir(root_path)
n_trials = 200
all_score_lr = []
all_score_svm = []
subj = [] # A104 got some technical issue
for file in os.listdir():
    if file.startswith('cbs_A108'): # cbs_A for the adults and cbs_b for the infants
        subj.append(file)
for s in subj:
    file_in = root_path + '/' + s + '/sss_fif/' + s   # load meg files
    epochs = mne.read_epochs(file_in +'_01_otp_raw_sss_proj_f80450_ffr_e_all.fif')  # load meg files
    
    # file_in = root_path + '/' + s + '/eeg/' + s # load eeg files
    # epochs = mne.read_epochs(file_in +'_01_cABR_e_all.fif')   # load eeg files
    if n_trials == 'all':
        epochs = epochs
    elif n_trials == 200:
        ##%% extract the FFR time series
        # 01_otp_raw_sss_proj_f80450_ffr_e: filter between 80-450; 01_otp_raw_sss_proj_f_ffr_e: filter between 80-2000 
        rand_ind = random.sample(range(min(len(epochs['Standardp'].events),len(epochs['Standardn'].events))),n_trials//2) 
        std_e = mne.concatenate_epochs([epochs['Standardp'][rand_ind],epochs['Standardn'][rand_ind]])
        std_e = mne.epochs.combine_event_ids(std_e, ['Standardp', 'Standardn'], {'Standard': 8})
        rand_ind = random.sample(range(min(len(epochs['Deviant1p'].events),len(epochs['Deviant1n'].events))),n_trials//2) 
        dev1_e = mne.concatenate_epochs([epochs['Deviant1p'][rand_ind],epochs['Deviant1n'][rand_ind]])
        dev1_e = mne.epochs.combine_event_ids(dev1_e, ['Deviant1p', 'Deviant1n'], {'Deviant1': 9})
        rand_ind = random.sample(range(min(len(epochs['Deviant2p'].events),len(epochs['Deviant2n'].events))),n_trials//2) 
        dev2_e = mne.concatenate_epochs([epochs['Deviant2p'][rand_ind],epochs['Deviant2n'][rand_ind]])
        dev2_e = mne.epochs.combine_event_ids(dev2_e, ['Deviant2p', 'Deviant2n'], {'Deviant2': 10})
        epochs = mne.concatenate_epochs([std_e,dev1_e,dev2_e])
    
    X = np.squeeze(epochs.get_data(picks='mag'))  ## only use the 102 mag sensors because catching deeper sources
    y = epochs.events[:, 2]  # target: standard, deviant1 and 2
    
    mdic = {"condition":y,"data":X}
    fname = root_path + 'mat/MEG_f80450/ntrial_200/dss_input/' + s +'_MEG_epoch_f80450_' + str(n_trials)
    savemat(fname + '.mat', mdic)
    del mdic, epochs

#%%####################################### analyze dss files
root_path='/media/tzcheng/storage2/CBS/'
os.chdir(root_path)
n_trials = 200
nch = 75

epochs = mne.read_epochs(root_path + 'cbs_A101/sss_fif/cbs_A101_01_otp_raw_sss_proj_f_cABR_e.fif')
evoked1 = mne.read_evokeds(root_path + 'cbs_A101/sss_fif/cbs_A101_01_otp_raw_sss_proj_f_evoked_substd_cabr.fif')[0]
evoked2 = mne.read_evokeds(root_path + 'cbs_A101/sss_fif/cbs_A101_01_otp_raw_sss_proj_f_evoked_substd_cabr.fif')[0]
epochs = epochs.pick_types('mag')
evoked1 = evoked1.pick_types('mag')
evoked2 = evoked2.pick_types('mag')

subj = [] 
for file in os.listdir():
    if file.startswith('cbs_A108'): # cbs_A for the adults and cbs_b for the infants
        subj.append(file)
for s in subj:
    meg = loadmat(root_path + 'mat/MEG_f80450/ntrial_200/dss_input/' + s +'_MEG_epoch_f80450_200.mat')
    meg = meg['data']
    dss_clean_meg = loadmat(root_path + 'mat/MEG_f80450/ntrial_200/dss_output/ba/clean_ba_' + s +'_MEG_epoch_f80450_200.mat')
    dss_clean_meg = dss_clean_meg['megclean2']
    
    evoked1.data = meg[:200,:,:].mean(0)
    evoked2.data = dss_clean_meg[:200,:,:].mean(0)
    evoked1.plot_topo()
    evoked2.plot_topo()
    
    fig, ax = plt.subplots(1,1)
    im = plt.imshow(meg[:200,:,:].mean(axis=1),aspect = 'auto', origin='lower', cmap='jet')
    plt.colorbar()
    im.set_clim(-2e-14,2e-14)
    
    fig, ax = plt.subplots(1,1)
    im = plt.imshow(dss_clean_meg[:,:,:].mean(axis=1),aspect = 'auto', origin='lower', cmap='jet')
    plt.colorbar()
    im.set_clim(-2e-22,2e-22)