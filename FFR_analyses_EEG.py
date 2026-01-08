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

@author: tzcheng
"""

#%%####################################### Import library  
import mne
import os
import math
import matplotlib.pyplot as plt
from scipy import stats,signal
from scipy.io import savemat, loadmat
import numpy as np
from numpy import dot
from numpy.linalg import norm
from scipy.stats import pearsonr
import scipy as sp
import pandas as pd
import scipy.stats as stats
from scipy.io import wavfile
import time
import copy
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

from sklearn.decomposition import PCA, FastICA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
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
            signal = np.load(root_path + 'cbsA_meeg_analysis/MEG/FFR/ntrial_200/group_' + name + '_pcffr80200_3_200_' + file_type + '.npy')
        elif subject_type == 'infants':   
            signal = np.load(root_path + 'cbsb_meg_analysis/MEG/FFR/ntrial_200/group_' + name + '_pcffr80200_3_200_' + file_type + '.npy')
    return fs, signal

def load_brainstem_file(file_type, ntrial):
    root_path = '/media/tzcheng/storage/Brainstem/'
    fs = 5000
    if file_type == 'EEG':
        p10_eng = np.load(root_path + 'EEG/p10_eng_eeg_ntr' + ntrial + '_01.npy')
        n40_eng = np.load(root_path + 'EEG/n40_eng_eeg_ntr' + ntrial + '_01.npy')
        p10_spa = np.load(root_path + 'EEG/p10_spa_eeg_ntr' + ntrial + '_01.npy')
        n40_spa = np.load(root_path + 'EEG/n40_spa_eeg_ntr' + ntrial + '_01.npy')
    ## not yet implemented MEG load elif file_type in ('sensor', 'roi','morph'):
    return fs, p10_eng, n40_eng, p10_spa, n40_spa
    
def do_CBS_trial_by_trial_decoding(root_path,n_trials):
    all_score_svm = []
    subj = [] # A104 got some technical issue
    for file in os.listdir():
        if file.startswith('cbs_A'): # cbs_A for the adults and cbs_b for the infants
            subj.append(file)
    for s in subj:
        file_in = root_path + '/' + s + '/eeg/' + s
        ##%% extract the FFR time series
        epochs = mne.read_epochs(file_in +'_01_cABR_e_' + str(n_trials) + '.fif')
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
        
        X = np.squeeze(epochs.get_data())  
        y = epochs.events[:, 2]  # target: standard, deviant1 and 2
        
        mdic = {"condition":y,"data":X}
        fname = s +'_epoch'
        savemat(fname + '.mat', mdic)
        del mdic, epochs
        
        ## Three way classification using ovr SVM
        clf = make_pipeline(
            StandardScaler(),
            SVC(kernel='rbf',gamma='auto')  
        )
        scores = cross_val_multiscore(clf, X, y, cv=5, n_jobs=None) # takes about 10 mins to run
        score = np.mean(scores, axis=0)
        print("Trial-by-trial decoding accuracy: %0.1f%%" % (100 * score,))
        all_score_svm.append(score)
    all_score_svm = np.asarray(all_score_svm)
    return all_score_svm
        
def do_brainstem_trial_by_trial_decoding(root_path,n_trials,decoding_type):
    ## Trial-by-trial decoding for spa vs. eng: not working well
    subjects_eng=['104','106','107','108','110','111','112','113','118','121','123','124','126','129','133']
    subjects_spa=['203','204','205','206','211','212','213','214','215','220','221','222','223','224','225'] # trim 226 so be n = 15 too
    # subjects_spa=['203','204','205','206','211','212','213','214','215','220','221','222','223','224','225','226'] 

    data_eng = []
    data_spa = []
        
    all_score_svm = []

    for se, sp in zip(subjects_eng,subjects_spa):
        epochs_eng = mne.read_epochs(root_path + 'preprocessed/ntrial_all/eng/brainstem_' + se  +'_p10_01_cabr_e_all.fif').pick_types(eeg=True, exclude=[])
        epochs_spa = mne.read_epochs(root_path + 'preprocessed/ntrial_all/spa/brainstem_' + sp  +'_p10_01_cabr_e_all.fif').pick_types(eeg=True, exclude=[])

        random.seed(15)
        rand_ind = random.sample(range(min(len(epochs_eng['44'].events),len(epochs_eng['88'].events))),n_trials//2) 
        epochs_eng = mne.concatenate_epochs([epochs_eng['44'][rand_ind],epochs_eng['88'][rand_ind]])
        rand_ind = random.sample(range(min(len(epochs_spa['44'].events),len(epochs_spa['88'].events))),n_trials//2) 
        epochs_spa = mne.concatenate_epochs([epochs_spa['44'][rand_ind],epochs_spa['88'][rand_ind]])
            
        data_eng.append(epochs_eng.get_data())
        data_spa.append(epochs_spa.get_data())  

    data_eng = np.squeeze(data_eng)
    data_spa = np.squeeze(data_spa)

    nt = np.shape(data_eng)[-1]
    X_eng = data_eng.reshape(-1,nt)
    X_spa = data_spa.reshape(-1,nt)

    X = np.concatenate((X_eng,X_spa),axis=0)
    y = np.concatenate((np.repeat(0,len(X_eng)),np.repeat(1,len(X_spa))))
                       
    ## SVM showed higher accuracy in trial-by-trial decoding
    clf = make_pipeline(
        StandardScaler(),
        SVC(kernel='rbf',gamma='auto')  
       )

    scores = cross_val_multiscore(clf, X, y, cv=5, n_jobs=4) 
    score = np.mean(scores, axis=0)
    print("Trial-by-trial decoding accuracy: %0.1f%%" % (100 * score,))
    all_score_svm.append(score)
    return all_score_svm
    
def do_subject_by_subject_decoding(X_list,times,ts,te,ncv,random_state=None):
    """
    X_list : list of ndarrays
        One array per condition/category
    """
    ## classifier
    clf = make_pipeline(
        StandardScaler(),  # z-score normalization
        SVC(kernel='rbf',gamma='auto',C=0.1,class_weight="balanced")  
        # SVC(kernel='linear', C=1)
    )
    tslice = slice(ff(times, ts), ff(times, te))
    X = []
    y = []
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

    ## see the weights 
    # from sklearn.model_selection import StratifiedKFold
    # cv = StratifiedKFold(n_splits=ncv, shuffle=False)

    # weights = []
    # scores = []

    # for train_idx, test_idx in cv.split(X, y):
    #     scaler = StandardScaler()
    #     X_train = scaler.fit_transform(X[train_idx])
    #     X_test  = scaler.transform(X[test_idx])
        
    #     clf = SVC(kernel='linear', C=1)
    #     clf.fit(X[train_idx], y[train_idx])

    #     scores.append(clf.score(X[test_idx], y[test_idx]))
    #     weights.append(clf.coef_.squeeze())


    # weights = np.array(weights)   # shape: (n_folds, n_timepoints)
    # plt.figure()
    # plt.plot(np.linspace(-0.02,0.2,1101),np.mean(weights,axis=0))
    # plt.xlim(-0.02,0.2)
    # plt.ylim(-7e-6, 7e-6)
    # plt.title('SVM weights across time')

# def do_SNR:

# def do_xcorr:

def plot_individuals(data_dict,n_cols,t):
    
    """
    data_dict = {
        'subj1': y1,
        'subj2': y2,
        ...
    }
    """
    n_subj = len(data_dict)
    n_rows = math.ceil(n_subj / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3 * n_rows),
        sharex=True, sharey=True
    )

    axes = axes.flatten()

    for ax, (subj, y) in zip(axes, data_dict.items()):
        ax.plot(t,y)
        ax.set_title(subj)

    # remove empty subplots
    for ax in axes[len(data_dict):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def plot_group_ffr(p10, n40, group_name,
                   time,
                   ylim=(-5e-7, 5e-7),
                   n_times=1101):
    """
    Plot mean FFR responses and differential response for one group.
    """
    
    # Mean responses
    plt.figure()
    plt.title(f'{group_name} speakers')
    plt.plot(t, p10.mean(0), label='p10')
    plt.plot(t, n40.mean(0), label='n40')
    plt.xlim(tmin, tmax)
    plt.ylim(*ylim)
    plt.legend()

    # Differential response
    plt.figure()
    plt.title(f'{group_name} speakers differential response (p10 − n40)')
    plt.plot(t, p10.mean(0) - n40.mean(0))
    plt.xlim(tmin, tmax)
    plt.ylim(*ylim)

def plot_decoding_histograms(scores_p10,
    scores_n40,
    bins=10,
    chance=0.5,
    labels=("p10", "n40"),
    xlim=(0, 1)):
    
    fig, ax = plt.subplots(1)

    ax.hist(scores_p10, bins=bins, alpha=0.6)
    ax.hist(scores_n40, bins=bins, alpha=0.6)

    ax.set_ylabel("Count", fontsize=20)
    ax.set_xlabel("Accuracy", fontsize=20)
    ax.set_xlim(*xlim)

    ax.legend(labels)

    # chance line
    ax.axvline(chance, color="grey", linestyle="--")

    # mean lines
    ax.axvline(np.mean(scores_p10), color="skyblue", linewidth=2)
    ax.axvline(np.mean(scores_n40), color="orange", linewidth=2)
    return fig, ax

#%%####################################### Set path
subjects_dir = '/media/tzcheng/storage2/subjects/'
times = np.linspace(-20,200,1101)

#%%####################################### load the data
file_type = 'EEG'
subject_type = 'adults'
fs,std = load_CBS_file(file_type, 'p10', subject_type)
fs,dev1 = load_CBS_file(file_type, 'n40', subject_type)
fs,dev2 = load_CBS_file(file_type, 'p40', subject_type)
    
## brainstem
ntrial = 'all'
fs, p10_eng, n40_eng, p10_spa, n40_spa = load_brainstem_file(file_type, ntrial)
    
#%%####################################### visualize the data to examine
## plot individual FFRs
subjects_eng=['104','106','107','108','110','111','112','113','118','121','123','124','126','129','133']
subjects_spa=['203','204','205','206','211','212','213','214','215','220','221','222','223','224','225','226'] ## 202 event code has some issues
t = np.linspace(-0.02, 0.2, 1101)

subjects_eng_dict = dict(zip(subjects_eng, p10_eng))
subjects_spa_dict = dict(zip(subjects_spa, p10_spa))
n_cols = 3
plot_individuals(subjects_eng_dict,n_cols,t)
plot_individuals(subjects_spa_dict,n_cols,t)

## plot average FFRs between p10 vs. n40
plot_group_ffr(p10_eng, n40_eng, 'English', t)
plot_group_ffr(p10_spa, n40_spa, 'Spanish', t)

## plot average FFRs between spa and eng
plot_group_ffr(p10_eng, p10_spa, 'p10')
plot_group_ffr(n40_eng, n40_spa, 'n40')

#%%####################################### trial-by-trial EEG decoding for each individual of brainstem dataset
root_path='/media/tzcheng/storage2/CBS/'
trial_by_trial_decoding_acc = do_CBS_trial_by_trial_decoding(root_path,n_trials=200)

root_path= '/media/tzcheng/storage/Brainstem/EEG/'
comparison = 'Eng/Spa' ## 'Eng/Spa' or 'p10/n40'
trial_by_trial_decoding_acc = do_brainstem_trial_by_trial_decoding(root_path,n_trials=200)

#%%####################################### Subject-by-subject EEG decoding brainstem dataset
## Run with one random seed 2
t = np.linspace(-0.02, 0.2, 1101)
ts = 0
te = 90
## modify ts, te to do C and V section decoding [0, 40] ba C section
## for ba, 10 ms + 90 ms = 100 ms; for mba, 40 ms + 90 ms = 130 ms
## epoch length -20 to 200 ms with sampling rate at 5000 Hz
ncv = 15
randseed = 2
decoding_acc = do_subject_by_subject_decoding([p10_eng, n40_eng], t, ts, te, ncv, randseed)
decoding_acc = do_subject_by_subject_decoding([p10_spa, n40_spa], t, ts, te, ncv, randseed)
decoding_acc = do_subject_by_subject_decoding([p10_eng, p10_spa], t, ts, te, ncv, randseed)
decoding_acc = do_subject_by_subject_decoding([n40_eng, n40_spa], t, ts, te, ncv, randseed)

#%%# Run with iterative random seeds
niter = 1000 # see how the random seed affects accuracy
scores_p10 = []
scores_n40 = []

## randomize the spanish speakers to use
rng = np.random.default_rng(1)
perm = rng.permutation(len(p10_spa))

for n_iter in np.arange(0,niter,1):
    print("iter " + str(n_iter))
    ## decode eng vs. spa speakers: keep both n = 15 vs. n1 = 15, n2 = 16 gave very higher than chance results
    decoding_acc_p10 = do_subject_by_subject_decoding([p10_eng, p10_spa[perm,:][:-1,:]], t, ts, te, ncv, None)
    scores_p10.append(np.mean(decoding_acc_p10, axis=0))
    decoding_acc_n40 = do_subject_by_subject_decoding([n40_eng, n40_spa[perm,:][:-1,:]], t, ts, te, ncv, None)
    scores_n40.append(np.mean(decoding_acc_n40, axis=0))
scores_p10 = np.array(scores_p10)
scores_n40 = np.array(scores_n40)
print(f"Accuracy: {np.mean(scores_p10):.3f}")
print(f"Accuracy: {np.mean(scores_n40):.3f}")
plot_decoding_histograms(scores_p10,scores_n40,bins=10,chance=0.5,labels=("p10", "n40"),xlim=(0, 1))

#%%####################################### Subject-by-subject EEG decoding CBS dataset
## change ts and te for C and V section decoding: for ba, 10 ms + 90 ms = 100 ms; for mba and pa, 40 ms + 90 ms = 130 ms
## epoch length -20 to 200 ms with sampling rate at 5000 Hz
## Use C section to decode mba and pa
ncv = len(std)
####################################### 3-way decoding: ba vs. pa vs. mba
scores_ba_mba_pa = do_subject_by_subject_decoding([std,dev1,dev2], t, ts, te, ncv, randseed)

####################################### 2-way decoding: ba vs. pa, ba vs. mba, pa vs. mba
scores_ba_mba = do_subject_by_subject_decoding([std,dev1], t, ts, te, ncv, randseed)
score_ba_mba = np.mean(scores_ba_mba, axis=0)
print("Decoding Accuracy between ba vs. mba: %0.1f%%" % (100 * score_ba_mba,))

scores_ba_pa = do_subject_by_subject_decoding([std,dev2], t, ts, te, ncv, randseed)
score_ba_pa = np.mean(scores_ba_mba, axis=0)
print("Decoding Accuracy between ba vs. pa: %0.1f%%" % (100 * score_ba_pa,))

scores_mba_pa = do_subject_by_subject_decoding([dev1,dev2], t, ts, te, ncv, randseed)
score_mba_pa = np.mean(scores_mba_pa, axis=0)
print("Decoding Accuracy between mba vs. pa: %0.1f%%" % (100 * score_mba_pa,))

#%%####################################### decoding acoustic signals from the misc, can add noise
root_path='/media/tzcheng/storage2/CBS/cbsA_meeg_analysis/misc/'
ts = ff(times,0) ## very important to set the window not including the final artifacts
te = ff(times,150) ## very important to set the window not including the final artifacts

std = np.load(root_path + 'adult_group_substd_misc_200.npy')[:,ts:te]
dev1 = np.load(root_path + 'adult_group_dev1_misc_200.npy')[:,ts:te]
dev2 = np.load(root_path + 'adult_group_dev2_misc_200.npy')[:,ts:te]

## the codes below are identical to the ones used in eeg
## classifier
clf = make_pipeline(
    StandardScaler(),  # z-score normalization
    SVC(kernel='rbf',gamma='auto',C=0.1)  
)

####################################### 3-way decoding: ba vs. pa vs. mba
y = np.concatenate((np.repeat(0,len(std)),np.repeat(1,len(dev1)),np.repeat(2,len(dev2))))

## preserve the subject ba, mba, pa relationship but randomize the order across subjects
rand_ind = np.arange(0,len(std))
random.Random(2).shuffle(rand_ind)
X = np.concatenate((std[rand_ind,:],dev1[rand_ind,:],dev2[rand_ind,:]),axis=0)

scores = cross_val_multiscore(clf, X, y, cv=18, n_jobs=None) # takes about 10 mins to run
score = np.mean(scores, axis=0)
print("Accuracy: %0.1f%%" % (100 * score,))

####################################### 2-way decoding: ba vs. pa, ba vs. mba, pa vs. mba
y = np.concatenate((np.repeat(0,len(std)),np.repeat(1,len(dev1))))
rand_ind = np.arange(0,len(std))
random.Random(2).shuffle(rand_ind)
X = np.concatenate((std[rand_ind,:],dev1[rand_ind,:]),axis=0)

scores_ba_mba = cross_val_multiscore(clf, X, y, cv=18, n_jobs=None) # takes about 10 mins to run
score_ba_mba = np.mean(scores_ba_mba, axis=0)
print("Decoding Accuracy between ba vs. mba: %0.1f%%" % (100 * score_ba_mba,))

rand_ind = np.arange(0,len(std))
random.Random(2).shuffle(rand_ind)
X = np.concatenate((std[rand_ind,:],dev2[rand_ind,:]),axis=0)

scores_ba_pa = cross_val_multiscore(clf, X, y, cv=18, n_jobs=None) # takes about 10 mins to run
score_ba_pa = np.mean(scores_ba_pa, axis=0)
print("Decoding Accuracy between ba vs. pa: %0.1f%%" % (100 * score_ba_pa,))

rand_ind = np.arange(0,len(std))
random.Random(2).shuffle(rand_ind)
X = np.concatenate((dev1[rand_ind,:],dev2[rand_ind,:]),axis=0)

scores_mba_pa = cross_val_multiscore(clf, X, y, cv=18, n_jobs=None) # takes about 10 mins to run
score_mba_pa = np.mean(scores_mba_pa, axis=0)
print("Decoding Accuracy between mba vs. pa: %0.1f%%" % (100 * score_mba_pa,))

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

#%%####################################### SNR analysis
EEG = EEG_pa_FFR
ind_noise = np.where(times<0)
ind_signal = np.where(np.logical_and(times>=0, times<=0.13)) # 0.1 for ba and 0.13 for mba and pa

## Group
rms_noise_s = np.sqrt(np.mean(EEG.mean(0)[ind_noise]**2))
rms_signal_s = np.sqrt(np.mean(EEG.mean(0)[ind_signal]**2))
SNR = rms_signal_s/rms_noise_s

## Individual
rms_noise_s = []
rms_signal_s = []

for s in range(len(EEG_pa_FFR)):
    rms_noise_s.append(np.sqrt(np.mean(EEG[s,ind_noise]**2)))
    rms_signal_s.append(np.sqrt(np.mean(EEG[s,ind_signal]**2)))
SNR = np.array(rms_signal_s)/np.array(rms_noise_s)
print('SNR: ' + str(np.array(SNR).mean()) + '(' + str(np.array(SNR).std()/np.sqrt(len(EEG_pa_FFR))) +')')

#%%####################################### xcorr analysis
root_path='/home/tzcheng/Documents/GitHub/Paper0_Paradigm/'

## Load FFR from 0
std = np.load(root_path + 'group_std_ffr_eeg_200.npy')[:,100:]
dev1 = np.load(root_path + 'group_dev1_ffr_eeg_200.npy')[:,100:]
dev2 = np.load(root_path + 'group_dev2_ffr_eeg_200.npy')[:,100:]

## second run
std = np.load(root_path + 'group_02_std_ffr_eeg_150.npy')[:,100:] # use 150 or all trials
dev1 = np.load(root_path + 'group_02_dev1_ffr_eeg_150.npy')[:,100:]
dev2 = np.load(root_path + 'group_02_dev2_ffr_eeg_150.npy')[:,100:]

# plt.figure()
# plt.plot(np.linspace(0,0.13,650),dev2_audio_r)

## Load real audio
fs, std_audio = wavfile.read(root_path + '+10.wav')
fs, dev1_audio = wavfile.read(root_path + '-40.wav')
fs, dev2_audio = wavfile.read(root_path + '+40.wav')
# Downsample
fs_new = 5000
num_std = int((len(std_audio)*fs_new)/fs)
num_dev = int((len(dev2_audio)*fs_new)/fs)  # #sample_new/fs_new=#sample/fs find number of samples in the resampled data
            
std_audio = signal.resample(std_audio, num_std, t=None, axis=0, window=None)
dev1_audio = signal.resample(dev1_audio, num_dev, t=None, axis=0, window=None)
dev2_audio = signal.resample(dev2_audio, num_dev, t=None, axis=0, window=None)

## Change audio0 and EEG0 to corresponding std, dev1, dev2
# Run the corresponding code section below
audio0 = std_audio
EEG0 = std
times0_audio = np.linspace(0,len(audio0)/fs_new,len(audio0))
times0_eeg = np.linspace(0,len(EEG0[0])/fs_new,len(EEG0[0]))

## std: noise burst from 0 ms 
ts = 100 # 0.02s (i.e. 0.02s after noise burst)
te = 500 # 0.1s
audio = audio0[ts:te] # try 0.02 to 0.1 s for std
EEG = EEG0[:,ts:te]
times_audio = times0_audio[ts:te]
times_eeg = times0_eeg[ts:te]
lags = signal.correlation_lags(len(audio),len(EEG[0]))
lags_s = lags/fs_new

## dev 1: noise burst from 40 ms (200th points)
ts = 200 + 100 # .06s (i.e. 0.02s after noise burst)
te = 650 # 0.13s
audio = audio0[ts:te] # try 0.042 to 0.13 s for dev2
EEG = EEG0[:,ts:te]
times_audio = times0_audio[ts:te]
times_eeg = times0_eeg[ts:te]
lags = signal.correlation_lags(len(audio),len(EEG[0]))
lags_s = lags/fs_new

## dev 2: noise burst from 0 ms (100th points)
ts = 100 # 0.02s
te = 650 # 0.13s
audio = audio0[ts:te] # try 0.02 to 0.13 s for dev2
EEG = EEG0[:,ts:te]
times_audio = times0_audio[ts:te]
times_eeg = times0_eeg[ts:te]
lags = signal.correlation_lags(len(audio),len(EEG[0]))
lags_s = lags/fs_new

## select the lag window to be between 7 to 14 ms
lag_window = [-0.007,-0.014]
lag_window_ind = np.where((lags_s<=lag_window[0]) & (lags_s>=lag_window[1]))

a = (audio - np.mean(audio))/np.std(audio)
a = a / np.linalg.norm(a)

## For grand average: a[n] is lagging behind b[n] by k sample periods
b = (EEG.mean(axis=0) - np.mean(EEG.mean(axis=0)))/np.std(EEG.mean(axis=0))
b = b / np.linalg.norm(b)
xcorr = signal.correlate(a,b,mode='full')
xcorr = abs(xcorr)
# xcorr_max = max(xcorr)
# xcorr_maxlag = np.argmax(xcorr)
# print("max lag: ", str(lags_s[xcorr_maxlag]*1000))

xcorr_max = max(xcorr[lag_window_ind]) # only select the xcorr from the time window shift
xcorr_maxlag = np.argmax(xcorr[lag_window_ind])
print("max lag: ", str(lags_s[lag_window_ind][xcorr_maxlag]*1000)) # only select the xcorr from the time window shift
print("max xcorr: ", str(xcorr_max))

## For each individual
xcorr_all_s = []
xcorr_lag_all_s = []

for s in np.arange(0,len(std),1):
    b = (EEG[s,:] - np.mean(EEG[s,:]))/np.std(EEG[s,:])
    
    ## the way matlab do xcorr normalization: the max is 1 if do a zero lag autocorrealtoin
    b = b / np.linalg.norm(b)

    xcorr = signal.correlate(a,b,mode='full')
    xcorr = abs(xcorr)
    xcorr_all_s.append(np.max(xcorr[lag_window_ind]))
    xcorr_lag_all_s.append(np.argmax(xcorr[lag_window_ind]))
    # xcorr_all_s.append(np.max(xcorr))
    # xcorr_lag_all_s.append(np.argmax(xcorr))

# print('abs xcorr between FFR & audio: ' + str(np.array(xcorr_all_s).mean()) + '(' + str(np.array(xcorr_all_s).std()/np.sqrt(len(std))) +')')
# print('abs xcorr lag between FFR & audio (ms): ' + str(np.array(lags_s[xcorr_lag_all_s]*1000).mean())+ '(' + str(np.array(lags_s[xcorr_lag_all_s]*1000).std()/np.sqrt(len(std))) +')')
# arr = np.array(xcorr_all_s)
# rounded_arr = np.around(arr, decimals=3)
# print(rounded_arr)
# print(lags_s[xcorr_lag_all_s]*1000)

print('abs xcorr between FFR & audio: ' + str(np.array(xcorr_all_s).mean()) + '(' + str(np.array(xcorr_all_s).std()/np.sqrt(len(std))) +')')
print('abs xcorr lag between FFR & audio (ms): ' + str(np.array(lags_s[lag_window_ind][xcorr_lag_all_s]*1000).mean())+ '(' + str(np.array(lags_s[lag_window_ind][xcorr_lag_all_s]*1000).std()/np.sqrt(len(std))) +')')
arr = np.array(xcorr_all_s)
rounded_arr = np.around(arr, decimals=3)
print(rounded_arr)
print(lags_s[lag_window_ind][xcorr_lag_all_s]*1000)

#%%####################################### spectrogram analysis
import librosa
import librosa.display

x = n40_eng.mean(0)
x = x.astype(np.float32)
x = x/np.max(np.abs(x))

fs = 5000

nfft = 128
S = librosa.stft(
    x,
    n_fft = nfft,
    hop_length = nfft//16,
    win_length = nfft,
    window = 'hann')
S_db = librosa.amplitude_to_db(np.abs(S),ref=np.max)

## Plot the waveform
fig, ax = plt.subplots(figsize=(10, 4))
librosa.display.waveshow(x,sr=fs,ax=ax, color = 'purple')
ax.set(title='Audio Waveform (Time Series)')
ax.set_xlabel("Time")
ax.set_ylabel("Amplitude")

## Plot the spectrogram
fig, ax = plt.subplots(figsize=(10, 4))
im = librosa.display.specshow(
    S_db,
    sr=fs,
    hop_length=nfft//16,
    x_axis='time',
    y_axis='hz',
    cmap='magma',
    ax=ax,
    vmin=-20,
    vmax=0
    
)
ax.set_title('Spectrogram')
ax.set_ylim([0,800])
fig.colorbar(im, ax=ax, format="%+2.0f dB")
plt.show()


#%%####################################### visualize the data to examine
plt.figure()
plt.title('p10 response')
plt.plot(np.linspace(-0.02,0.2,1101),p10_eng.mean(0))
plt.plot(np.linspace(-0.02,0.2,1101),p10_spa.mean(0))
plt.xlim(-0.02,0.2)
plt.ylim(-5e-7, 5e-7)
plt.legend(['Eng','Spa'])

plt.figure()
plt.title('English vs. Spanish speakers differential p10 response')
plt.plot(np.linspace(-0.02,0.2,1101),p10_eng.mean(0)-p10_spa.mean(0))
plt.xlim(-0.02,0.2)
plt.ylim(-5e-7, 5e-7)

plt.figure()
plt.title('n40 response')
plt.plot(np.linspace(-0.02,0.2,1101),n40_eng.mean(0))
plt.plot(np.linspace(-0.02,0.2,1101),n40_spa.mean(0))
plt.xlim(-0.02,0.2)
plt.ylim(-5e-7, 5e-7)
plt.legend(['Eng','Spa'])

plt.figure()
plt.title('English vs. Spanish speakers differential n40 response')
plt.plot(np.linspace(-0.02,0.2,1101),n40_eng.mean(0)-n40_spa.mean(0))
plt.xlim(-0.02,0.2)
plt.ylim(-5e-7, 5e-7)

