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
    ## load and plot the time series of 'p10','n40','p40' for 'audio','misc', 'eeg', 'sensor', 'ROI', 'morph' in "infants" or "adults"
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
    elif file_type == 'misc':
        signal = np.load(root_path + 'cbsA_meeg_analysis/misc/adult_group_' + sound_type + '_misc_200.npy')
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
        # SVC(kernel='linear', C=1,class_weight='balanced')
       )

    scores = cross_val_multiscore(clf, X, y, cv=5, n_jobs=4) 
    score = np.mean(scores, axis=0)
    print("Trial-by-trial decoding accuracy: %0.1f%%" % (100 * score,))
    all_score_svm.append(score)
    return all_score_svm
    
def do_subject_by_subject_decoding(X_list,times,ts,te,ncv,shuffle,random_state):
    """
    X_list : list of ndarrays
        One array per condition/category
    """
    ## classifier
    clf = make_pipeline(
        StandardScaler(),  # z-score normalization
        # SVC(kernel='rbf',gamma='auto',C=0.1,class_weight='balanced')  
        SVC(kernel='linear', C=1,class_weight='balanced')
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
    
    ## using leave one out vs. leave one group out with stratified and matching subjects order give same results
    # from sklearn.model_selection import LeaveOneGroupOut
    # groups = np.concatenate((np.arange(0,18,1),np.arange(0,18,1)))
    # logo = LeaveOneGroupOut()
    # logo.get_n_splits(groups=groups)
    # for i, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
    #     print(f"Fold {i}:")
    #     print(f"  Train: index={train_idx}, group={groups[train_idx]}")
    #     print(f"  Test:  index={test_idx}, group={groups[test_idx]}")
    #     scaler = StandardScaler()
    #     X_train = scaler.fit_transform(X[train_idx])
    #     X_test  = scaler.transform(X[test_idx])
        
    #     clf = SVC(kernel='linear', C=1)
    #     clf.fit(X_train, y[train_idx])

    #     scores.append(clf.score(X_test, y[test_idx]))    
    # score = np.mean(scores, axis=0)
    # print("Decoding Accuracy %0.1f%%" % (100 * score))
    
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
    #     clf.fit(X_train, y[train_idx])

    #     scores.append(clf.score(X_test, y[test_idx]))
    #     weights.append(clf.coef_.squeeze())

    # score = np.mean(scores, axis=0)
    # print("Decoding Accuracy %0.1f%%" % (100 * score))
    # weights = np.array(weights)   # shape: (n_folds, n_timepoints)
    # plt.figure()
    # plt.plot(np.linspace(-0.02,0.2,1101),np.mean(weights,axis=0))
    # plt.xlim(-0.02,0.2)
    # plt.ylim(-7e-6, 7e-6)
    # plt.title('SVM weights across time')

def bandpass_filter(data, fs, fmin, fmax, order=4):
    nyq = fs / 2
    b, a = butter(
        order,
        [fmin / nyq, fmax / nyq],
        btype="band"
    )
    return filtfilt(b, a, data, axis=-1)

def itpc_hilbert(
    data,
    fs,
    times,ts,te,
    fmin,
    fmax,
    axis=0
):
    """
    Compute ITPC using Hilbert transform.

    Parameters
    ----------
    data : ndarray
        Shape (trials, times) or (trials, channels, times)
    fs : float
        Sampling rate (Hz)
    fmin, fmax : float
        Band-pass frequencies (Hz)
    axis : int
        Trial axis (default=0)

    Returns
    -------
    itpc : ndarray
        ITPC over time (and channels, if present)
    """
    # 0. crop the data
    tslice = slice(ff(times, ts), ff(times, te))
    data = data[:,tslice]
    
    # 1. Band-pass filter
    data_filt = bandpass_filter(data, fs, fmin, fmax)

    # 2. Hilbert transform
    analytic = hilbert(data_filt, axis=-1)

    # 3. Extract phase
    phase = np.angle(analytic)

    # 4. Unit phase vectors
    phase_vec = np.exp(1j * phase)

    # 5. Average across trials
    itpc = np.abs(np.mean(phase_vec, axis=axis))

    return itpc

def do_SNR(data,times,ts,te,fmin, fmax, level):
    sfreq = 5000
    total_len = len(times)

    ind_noise = np.where(times<ts)
    ind_signal = np.where(np.logical_and(times>=ts, times<=te)) 
    if level == 'group':
        ## temporal SNR
        rms_noise_s = np.sqrt(np.mean(data.mean(0)[ind_noise]**2))
        rms_signal_s = np.sqrt(np.mean(data.mean(0)[ind_signal]**2))
        SNR_t = rms_signal_s/rms_noise_s
        
        ## spectral SNR
        psds_noise, freqs_noise = mne.time_frequency.psd_array_welch(
        data.mean(0)[ind_noise],sfreq, # could replace with label time series
        n_fft=total_len,
        n_overlap=0,
        n_per_seg=total_len,
        fmin=fmin,
        fmax=fmax,)
    
        psds_signal, freqs_signal = mne.time_frequency.psd_array_welch(
        data.mean(0)[ind_signal],sfreq, # could replace with label time series
        n_fft=total_len,
        n_overlap=0,
        n_per_seg=total_len,
        fmin=fmin,
        fmax=fmax,)
        SNR_s = psds_signal[ff(freqs_signal,91)]/psds_noise[ff(freqs_noise,91)] # find the closest peak to the audio ~90 Hz
        SNR_s = np.max(psds_signal)/np.max(psds_noise)
        
        plt.figure()
        plt.plot(freqs_signal,psds_signal)
        plt.plot(freqs_signal,psds_noise)
    elif level == 'individual':
        ## temporal SNR
        rms_noise_s = []
        rms_signal_s = []

        for s in range(len(data)):
            rms_noise_s.append(np.sqrt(np.mean(data[s,ind_noise]**2)))
            rms_signal_s.append(np.sqrt(np.mean(data[s,ind_signal]**2)))
        SNR_t = np.array(rms_signal_s)/np.array(rms_noise_s)
        
        ## spectral SNR
        psds_noise, freqs_noise = mne.time_frequency.psd_array_welch(
        np.squeeze(data[:,ind_noise]),sfreq, # could replace with label time series
        n_fft=total_len,
        n_overlap=0,
        n_per_seg=total_len,
        fmin=fmin,
        fmax=fmax,)
        
        psds_signal, freqs_signal = mne.time_frequency.psd_array_welch(
        np.squeeze(data[:,ind_signal]),sfreq, # could replace with label time series
        n_fft=total_len,
        n_overlap=0,
        n_per_seg=total_len,
        fmin=fmin,
        fmax=fmax,)
        
        plt.figure()
        plt.plot(freqs_signal,np.mean(psds_signal,axis=0))
        plt.plot(freqs_signal,np.mean(psds_noise,axis=0))
        
        SNR_s = psds_signal[:,ff(freqs_signal,91)]/psds_noise[:,ff(freqs_noise,91)] # find the closest peak to the audio ~90 Hz
        SNR_s = np.max(psds_signal,axis=0)/np.max(psds_noise,axis=0)

    return SNR_t, SNR_s

def zscore_and_normalize(x):
    x = (x - np.mean(x)) / np.std(x)
    return x / np.linalg.norm(x)

def do_xcorr(audio,EEG,fs_audio,fs_eeg,ts,te,level,lag_window_s=(-0.014, -0.007)):
    # Downsample
    num = int((len(audio)*fs_eeg)/fs_audio)    
    audio_rs = signal.resample(audio, num, t=None, axis=0, window=None)
    times_audio = np.linspace(0,len(audio_rs)/fs_eeg,len(audio_rs))
    times_eeg = np.linspace(-0.02,0.2,1101)

    ## p10: noise burst from 0 ms (100th points) ts = 100 # 0.02s (i.e. 0.02s after noise burst) te = 500 # 0.1s
    ## n40: noise burst from 40 ms (200th points) ts = 200 + 100 # 0.06s (i.e. 0.02s after noise burst) te = 650 # 0.13s
    ## p40: noise burst from 0 ms (100th points) ts = 100 # 0.02s te = 650 # 0.13s
    tslice_audio = slice(ff(times_audio, ts), ff(times_audio, te)+1) ## +1 hack to make sure slice in audio and eeg are the same length
    tslice_EEG = slice(ff(times_eeg, ts), ff(times_eeg, te))

    stim = audio_rs[tslice_audio]
    resp = EEG[:,tslice_EEG]
    lags = signal.correlation_lags(len(stim),resp.shape[1])
    lags_s = lags/fs_eeg

    lag_min, lag_max = np.array(lag_window_s)
    lag_mask = (lags_s >= lag_min) & (lags_s <= lag_max)

    stim_z = zscore_and_normalize(stim)

    ## For grand average: a[n] is lagging behind b[n] by k sample periods
    if level == 'group':
        resp_mean = resp.mean(axis=0)
        resp_mean_z = zscore_and_normalize(resp_mean)
        xcorr = signal.correlate(stim_z,resp_mean_z,mode='full')
        xcorr = abs(xcorr)
        xcorr_win = xcorr[lag_mask]
        lag_win = lags_s[lag_mask]
        max_idx = np.argmax(xcorr_win)

        return {
            "xcorr_max": xcorr_win[max_idx],
            "xcorr_lag_ms": lag_win[max_idx]
        }
    elif level == 'individual':
        xcorr_max = []
        xcorr_lag = []

        for subj_resp in resp:
            resp_z = zscore_and_normalize(subj_resp)
            xcorr = np.abs(signal.correlate(stim_z, resp_z, mode="full"))

            xcorr_win = xcorr[lag_mask]
            lag_win = lags_s[lag_mask]

            idx = np.argmax(xcorr_win)
            xcorr_max.append(xcorr_win[idx])
            xcorr_lag.append(lag_win[idx])

        return {
            "xcorr_max": np.array(xcorr_max),
            "xcorr_lag_ms": np.array(xcorr_lag)
        }

    else:
        raise ValueError("level must be 'group' or 'individual'")

def do_autocorr_sliding(
    eeg,
    fs_eeg,
    times_eeg,
    ts,
    te,
    level,
    lag_window_ms=(-0.05, 0.05),
    win_ms=0.04,
    step_ms=0.02,
):
    """
    Sliding-window EEG autocorrelation.

    Parameters
    ----------
    eeg : ndarray (n_subjects, n_times)
    fs_eeg : float
    times_eeg : ndarray
    ts, te : float
        Time window (in seconds) to analyze
    level : {'group', 'individual'}
    lag_window_ms : tuple
        Lag window in ms (e.g., (-50, 50))
    win_ms : float
        Sliding window length in ms
    step_ms : float
        Step size in ms
    """

    # --------------------
    # Time slice
    # --------------------
    tslice = slice(ff(times_eeg, ts), ff(times_eeg, te))
    resp = eeg[:, tslice]  # (subjects × time)
    n_subjects, n_times = resp.shape

    # --------------------
    # Convert window params to samples
    # --------------------
    win_samp = int(win_ms / 1000 * fs_eeg)
    step_samp = int(step_ms / 1000 * fs_eeg)

    if win_samp >= n_times:
        raise ValueError("Sliding window longer than data segment")

    # --------------------
    # Lags (defined by window length)
    # --------------------
    lags = signal.correlation_lags(win_samp, win_samp)
    lags_s = lags / fs_eeg

    lag_min, lag_max = np.array(lag_window_ms) / 1000
    lag_mask = (lags_s >= lag_min) & (lags_s <= lag_max)

    lags_ms = lags_s[lag_mask] * 1000

    # --------------------
    # Sliding windows
    # --------------------
    win_starts = np.arange(0, n_times - win_samp + 1, step_samp)
    win_centers = times_eeg[tslice][win_starts + win_samp // 2]

    # --------------------
    # Group-level
    # --------------------
    if level == "group":
        resp_mean = resp.mean(axis=0)

        acorr_windows = []

        for ws in win_starts:
            seg = resp_mean[ws : ws + win_samp]
            seg_z = zscore_and_normalize(seg)

            acorr = signal.correlate(seg_z, seg_z, mode="full")
            acorr_windows.append(np.abs(acorr)[lag_mask])

        return {
            "autocorr": np.array(acorr_windows),  # (windows × lags)
            "lags_ms": lags_ms,
            "times": win_centers,
        }

    # --------------------
    # Individual-level
    # --------------------
    elif level == "individual":
        acorr_all = []

        for subj_resp in resp:
            subj_windows = []

            for ws in win_starts:
                seg = subj_resp[ws : ws + win_samp]
                seg_z = zscore_and_normalize(seg)

                acorr = signal.correlate(seg_z, seg_z, mode="full")
                subj_windows.append(np.abs(acorr)[lag_mask])

            acorr_all.append(subj_windows)

        return {
            "autocorr": np.array(acorr_all),  # (subjects × windows × lags)
            "lags_ms": lags_ms,
            "times": win_centers,
        }

    else:
        raise ValueError("level must be 'group' or 'individual'")

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
times = np.linspace(-0.02, 0.2, 1101)

#%%####################################### load the data
file_type = 'EEG'
subject_type = 'adults'
fs,std = load_CBS_file(file_type, 'p10', subject_type)
fs,dev1 = load_CBS_file(file_type, 'n40', subject_type)
fs,dev2 = load_CBS_file(file_type, 'p40', subject_type)
    
## brainstem
file_type = 'EEG'
ntrial = '200'
fs, p10_eng, n40_eng, p10_spa, n40_spa = load_brainstem_file(file_type, ntrial)
    
#%%####################################### visualize the data to examine
## plot individual FFRs
subjects_eng=['104','106','107','108','110','111','112','113','118','121','123','124','126','129','133']
subjects_spa=['203','204','205','206','211','212','213','214','215','220','221','222','223','224','225','226'] ## 202 event code has some issues

subjects_eng_dict = dict(zip(subjects_eng, p10_eng))
subjects_spa_dict = dict(zip(subjects_spa, p10_spa))
n_cols = 3
plot_individuals(subjects_eng_dict,n_cols,times)
plot_individuals(subjects_spa_dict,n_cols,times)

## plot average FFRs between p10 vs. n40
plot_group_ffr(p10_eng, n40_eng, 'p10','n40', times)
plot_group_ffr(p10_spa, n40_spa, 'p10','n40', times)

## plot average FFRs between spa and eng
plot_group_ffr(p10_eng, p10_spa, 'eng','spa', times)
plot_group_ffr(n40_eng, n40_spa, 'eng','spa', times)

#%%####################################### trial-by-trial EEG decoding for each individual of brainstem dataset
root_path='/media/tzcheng/storage2/CBS/'
trial_by_trial_decoding_acc = do_CBS_trial_by_trial_decoding(root_path,n_trials=200)

root_path= '/media/tzcheng/storage/Brainstem/EEG/'
comparison = 'Eng/Spa' ## 'Eng/Spa' or 'p10/n40'
trial_by_trial_decoding_acc = do_brainstem_trial_by_trial_decoding(root_path,n_trials=200)

#%%####################################### Subject-by-subject EEG decoding brainstem dataset
## Run with one random seed 2
ts = 0
te = 0.2
## modify ts, te to do C and V section decoding [0, 40] ba C section
## for ba, 10 ms + 90 ms = 100 ms; for mba, 40 ms + 90 ms = 130 ms
## epoch length -20 to 200 ms with sampling rate at 5000 Hz
ncv = 15
randseed = 2

decoding_acc = do_subject_by_subject_decoding([p10_eng, n40_eng], times, ts, te, len(p10_eng), 'keep pair', randseed)
decoding_acc = do_subject_by_subject_decoding([p10_spa, n40_spa], times, ts, te, len(p10_spa), 'keep pair', randseed)
decoding_acc = do_subject_by_subject_decoding([p10_eng, p10_spa], times, ts, te, len(p10_eng), 'full', randseed)
decoding_acc = do_subject_by_subject_decoding([n40_eng, n40_spa], times, ts, te, len(p10_eng), 'full', randseed)

#%%# Run with iterative random seeds to test spa vs. eng
ts = 0
te = 0.20
niter = 1000 # see how the random seed affects accuracy
ncv = 15
shuffle = "full"

scores_p10 = []
scores_n40 = []

## randomize the spanish speakers to use
rng = np.random.default_rng(1)
perm = rng.permutation(len(p10_spa))

for n_iter in np.arange(0,niter,1):
    print("iter " + str(n_iter))
    ## decode eng vs. spa speakers: keep both n = 15 vs. n1 = 15, n2 = 16 gave very higher than chance results
    decoding_acc_p10 = do_subject_by_subject_decoding([p10_eng, p10_spa[perm,:][:-1,:]], times, ts, 0.04, ncv, shuffle, None)
    scores_p10.append(np.mean(decoding_acc_p10, axis=0))
    decoding_acc_n40 = do_subject_by_subject_decoding([n40_eng, n40_spa[perm,:][:-1,:]], times, ts, 0.07, ncv, shuffle, None)
    scores_n40.append(np.mean(decoding_acc_n40, axis=0))
scores_p10 = np.array(scores_p10)
scores_n40 = np.array(scores_n40)
print(f"Accuracy: {np.mean(scores_p10):.3f}")
print(f"Accuracy: {np.mean(scores_n40):.3f}")
plot_decoding_histograms(scores_p10,scores_n40,bins=5,chance=0.5,labels=("p10", "n40"),xlim=(0, 1))

#%%# Run with iterative random seeds to test differential wave (n40-p10) between spa vs. eng
ts = 0
te = 0.15
niter = 1000 # see how the random seed affects accuracy
ncv = 15
shuffle = "keep pair"
scores_diff = []

## randomize the spanish speakers to use
rng = np.random.default_rng(1)
perm = rng.permutation(len(p10_spa))
eng_diff = n40_eng-p10_eng
spa_diff = n40_spa-p10_spa

for n_iter in np.arange(0,niter,1):
    print("iter " + str(n_iter))
    ## decode eng vs. spa speakers: keep both n = 15 vs. n1 = 15, n2 = 16 gave very higher than chance results
    decoding_acc = do_subject_by_subject_decoding([eng_diff, spa_diff[perm,:][:-1,:]], times, ts, te, ncv, shuffle, None)
    scores_diff.append(np.mean(decoding_acc, axis=0))

scores_diff = np.array(scores_diff)
print(f"Accuracy: {np.mean(scores_diff):.3f}")

#%%# permuation test to compare p10/n40 vs. p10/p40 decoding in eng
ts = 0
te = 0.06
niter = 1000 # see how the random seed affects accuracy
shuffle = "keep pair"
randseed = 2
ncv = len(std)

## Use sub section to decode EEG: [caution] need to reload the std, dev1, dev2 to ensure the full length
# std = std[:,slice(ff(times, ts), ff(times, te))]
# dev1 = dev1[:,slice(ff(times, ts), ff(times, te))]
# dev2 = dev2[:,slice(ff(times, ts), ff(times, te))]

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
ax.axvline(np.mean(diff_scores_perm), color="grey", linestyle="--", linewidth=2)
# 95% line
ax.axvline(np.percentile(diff_scores_perm,95),ymin=0,ymax=1000,color='grey',linewidth=2)
# mean lines
ax.axvline(diff_acc, color="red", linewidth=1)

#%%# permuation test to compare Eng p10/n40 acc and Spa p10/n40 acc
ts = 0
te = 0.2
niter = 1000 # see how the random seed affects accuracy
shuffle = "full"
randseed = 2
ncv = 15

## random select 15 spa subjects to match the 15 eng subjects
rng0 = np.random.default_rng(0)
perm = rng0.permutation(len(p10_spa))
p10_spa_15 = p10_spa[perm,:][:-1,:]
n40_spa_15 = n40_spa[perm,:][:-1,:]

## real difference between eng and spa decoding accuracy of p10 vs. n40 sounds
decoding_acc_eng = do_subject_by_subject_decoding([p10_eng, n40_eng], times, ts, te, ncv, shuffle, randseed)
decoding_acc_spa = do_subject_by_subject_decoding([p10_spa_15, n40_spa_15], times, ts, te, ncv, shuffle, randseed)
diff_acc = np.mean(decoding_acc_eng, axis=0) - np.mean(decoding_acc_spa, axis=0)

## permute between 2 language groups
diff_scores_perm = []
p10_all = np.vstack([p10_eng,p10_spa_15])
n40_all = np.vstack([n40_eng,n40_spa_15])
n_total = p10_all.shape[0]
rng = np.random.default_rng(None)
 
for n_iter in np.arange(0,niter,1):
    print("iter " + str(n_iter))
    perm_ind = rng.permutation(n_total)
    group1_ind = perm_ind[:n_total//2]
    group2_ind = perm_ind[n_total//2:]
    p10_group1 = p10_all[group1_ind]
    p10_group2 = p10_all[group2_ind]
    n40_group1 = n40_all[group1_ind]
    n40_group2 = n40_all[group2_ind]

    decoding_acc_group1_perm = do_subject_by_subject_decoding([p10_group1, n40_group1], times, ts, te, ncv, shuffle, randseed)
    decoding_acc_group2_perm = do_subject_by_subject_decoding([p10_group2, n40_group2], times, ts, te, ncv, shuffle, randseed)

    diff_scores_perm.append(np.mean(decoding_acc_group1_perm, axis=0) - np.mean(decoding_acc_group2_perm, axis=0))
diff_scores_perm = np.array(diff_scores_perm)
print(f"Accuracy: {np.mean(diff_scores_perm):.3f}")

fig, ax = plt.subplots(1)
ax.hist(diff_scores_perm, bins=7, alpha=0.6)
ax.set_ylabel("Count", fontsize=20)
ax.set_xlabel("Accuracy Difference", fontsize=20)

# chance line
ax.axvline(np.mean(diff_scores_perm), color="grey", linestyle="--", linewidth=2)
# 95% line
ax.axvline(np.percentile(diff_scores_perm,95),ymin=0,ymax=1000,color='grey', linestyle="--", linewidth=2)
# mean lines
ax.axvline(diff_acc, color="red", linewidth=1)

#%%####################################### Subject-by-subject EEG or misc decoding CBS dataset
## change ts and te for C and V section decoding: for ba, 10 ms + 90 ms = 100 ms; for mba and pa, 40 ms + 90 ms = 130 ms
## epoch length -20 to 200 ms with sampling rate at 5000 Hz

## Use V section to decode EEG
# ts = 0.02, te = 0.13; ts = 0.06, te = 0.17
std_v = std[:,slice(ff(times, 0.06), ff(times, 0.2))]
dev1_v = dev1[:,slice(ff(times, 0.06), ff(times, 0.2))]
dev2_v = dev2[:,slice(ff(times, 0.06), ff(times, 0.2))]

scores_ba_mba_pa = do_subject_by_subject_decoding([std_v,dev1_v,dev2_v], times, -1000, 1000, 18, 'keep pair', None)
scores_ba_mba = do_subject_by_subject_decoding([std_v,dev1_v], times, -1000, 1000, 18, 'keep pair', None)
scores_ba_pa = do_subject_by_subject_decoding([std_v,dev2_v], times, -1000, 1000, 18, 'keep pair', None)
scores_mba_pa = do_subject_by_subject_decoding([dev1_v,dev2_v], times, -1000, 1000, 18, 'keep pair', None)

## Use C section to decode
ncv = len(std)
####################################### 3-way decoding: ba vs. pa vs. mba
scores_ba_mba_pa = do_subject_by_subject_decoding([std,dev1,dev2], times, ts, te, ncv, shuffle, randseed)

####################################### 2-way decoding: ba vs. pa, ba vs. mba, pa vs. mba
scores_ba_mba = do_subject_by_subject_decoding([std,dev1], times, ts, te, ncv, shuffle, randseed)
score_ba_mba = np.mean(scores_ba_mba, axis=0)
print("Decoding Accuracy between ba vs. mba: %0.1f%%" % (100 * score_ba_mba,))

scores_ba_pa = do_subject_by_subject_decoding([std,dev2], times, ts, te, ncv, shuffle, randseed)
score_ba_pa = np.mean(scores_ba_mba, axis=0)
print("Decoding Accuracy between ba vs. pa: %0.1f%%" % (100 * score_ba_pa,))

scores_mba_pa = do_subject_by_subject_decoding([dev1,dev2], times, ts, te, ncv, shuffle, randseed)
score_mba_pa = np.mean(scores_mba_pa, axis=0)
print("Decoding Accuracy between mba vs. pa: %0.1f%%" % (100 * score_mba_pa,))

#%%####################################### Spectrum analysis
fmin = 50
fmax = 150

signal = n40_spa.mean(0) # CBS: std, dev1, dev2 (EEG, audio, misc); brainstem: p10_eng, p10_spa, n40_eng, n40_spa (EEG)
psds, freqs = mne.time_frequency.psd_array_welch(
    signal,fs, # could replace with label time series
    n_fft=len(signal),
    n_overlap=0,
    n_per_seg=None,
    fmin=fmin,
    fmax=fmax,)
plt.figure()
plt.title('Spectrum')
plt.plot(freqs,psds)
plt.xlim([60, 140])

#%%####################################### Spectrogram analysis
import librosa
import librosa.display

signal = p10_eng.mean(0)
signal = signal.astype(np.float32)
signal = signal/np.max(np.abs(signal))

nfft = 128
S = librosa.stft(
    signal,
    n_fft = nfft,
    hop_length = nfft//16,
    win_length = nfft,
    window = 'hann')
S_db = librosa.amplitude_to_db(np.abs(S),ref=np.max)

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

#%%####################################### ITPC analysis
data = p10_spa

ts = 0.025
te = 0.1 # 100 for ba and 130 for mba and pa
tslice = slice(ff(times, ts), ff(times, te))
fmin = 60
fmax = 140
itpc = itpc_hilbert(data, fs,times,ts,te,fmin,fmax)
# plt.figure()
plt.plot(times[tslice],itpc)
plt.xlim(times[tslice][0], times[tslice][-1])
plt.legend(['eng','spa'])
plt.title('ITPC between ' + str(fmin) + ' Hz and ' + str(fmax) + ' Hz')

#%%####################################### SNR analysis
level = 'individual' # 'group' (mean first then SNR) or 'individual' (SNR on individual level then mean)

data = n40_eng

ts = 0
te = 0.13 # 100 for ba and 130 for mba and pa
fmin = 50
fmax = 150

SNR_t, SNR_s = do_SNR(data,times,ts,te,fmin, fmax,level)
print('temporal SNR: ' + str(np.array(SNR_t).mean()) + '(' + str(np.array(SNR_t).std()/np.sqrt(len(data))) +')')
print('spectral SNR: ' + str(np.array(SNR_s).mean()) + '(' + str(np.array(SNR_s).std()/np.sqrt(len(data))) +')')

#%%####################################### xcorr analysis
level = 'individual'

ts = 0 # 0.02 for ba and pa, 0.06 for mba
te = 0.20 # 0.1 for ba and 0.13 for mba and pa, this is hard cut off because audio files are this long

fs_audio, p10_audio = load_CBS_file('audio', 'p10', 'adults')
fs_audio, n40_audio = load_CBS_file('audio', 'n40', 'adults')
fs_eeg, p10_eng, n40_eng, p10_spa, n40_spa = load_brainstem_file(file_type, ntrial)

audio = p10_audio
EEG = p10_eng
audio = stats.zscore(audio)
EEG = stats.zscore(EEG,axis=-1)

xcorr = do_xcorr(audio,EEG,fs_audio,fs_eeg,ts,te,level)

print(np.mean(xcorr['xcorr_max']))
print(np.mean(xcorr['xcorr_lag_ms']))

#%%
xcorr1 = xcorr['xcorr_max']
lag1 = xcorr['xcorr_lag_ms']

stats.ttest_rel(xcorr1,xcorr2)

#%%
# print(xcorr)
print(np.mean(xcorr['xcorr_max']))
print(np.mean(xcorr['xcorr_lag_ms']))

## run stats ANOVA
import pingouin as pg
import pandas as pd

df = pd.read_csv('/home/tzcheng/Downloads/FFR_xcorr_window2 - Brainstem - 3000.csv')
aov = pg.mixed_anova(
    dv='xcorr_lags',
    within='condition',
    between='group',
    subject='subject',
    data=df
)
print(aov)

df = pd.read_csv('/home/tzcheng/Downloads/FFR_xcorr - CBS.csv')
condition1 = df.loc[(df["condition"] == 'p10') & (df["group"]== 'eng')]['xcorr_lags']
condition2 = df.loc[(df["condition"] == 'n40') & (df["group"]== 'eng')]['xcorr_lags']
stats.ttest_rel(condition1,condition2)

## run stats paired and independent ttest
df.groupby(['condition', 'group'])['xcorr_coef'].mean()
df.groupby(['condition', 'group'])['xcorr_lags'].mean()

condition1 = df.loc[(df["condition"] == 'p10') & (df["group"]== 'eng')]['xcorr_lags']
condition2 = df.loc[(df["condition"] == 'n40') & (df["group"]== 'eng')]['xcorr_lags']
stats.ttest_rel(condition1,condition2)
group1 = df.loc[(df["condition"] == 'p10') & (df["group"]== 'eng')]['xcorr_lags']
group2 = df.loc[(df["condition"] == 'p10') & (df["group"]== 'spa')]['xcorr_lags']
stats.ttest_ind(group1,group2)


#%%####################################### autocorr analysis
level = 'group'
autocorr = do_autocorr_sliding(std, fs_eeg, times, ts, te, level)