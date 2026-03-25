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
import scipy as sp
from scipy import stats, signal
from scipy.signal import butter, filtfilt, hilbert
from scipy.io import savemat
import numpy as np
from scipy.io import wavfile
from mne.decoding import (
    cross_val_multiscore,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import random
import librosa
import librosa.display

#%%####################################### Define functions
## General supporting functions
def ff(input_arr,target): 
    ## find the idx of the closest freqeuncy (time) in freqs (times)
    delta = 1000000000
    idx = -1
    for i, val in enumerate(input_arr):
        if abs(input_arr[i]-target) < delta:
            idx = i
            delta = abs(input_arr[i]-target)
    return idx

def zero_pad(signal, target_len):
    pad_len = target_len - len(signal)
    if pad_len <= 0:
        return signal[:target_len]  # trim if too long
    return np.pad(signal, (0, pad_len), mode='constant')

def random_select(data,num_select,randseed):
    rng = np.random.default_rng(randseed)
    perm = rng.permutation(len(data))
    new_data = data[perm][:num_select]
    return new_data

## Loading functions
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

def load_brainstem_file(file_type, nfilter, ntrial, ntop):
    """
    file_type: 'sensor', 'roi','morph'
    nfilter: '80200' or '802000'
    ntrial: '200' or 'all'
    ntop = '0' or '3' '0': no PCA was performed
    """
    root_path = '/media/tzcheng/storage/Brainstem/'
    fs = 5000
    if file_type == 'EEG':
        p10_eng = np.load(root_path + 'EEG/p10_eng_eeg_ntr' + ntrial + '_01.npy')
        n40_eng = np.load(root_path + 'EEG/n40_eng_eeg_ntr' + ntrial + '_01.npy')
        p10_spa = np.load(root_path + 'EEG/p10_spa_eeg_ntr' + ntrial + '_01.npy')
        n40_spa = np.load(root_path + 'EEG/n40_spa_eeg_ntr' + ntrial + '_01.npy')
    elif file_type in ('sensor', 'roi','morph'): ## for the MEG
        p10_eng = np.load(root_path + 'MEG/FFR/eng_group_pcffr' + nfilter + '_ntrial' + ntrial + '_' + ntop + '_p10_01_' + file_type + '.npy')
        n40_eng = np.load(root_path + 'MEG/FFR/eng_group_pcffr' + nfilter + '_ntrial' + ntrial + '_' + ntop + '_n40_01_' + file_type + '.npy')
        p10_spa = np.load(root_path + 'MEG/FFR/spa_group_pcffr' + nfilter + '_ntrial' + ntrial + '_' + ntop + '_p10_01_' + file_type + '.npy')
        n40_spa = np.load(root_path + 'MEG/FFR/spa_group_pcffr' + nfilter + '_ntrial' + ntrial + '_' + ntop + '_n40_01_' + file_type + '.npy')
    return fs, p10_eng, n40_eng, p10_spa, n40_spa

## Plotting functions
def plot_err(group_stc,color,t):
    group_avg=np.mean(group_stc,axis=0)
    err=np.std(group_stc,axis=0)/np.sqrt(group_stc.shape[0])
    up=group_avg+err
    lw=group_avg-err
    plt.plot(t,group_avg,color=color)
    plt.fill_between(t,up,lw,color=color,alpha=0.5)

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
    plt.figure(figsize=(8,5))
    plt.plot(times, data1.mean(0), label=label1)
    plt.plot(times, data2.mean(0), label=label2)
    plt.xlim(np.min(times), np.max(times))
    plt.ylim(*ylim)
    plt.legend()

    # Differential response
    plt.figure()
    plt.title('Differential response')
    plot_err(data2-data1,'k',times)
    plt.xlim(np.min(times), np.max(times))
    plt.ylim(*ylim)

def plot_audio_ffr(times, audio, fs_audio, ffr, te, ts = 0, fs_eeg = 5000, label1='audio',label2='ffr'):
    """
    Plot rescaled audio and FFR to observe the temporal relationship.
    """
    num = int((len(audio)*fs_eeg)/fs_audio)    
    audio_rs = signal.resample(audio, num, t=None, axis=0, window=None)
    audio_rs_norm = (audio_rs - np.mean(audio_rs)) / np.std(audio_rs)
    ffr_norm = (ffr - np.mean(ffr, axis=1, keepdims=True)) / np.std(ffr, axis=1, keepdims=True)
    times_audio = np.linspace(0,len(audio_rs)/fs_eeg,len(audio_rs))
    tslice_audio = slice(ff(times_audio, ts), ff(times_audio, te))
    tslice_EEG = slice(ff(times, ts), ff(times, te))
    
    plt.figure()
    plt.plot(times_audio[tslice_audio],audio_rs_norm[tslice_audio],label=label1)
    plt.plot(times[tslice_EEG],ffr_norm[:,tslice_EEG].mean(0),label=label2)
    plt.legend()
    
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

## Decoding functions
def do_subject_by_subject_decoding(X_list,times,ts,te,ncv,shuffle,random_state):
    """
    X_list : list of ndarrays
        One array per condition/category
    """

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
   
    ## classifier
    clf = make_pipeline(
        StandardScaler(),  # z-score normalization
        # SVC(kernel='rbf',gamma='auto',C=0.1,class_weight='balanced')  
        SVC(kernel='linear', C=1,class_weight='balanced')
    )
    scores = cross_val_multiscore(clf, X, y, cv=ncv, n_jobs=None)
    score = np.mean(scores, axis=0)
    print("Decoding Accuracy %0.1f%%" % (100 * score))
    
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
    # plt.plot(times[tslice],np.mean(weights,axis=0))
    # plt.title('SVM weights across time')
    
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

def run_random_seed_decoding(
    condition_pairs,
    times,
    ts,
    te,
    niter=1000,
    shuffle="full",
    labels=None,
    plot=True
):

    n_pairs = len(condition_pairs)
    scores_all = [[] for _ in range(n_pairs)]

    for n_iter in range(niter):
        print(f"iter {n_iter}")

        for i, ((cond1, cond2)) in enumerate(condition_pairs):

            acc = do_subject_by_subject_decoding(
                [cond1, cond2],
                times,
                ts,
                te,
                len(cond1),   # match CV to group size
                shuffle,
                None
            )
            scores_all[i].append(np.mean(acc, axis=0))

    scores_all = [np.array(s) for s in scores_all]

    # print mean accuracy
    for i, scores in enumerate(scores_all):
        label = labels[i] if labels else f"cond{i}"
        print(f"Accuracy {label}: {np.mean(scores):.3f}")

    if plot and n_pairs == 2:
        plot_decoding_histograms(
            scores_all[0], scores_all[1],
            bins=5,
            chance=0.5,
            labels=labels if labels else ("cond0", "cond1"),
            xlim=(0, 1)
        )
    return scores_all

def make_decode_fn(times, ts, te, ncv, shuffle, randseed):
    def decode_fn(group1, group2):
        acc1 = do_subject_by_subject_decoding(
            group1, times, ts, te, ncv, shuffle, randseed
        )
        acc2 = do_subject_by_subject_decoding(
            group2, times, ts, te, ncv, shuffle, randseed
        )
        return np.mean(acc1, axis=0) - np.mean(acc2, axis=0)
    return decode_fn

def permutation_decoding_diff(
    condition_pairs,
    decode_fn,
    niter=1000,
    rng=None,
    plot = True,
    verbose=True
):
    """
    condition_pairs: (group1, group2)
        group1, group2 are lists of arrays
        e.g. ([p10_eng, n40_eng], [p10_spa, n40_spa])
    """
    if rng is None:
        rng = np.random.default_rng()

    group1, group2 = condition_pairs

    # real difference
    real_diff = decode_fn(group1, group2)

    # stack corresponding conditions
    stacked = [np.vstack([g1, g2]) for g1, g2 in zip(group1, group2)]
    n_total = stacked[0].shape[0]

    diff_scores_perm = []

    for i in range(niter):
        if verbose:
            print(f"iter {i}")

        perm = rng.permutation(n_total)
        idx1 = perm[:n_total // 2]
        idx2 = perm[n_total // 2:]

        perm_group1 = [s[idx1] for s in stacked]
        perm_group2 = [s[idx2] for s in stacked]

        diff_scores_perm.append(decode_fn(perm_group1, perm_group2))

    if plot:
        # ---- plotting ----
        fig, ax = plt.subplots(1)
        ax.hist(diff_scores_perm, bins=5, alpha=0.6)

        ax.set_ylabel("Count", fontsize=20)
        ax.set_xlabel("Accuracy Difference", fontsize=20)

        ax.axvline(np.mean(diff_scores_perm), color="grey", linestyle="--", linewidth=2)
        ax.axvline(np.percentile(diff_scores_perm, 95), color="grey", linestyle="--", linewidth=2)
        ax.axvline(real_diff, color="red", linewidth=2)
    return real_diff, np.array(diff_scores_perm), fig, ax

def run_sliding_time_decoding(
    p10_eng,
    n40_eng,
    p10_spa,
    n40_spa,
    times,
    ts_global,
    te_global,
    window_length,
    window_step,
    ncv_eng,
    ncv_spa,
    shuffle,
    randseed,
    plot=True
):
    """
    Sliding window decoding for English vs Spanish groups.
    """

    acc_eng_by_window = []
    acc_spa_by_window = []

    window_starts = np.arange(
        ts_global,
        te_global - window_length,
        window_step
    )

    for t_start in window_starts:
        t_end = t_start + window_length

        decoding_acc_eng = do_subject_by_subject_decoding(
            [p10_eng, n40_eng],
            times,
            t_start,
            t_end,
            ncv_eng,
            shuffle,
            randseed
        )

        decoding_acc_spa = do_subject_by_subject_decoding(
            [p10_spa, n40_spa],
            times,
            t_start,
            t_end,
            ncv_spa,
            shuffle,
            randseed
        )

        acc_eng_by_window.append(np.mean(decoding_acc_eng))
        acc_spa_by_window.append(np.mean(decoding_acc_spa))

    # Convert to arrays
    acc_eng_by_window = np.array(acc_eng_by_window)
    acc_spa_by_window = np.array(acc_spa_by_window)

    # Window centers (ms)
    window_centers_ms = (window_starts + window_length / 2) * 1000

    # Optional plotting
    if plot:
        plt.figure(figsize=(6, 4))

        plt.plot(window_centers_ms, acc_eng_by_window,
                 label='English')
        plt.plot(window_centers_ms, acc_spa_by_window,
                 label='Spanish')

        plt.axhline(0.5, linestyle='--', label='Chance')

        plt.xlabel('Time (ms)')
        plt.ylabel('Decoding accuracy')
        plt.title('Sliding window decoding accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return {
        "times_ms": window_centers_ms,
        "acc_eng": acc_eng_by_window,
        "acc_spa": acc_spa_by_window
    }

def run_increment_decoding(
    cond1_a, cond1_b,          # e.g., p10_eng, n40_eng
    cond2_a, cond2_b,          # e.g., p10_spa, n40_spa
    times,
    ts=0,
    te=0.2,
    window_step=0.005,
    ncv1=15,
    ncv2=16,
    shuffle="keep pair",
    randseed=2,
    do_permutation=True,
    niter=500,
    labels=("Cond1", "Cond2"),
    plot=True
):
    """
    General increment decoding function.
    Works for:
    - Eng vs Spa
    - p10/n40 vs p10/p40
    """

    window_ends = np.arange(window_step, te + window_step, window_step)

    acc1_all = []
    acc2_all = []
    diff_real = []
    diff_perm_95 = []

    # --- fixed noise (important for your design) ---
    noise1 = gen_noise(cond1_a, randseed)
    noise2 = gen_noise(cond2_a, randseed)

    for t_end in window_ends:

        # Apply masking
        c1a_w = mask_with_fixed_noise(cond1_a, times, t_end, noise1)
        c1b_w = mask_with_fixed_noise(cond1_b, times, t_end, noise1)

        c2a_w = mask_with_fixed_noise(cond2_a, times, t_end, noise2)
        c2b_w = mask_with_fixed_noise(cond2_b, times, t_end, noise2)

        # Decode
        acc1 = do_subject_by_subject_decoding(
            [c1a_w, c1b_w], times, ts, te, ncv1, shuffle, randseed
        )
        acc2 = do_subject_by_subject_decoding(
            [c2a_w, c2b_w], times, ts, te, ncv2, shuffle, randseed
        )

        acc1_mean = np.mean(acc1)
        acc2_mean = np.mean(acc2)

        acc1_all.append(acc1_mean)
        acc2_all.append(acc2_mean)

        # Real difference
        diff = acc1_mean - acc2_mean
        diff_real.append(diff)

        # --- Permutation test ---
        if do_permutation:
            diff_scores = []

            data_a = np.vstack([cond1_a, cond2_a])
            data_b = np.vstack([cond1_b, cond2_b])
            n_total = data_a.shape[0]

            rng = np.random.default_rng(None)

            for _ in range(niter):
                print(f"iter {_}")
                perm = rng.permutation(n_total)
                g1 = perm[:n_total // 2]
                g2 = perm[n_total // 2:]

                acc_g1 = do_subject_by_subject_decoding(
                    [data_a[g1], data_b[g1]],
                    times, ts, te, ncv1, shuffle, randseed
                )
                acc_g2 = do_subject_by_subject_decoding(
                    [data_a[g2], data_b[g2]],
                    times, ts, te, ncv1, shuffle, randseed
                )

                diff_scores.append(np.mean(acc_g1) - np.mean(acc_g2))

            diff_perm_95.append(np.percentile(diff_scores, 95))

    # Convert to arrays
    acc1_all = np.array(acc1_all)
    acc2_all = np.array(acc2_all)
    diff_real = np.array(diff_real)
    diff_perm_95 = np.array(diff_perm_95) if do_permutation else None

    window_sizes_ms = window_ends * 1000

    # --- Plotting ---
    if plot:
        plt.figure(figsize=(6, 4))
        plt.plot(window_sizes_ms, acc1_all, label=labels[0], marker='o')
        plt.plot(window_sizes_ms, acc2_all, label=labels[1], marker='o')
        plt.axhline(0.5, linestyle='--', label='Chance')

        plt.xlabel('Window size (ms)')
        plt.ylabel('Decoding accuracy')
        plt.title('Decoding vs available signal')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        if do_permutation:
            plt.figure(figsize=(6, 4))
            plt.plot(window_sizes_ms, diff_real, label='Real difference', marker='o')
            plt.plot(window_sizes_ms, diff_perm_95, label='95% perm', marker='o')
            plt.axhline(0, linestyle='--', label='No difference')

            plt.xlabel('Window size (ms)')
            plt.ylabel('Accuracy difference')
            plt.title(f'{labels[0]} - {labels[1]}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    return {
        "window_ms": window_sizes_ms,
        "acc1": acc1_all,
        "acc2": acc2_all,
        "diff_real": diff_real,
        "diff_perm_95": diff_perm_95
    }

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
        
def do_brainstem_trial_by_trial_decoding(root_path,n_trials='ntrial_all'):
    ## Trial-by-trial decoding for spa vs. eng: not working well
    subjects_eng=['104','106','107','108','110','111','112','113','118','121','123','124','126','129','133']
    subjects_spa=['203','204','205','206','211','212','213','214','215','220','221','222','223','224','225'] # trim 226 so be n = 15 too
    # subjects_spa=['203','204','205','206','211','212','213','214','215','220','221','222','223','224','225','226'] 
    
    ## SVM showed higher accuracy in trial-by-trial decoding
    clf = make_pipeline(
        StandardScaler(),
        # SVC(kernel='rbf',gamma='auto')  
        SVC(kernel='linear', C=1,class_weight='balanced')
       )
    
    all_score_svm = []

    for se, sp in zip(subjects_eng,subjects_spa):
        ## do eng first because they are the focus 
        epochs_eng_p10 = mne.read_epochs(root_path + 'preprocessed/' + n_trials + '/eng/brainstem_' + se  +'_p10_01_cabr_e_all.fif').pick_types(eeg=True, exclude=[])
        epochs_eng_n40 = mne.read_epochs(root_path + 'preprocessed/' + n_trials + '/eng/brainstem_' + se  +'_n40_01_cabr_e_all.fif').pick_types(eeg=True, exclude=[])
        
        ## could do spa too
        epochs_spa_p10 = mne.read_epochs(root_path + 'preprocessed/' + n_trials + '/spa/brainstem_' + sp  +'_p10_01_cabr_e_all.fif').pick_types(eeg=True, exclude=[])
        epochs_spa_n40 = mne.read_epochs(root_path + 'preprocessed/' + n_trials + '/spa/brainstem_' + sp  +'_n40_01_cabr_e_all.fif').pick_types(eeg=True, exclude=[])
        
        X_p10 = np.squeeze(epochs_eng_p10.get_data())
        X_n40 = np.squeeze(epochs_eng_n40.get_data())

        X = np.concatenate((X_p10,X_n40),axis=0)
        y = np.concatenate((np.repeat(0,len(X_p10)),np.repeat(1,len(X_n40))))
                       
        scores = cross_val_multiscore(clf, X, y, cv=5, n_jobs=4) 
        score = np.mean(scores, axis=0)
        print("Trial-by-trial decoding accuracy: %0.1f%%" % (100 * score,))
        all_score_svm.append(score)
        return all_score_svm

## Signal processing functions
def replace_onset_with_baseline_noise(
        ffr,
        times,
        onset_start,
        onset_end,
        baseline_start=-0.02,
        baseline_end=0.0,
        plot_new = False):
    """
    Replace onset period with prestimulus noise for the mask-peak decoding

    Parameters
    ----------
    ffr : array (subjects, time)
    times : time vector (seconds)
    """

    ffr_new = ffr.copy()

    onset_mask = (times >= onset_start) & (times <= onset_end)
    baseline_mask = (times >= baseline_start) & (times < baseline_end)

    onset_idx = np.where(onset_mask)[0]
    baseline_idx = np.where(baseline_mask)[0]

    n_sub = ffr.shape[0]

    for sub in range(n_sub):

        baseline_noise = np.tile(ffr[sub, baseline_idx],10)

        # sample baseline noise (with replacement)
        replacement = baseline_noise[:len(onset_idx)]
        ffr_new[sub, onset_idx] = replacement

    if plot_new:
        ## plot average FFRs between p10 vs. n40 to check
        plt.figure()
        plt.plot(times,ffr)
        plt.plot(times,ffr_new)

    return ffr_new

def bandpass_filter(data, fs, fmin, fmax, order=4):
    nyq = fs / 2
    b, a = butter(
        order,
        [fmin / nyq, fmax / nyq],
        btype="band"
    )
    return filtfilt(b, a, data, axis=-1)

def bandpower_hilbert(data, fs, f_min, f_max, order=4):

    nyq = fs / 2
    b, a = butter(order, [f_min/nyq, f_max/nyq], btype='band')

    analytic_all = []
    
    for subj in range(data.shape[0]):
        signal = data[subj].astype(np.float32)

        # zero-phase filtering
        filtered = filtfilt(b, a, signal)

        # analytic signal
        analytic = hilbert(filtered)

        # amplitude envelope
        amplitude = np.abs(analytic)

        analytic_all.append(amplitude)

    return np.array(analytic_all)

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
    xx = x / np.linalg.norm(x)
    return xx

def do_xcorr(audio,EEG,fs_audio,fs_eeg,ts,te,level,lag_window_s=(-0.014, -0.007)):
    # Downsample
    num = int((len(audio)*fs_eeg)/fs_audio)    
    audio_rs = signal.resample(audio, num, t=None, axis=0, window=None)
    times_audio = np.linspace(0,len(audio_rs)/fs_eeg,len(audio_rs))
    times_eeg = np.linspace(-0.02,0.2,1101)

    ## p10: noise burst from 0 ms (100th points) ts = 100 # 0.02s (i.e. 0.02s after noise burst) te = 500 # 0.1s
    ## n40: noise burst from 40 ms (200th points) ts = 200 + 100 # 0.06s (i.e. 0.02s after noise burst) te = 650 # 0.13s
    ## p40: noise burst from 0 ms (100th points) ts = 100 # 0.02s te = 650 # 0.13s
    tslice_audio = slice(ff(times_audio, ts), ff(times_audio, te)) ## +1 hack to make sure slice in audio and eeg are the same length
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
            
            ## plot the signal
            fig, axes = plt.subplots(
                3, 1            )

            axes = axes.flatten()
            axes[0].plot(times_audio[tslice_audio], stim)
            axes[1].plot(times_eeg[tslice_EEG],resp_z)
            axes[2].plot(lags_s,xcorr)
            axes[2].axvline(lag_min, color="grey", linestyle="--")
            axes[2].axvline(lag_max, color="grey", linestyle="--")

        return {
            "xcorr_max": np.array(xcorr_max),
            "xcorr_lag_ms": np.array(xcorr_lag)
        }
    else:
        raise ValueError("level must be 'group' or 'individual'")

def do_xcorr_sliding(audio, EEG, fs_audio, fs_eeg,
                     ts, te,
                     win_len_s,
                     step_s,
                     lag_window_s=(-0.014, -0.007)):

    # --- Downsample audio to EEG sampling rate ---
    num = int((len(audio) * fs_eeg) / fs_audio)
    audio_rs = signal.resample(audio, num, axis=0)

    times_audio = np.linspace(0, len(audio_rs)/fs_eeg, len(audio_rs))
    times_eeg = np.linspace(-0.02, 0.2, EEG.shape[1])

    tslice_audio = slice(ff(times_audio, ts), ff(times_audio, te))
    tslice_EEG   = slice(ff(times_eeg, ts), ff(times_eeg, te))

    stim = audio_rs[tslice_audio]
    resp = EEG[:, tslice_EEG]

    # --- Window parameters in samples ---
    win_len = int(win_len_s * fs_eeg)
    step = int(step_s * fs_eeg)

    n_samples = len(stim)
    n_windows = int((n_samples - win_len) / step) + 1

    # --- Lag setup ---
    lags = signal.correlation_lags(win_len, win_len)
    lags_s = lags / fs_eeg

    lag_min, lag_max = lag_window_s
    lag_mask = (lags_s >= lag_min) & (lags_s <= lag_max)

    # --- Storage ---
    xcorr_max = np.zeros((resp.shape[0], n_windows))
    xcorr_lag = np.zeros((resp.shape[0], n_windows))
    window_times = []

    # --- Sliding window loop ---
    for w in range(n_windows):
        start = w * step
        end = start + win_len

        stim_win = stim[start:end]
        stim_z = zscore_and_normalize(stim_win)

        window_times.append((start + end) / 2 / fs_eeg)

        for i, subj_resp in enumerate(resp):
            resp_win = subj_resp[start:end]
            resp_z = zscore_and_normalize(resp_win)

            xcorr = np.abs(signal.correlate(stim_z, resp_z, mode="full"))
            xcorr_win = xcorr[lag_mask]
            lag_win = lags_s[lag_mask]

            idx = np.argmax(xcorr_win)

            xcorr_max[i, w] = xcorr_win[idx]
            xcorr_lag[i, w] = lag_win[idx]

    return np.array(window_times), xcorr_max, xcorr_lag

def mask_with_fixed_noise(data, times, t_end, fixed_noise):
    data_out = data.copy()
    mask = times > t_end
    data_out[:, mask] = fixed_noise[:, mask]
    return data_out

def gen_noise(data,randseed):
    np.random.seed(randseed)
    # estimate noise scale from real data
    noise_std = np.std(data, axis=1, keepdims=True)
    noise_out = np.random.normal(
        loc=0,
        scale=noise_std,
        size=data.shape
    )
    return noise_out


#%%####################################### Define functions
## General supporting functions
def ff(input_arr,target): 
    ## find the idx of the closest freqeuncy (time) in freqs (times)
    delta = 1000000000
    idx = -1
    for i, val in enumerate(input_arr):
        if abs(input_arr[i]-target) < delta:
            idx = i
            delta = abs(input_arr[i]-target)
    return idx

def zero_pad(signal, target_len):
    pad_len = target_len - len(signal)
    if pad_len <= 0:
        return signal[:target_len]  # trim if too long
    return np.pad(signal, (0, pad_len), mode='constant')

def random_select(data,num_select,randseed):
    rng = np.random.default_rng(randseed)
    perm = rng.permutation(len(data))
    new_data = data[perm][:num_select]
    return new_data

## Loading functions
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

## Plotting functions
def plot_err(group_stc,color,t):
    group_avg=np.mean(group_stc,axis=0)
    err=np.std(group_stc,axis=0)/np.sqrt(group_stc.shape[0])
    up=group_avg+err
    lw=group_avg-err
    plt.plot(t,group_avg,color=color)
    plt.fill_between(t,up,lw,color=color,alpha=0.5)

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
    plt.figure(figsize=(8,5))
    plt.plot(times, data1.mean(0), label=label1)
    plt.plot(times, data2.mean(0), label=label2)
    plt.xlim(np.min(times), np.max(times))
    plt.ylim(*ylim)
    plt.legend()

    # Differential response
    plt.figure()
    plt.title('Differential response')
    plot_err(data2-data1,'k',times)
    plt.xlim(np.min(times), np.max(times))
    plt.ylim(*ylim)

def plot_audio_ffr(times, audio, fs_audio, ffr, te, ts = 0, fs_eeg = 5000, label1='audio',label2='ffr'):
    """
    Plot rescaled audio and FFR to observe the temporal relationship.
    """
    num = int((len(audio)*fs_eeg)/fs_audio)    
    audio_rs = signal.resample(audio, num, t=None, axis=0, window=None)
    audio_rs_norm = (audio_rs - np.mean(audio_rs)) / np.std(audio_rs)
    ffr_norm = (ffr - np.mean(ffr, axis=1, keepdims=True)) / np.std(ffr, axis=1, keepdims=True)
    times_audio = np.linspace(0,len(audio_rs)/fs_eeg,len(audio_rs))
    tslice_audio = slice(ff(times_audio, ts), ff(times_audio, te))
    tslice_EEG = slice(ff(times, ts), ff(times, te))
    
    plt.figure()
    plt.plot(times_audio[tslice_audio],audio_rs_norm[tslice_audio],label=label1)
    plt.plot(times[tslice_EEG],ffr_norm[:,tslice_EEG].mean(0),label=label2)
    plt.legend()
    
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

## Decoding functions
def do_subject_by_subject_decoding(X_list,times,ts,te,ncv,shuffle,random_state):
    """
    X_list : list of ndarrays
        One array per condition/category
    """

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
   
    ## classifier
    clf = make_pipeline(
        StandardScaler(),  # z-score normalization
        # SVC(kernel='rbf',gamma='auto',C=0.1,class_weight='balanced')  
        SVC(kernel='linear', C=1,class_weight='balanced')
    )
    scores = cross_val_multiscore(clf, X, y, cv=ncv, n_jobs=None)
    score = np.mean(scores, axis=0)
    print("Decoding Accuracy %0.1f%%" % (100 * score))
    
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
    # plt.plot(times[tslice],np.mean(weights,axis=0))
    # plt.title('SVM weights across time')
    
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

def run_random_seed_decoding(
    condition_pairs,
    times,
    ts,
    te,
    niter=1000,
    shuffle="full",
    labels=None,
    plot=True
):

    n_pairs = len(condition_pairs)
    scores_all = [[] for _ in range(n_pairs)]

    for n_iter in range(niter):
        print(f"iter {n_iter}")

        for i, ((cond1, cond2)) in enumerate(condition_pairs):

            acc = do_subject_by_subject_decoding(
                [cond1, cond2],
                times,
                ts,
                te,
                len(cond1),   # match CV to group size
                shuffle,
                None
            )
            scores_all[i].append(np.mean(acc, axis=0))

    scores_all = [np.array(s) for s in scores_all]

    # print mean accuracy
    for i, scores in enumerate(scores_all):
        label = labels[i] if labels else f"cond{i}"
        print(f"Accuracy {label}: {np.mean(scores):.3f}")

    if plot and n_pairs == 2:
        plot_decoding_histograms(
            scores_all[0], scores_all[1],
            bins=5,
            chance=0.5,
            labels=labels if labels else ("cond0", "cond1"),
            xlim=(0, 1)
        )
    return scores_all

def make_decode_fn(times, ts, te, ncv, shuffle, randseed):
    def decode_fn(group1, group2):
        acc1 = do_subject_by_subject_decoding(
            group1, times, ts, te, ncv, shuffle, randseed
        )
        acc2 = do_subject_by_subject_decoding(
            group2, times, ts, te, ncv, shuffle, randseed
        )
        return np.mean(acc1, axis=0) - np.mean(acc2, axis=0)
    return decode_fn

def permutation_decoding_diff(
    condition_pairs,
    decode_fn,
    niter=1000,
    rng=None,
    plot = True,
    verbose=True
):
    """
    condition_pairs: (group1, group2)
        group1, group2 are lists of arrays
        e.g. ([p10_eng, n40_eng], [p10_spa, n40_spa])
    """
    if rng is None:
        rng = np.random.default_rng()

    group1, group2 = condition_pairs

    # real difference
    real_diff = decode_fn(group1, group2)

    # stack corresponding conditions
    stacked = [np.vstack([g1, g2]) for g1, g2 in zip(group1, group2)]
    n_total = stacked[0].shape[0]

    diff_scores_perm = []

    for i in range(niter):
        if verbose:
            print(f"iter {i}")

        perm = rng.permutation(n_total)
        idx1 = perm[:n_total // 2]
        idx2 = perm[n_total // 2:]

        perm_group1 = [s[idx1] for s in stacked]
        perm_group2 = [s[idx2] for s in stacked]

        diff_scores_perm.append(decode_fn(perm_group1, perm_group2))

    if plot:
        # ---- plotting ----
        fig, ax = plt.subplots(1)
        ax.hist(diff_scores_perm, bins=5, alpha=0.6)

        ax.set_ylabel("Count", fontsize=20)
        ax.set_xlabel("Accuracy Difference", fontsize=20)

        ax.axvline(np.mean(diff_scores_perm), color="grey", linestyle="--", linewidth=2)
        ax.axvline(np.percentile(diff_scores_perm, 95), color="grey", linestyle="--", linewidth=2)
        ax.axvline(real_diff, color="red", linewidth=2)
    return real_diff, np.array(diff_scores_perm), fig, ax

def run_sliding_time_decoding(
    p10_eng,
    n40_eng,
    p10_spa,
    n40_spa,
    times,
    ts_global,
    te_global,
    window_length,
    window_step,
    ncv_eng,
    ncv_spa,
    shuffle,
    randseed,
    plot=True
):
    """
    Sliding window decoding for English vs Spanish groups.
    """

    acc_eng_by_window = []
    acc_spa_by_window = []

    window_starts = np.arange(
        ts_global,
        te_global - window_length,
        window_step
    )

    for t_start in window_starts:
        t_end = t_start + window_length

        decoding_acc_eng = do_subject_by_subject_decoding(
            [p10_eng, n40_eng],
            times,
            t_start,
            t_end,
            ncv_eng,
            shuffle,
            randseed
        )

        decoding_acc_spa = do_subject_by_subject_decoding(
            [p10_spa, n40_spa],
            times,
            t_start,
            t_end,
            ncv_spa,
            shuffle,
            randseed
        )

        acc_eng_by_window.append(np.mean(decoding_acc_eng))
        acc_spa_by_window.append(np.mean(decoding_acc_spa))

    # Convert to arrays
    acc_eng_by_window = np.array(acc_eng_by_window)
    acc_spa_by_window = np.array(acc_spa_by_window)

    # Window centers (ms)
    window_centers_ms = (window_starts + window_length / 2) * 1000

    # Optional plotting
    if plot:
        plt.figure(figsize=(6, 4))

        plt.plot(window_centers_ms, acc_eng_by_window,
                 label='English')
        plt.plot(window_centers_ms, acc_spa_by_window,
                 label='Spanish')

        plt.axhline(0.5, linestyle='--', label='Chance')

        plt.xlabel('Time (ms)')
        plt.ylabel('Decoding accuracy')
        plt.title('Sliding window decoding accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return {
        "times_ms": window_centers_ms,
        "acc_eng": acc_eng_by_window,
        "acc_spa": acc_spa_by_window
    }

def run_increment_decoding(
    cond1_a, cond1_b,          # e.g., p10_eng, n40_eng
    cond2_a, cond2_b,          # e.g., p10_spa, n40_spa
    times,
    ts=0,
    te=0.2,
    window_step=0.005,
    ncv1=15,
    ncv2=16,
    shuffle="keep pair",
    randseed=2,
    do_permutation=True,
    niter=500,
    labels=("Cond1", "Cond2"),
    plot=True
):
    """
    General increment decoding function.
    Works for:
    - Eng vs Spa
    - p10/n40 vs p10/p40
    """

    window_ends = np.arange(window_step, te + window_step, window_step)

    acc1_all = []
    acc2_all = []
    diff_real = []
    diff_perm_95 = []

    # --- fixed noise (important for your design) ---
    noise1 = gen_noise(cond1_a, randseed)
    noise2 = gen_noise(cond2_a, randseed)

    for t_end in window_ends:

        # Apply masking
        c1a_w = mask_with_fixed_noise(cond1_a, times, t_end, noise1)
        c1b_w = mask_with_fixed_noise(cond1_b, times, t_end, noise1)

        c2a_w = mask_with_fixed_noise(cond2_a, times, t_end, noise2)
        c2b_w = mask_with_fixed_noise(cond2_b, times, t_end, noise2)

        # Decode
        acc1 = do_subject_by_subject_decoding(
            [c1a_w, c1b_w], times, ts, te, ncv1, shuffle, randseed
        )
        acc2 = do_subject_by_subject_decoding(
            [c2a_w, c2b_w], times, ts, te, ncv2, shuffle, randseed
        )

        acc1_mean = np.mean(acc1)
        acc2_mean = np.mean(acc2)

        acc1_all.append(acc1_mean)
        acc2_all.append(acc2_mean)

        # Real difference
        diff = acc1_mean - acc2_mean
        diff_real.append(diff)

        # --- Permutation test ---
        if do_permutation:
            diff_scores = []

            data_a = np.vstack([cond1_a, cond2_a])
            data_b = np.vstack([cond1_b, cond2_b])
            n_total = data_a.shape[0]

            rng = np.random.default_rng(None)

            for _ in range(niter):
                print(f"iter {_}")
                perm = rng.permutation(n_total)
                g1 = perm[:n_total // 2]
                g2 = perm[n_total // 2:]

                acc_g1 = do_subject_by_subject_decoding(
                    [data_a[g1], data_b[g1]],
                    times, ts, te, ncv1, shuffle, randseed
                )
                acc_g2 = do_subject_by_subject_decoding(
                    [data_a[g2], data_b[g2]],
                    times, ts, te, ncv1, shuffle, randseed
                )

                diff_scores.append(np.mean(acc_g1) - np.mean(acc_g2))

            diff_perm_95.append(np.percentile(diff_scores, 95))

    # Convert to arrays
    acc1_all = np.array(acc1_all)
    acc2_all = np.array(acc2_all)
    diff_real = np.array(diff_real)
    diff_perm_95 = np.array(diff_perm_95) if do_permutation else None

    window_sizes_ms = window_ends * 1000

    # --- Plotting ---
    if plot:
        plt.figure(figsize=(6, 4))
        plt.plot(window_sizes_ms, acc1_all, label=labels[0], marker='o')
        plt.plot(window_sizes_ms, acc2_all, label=labels[1], marker='o')
        plt.axhline(0.5, linestyle='--', label='Chance')

        plt.xlabel('Window size (ms)')
        plt.ylabel('Decoding accuracy')
        plt.title('Decoding vs available signal')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        if do_permutation:
            plt.figure(figsize=(6, 4))
            plt.plot(window_sizes_ms, diff_real, label='Real difference', marker='o')
            plt.plot(window_sizes_ms, diff_perm_95, label='95% perm', marker='o')
            plt.axhline(0, linestyle='--', label='No difference')

            plt.xlabel('Window size (ms)')
            plt.ylabel('Accuracy difference')
            plt.title(f'{labels[0]} - {labels[1]}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    return {
        "window_ms": window_sizes_ms,
        "acc1": acc1_all,
        "acc2": acc2_all,
        "diff_real": diff_real,
        "diff_perm_95": diff_perm_95
    }

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
        
def do_brainstem_trial_by_trial_decoding(root_path,n_trials='ntrial_all'):
    ## Trial-by-trial decoding for spa vs. eng: not working well
    subjects_eng=['104','106','107','108','110','111','112','113','118','121','123','124','126','129','133']
    subjects_spa=['203','204','205','206','211','212','213','214','215','220','221','222','223','224','225'] # trim 226 so be n = 15 too
    # subjects_spa=['203','204','205','206','211','212','213','214','215','220','221','222','223','224','225','226'] 
    
    ## SVM showed higher accuracy in trial-by-trial decoding
    clf = make_pipeline(
        StandardScaler(),
        # SVC(kernel='rbf',gamma='auto')  
        SVC(kernel='linear', C=1,class_weight='balanced')
       )
    
    all_score_svm = []

    for se, sp in zip(subjects_eng,subjects_spa):
        ## do eng first because they are the focus 
        epochs_eng_p10 = mne.read_epochs(root_path + 'preprocessed/' + n_trials + '/eng/brainstem_' + se  +'_p10_01_cabr_e_all.fif').pick_types(eeg=True, exclude=[])
        epochs_eng_n40 = mne.read_epochs(root_path + 'preprocessed/' + n_trials + '/eng/brainstem_' + se  +'_n40_01_cabr_e_all.fif').pick_types(eeg=True, exclude=[])
        
        ## could do spa too
        epochs_spa_p10 = mne.read_epochs(root_path + 'preprocessed/' + n_trials + '/spa/brainstem_' + sp  +'_p10_01_cabr_e_all.fif').pick_types(eeg=True, exclude=[])
        epochs_spa_n40 = mne.read_epochs(root_path + 'preprocessed/' + n_trials + '/spa/brainstem_' + sp  +'_n40_01_cabr_e_all.fif').pick_types(eeg=True, exclude=[])
        
        X_p10 = np.squeeze(epochs_eng_p10.get_data())
        X_n40 = np.squeeze(epochs_eng_n40.get_data())

        X = np.concatenate((X_p10,X_n40),axis=0)
        y = np.concatenate((np.repeat(0,len(X_p10)),np.repeat(1,len(X_n40))))
                       
        scores = cross_val_multiscore(clf, X, y, cv=5, n_jobs=4) 
        score = np.mean(scores, axis=0)
        print("Trial-by-trial decoding accuracy: %0.1f%%" % (100 * score,))
        all_score_svm.append(score)
        return all_score_svm

## Signal processing functions
def replace_onset_with_baseline_noise(
        ffr,
        times,
        onset_start,
        onset_end,
        baseline_start=-0.02,
        baseline_end=0.0,
        plot_new = False):
    """
    Replace onset period with prestimulus noise for the mask-peak decoding

    Parameters
    ----------
    ffr : array (subjects, time)
    times : time vector (seconds)
    """

    ffr_new = ffr.copy()

    onset_mask = (times >= onset_start) & (times <= onset_end)
    baseline_mask = (times >= baseline_start) & (times < baseline_end)

    onset_idx = np.where(onset_mask)[0]
    baseline_idx = np.where(baseline_mask)[0]

    n_sub = ffr.shape[0]

    for sub in range(n_sub):

        baseline_noise = np.tile(ffr[sub, baseline_idx],10)

        # sample baseline noise (with replacement)
        replacement = baseline_noise[:len(onset_idx)]
        ffr_new[sub, onset_idx] = replacement

    if plot_new:
        ## plot average FFRs between p10 vs. n40 to check
        plt.figure()
        plt.plot(times,ffr)
        plt.plot(times,ffr_new)

    return ffr_new

def bandpass_filter(data, fs, fmin, fmax, order=4):
    nyq = fs / 2
    b, a = butter(
        order,
        [fmin / nyq, fmax / nyq],
        btype="band"
    )
    return filtfilt(b, a, data, axis=-1)

def bandpower_hilbert(data, fs, f_min, f_max, order=4):

    nyq = fs / 2
    b, a = butter(order, [f_min/nyq, f_max/nyq], btype='band')

    analytic_all = []
    
    for subj in range(data.shape[0]):
        signal = data[subj].astype(np.float32)

        # zero-phase filtering
        filtered = filtfilt(b, a, signal)

        # analytic signal
        analytic = hilbert(filtered)

        # amplitude envelope
        amplitude = np.abs(analytic)

        analytic_all.append(amplitude)

    return np.array(analytic_all)

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
    xx = x / np.linalg.norm(x)
    return xx

def do_xcorr(audio,EEG,fs_audio,fs_eeg,ts,te,level,lag_window_s=(-0.014, -0.007)):
    # Downsample
    num = int((len(audio)*fs_eeg)/fs_audio)    
    audio_rs = signal.resample(audio, num, t=None, axis=0, window=None)
    times_audio = np.linspace(0,len(audio_rs)/fs_eeg,len(audio_rs))
    times_eeg = np.linspace(-0.02,0.2,1101)

    ## p10: noise burst from 0 ms (100th points) ts = 100 # 0.02s (i.e. 0.02s after noise burst) te = 500 # 0.1s
    ## n40: noise burst from 40 ms (200th points) ts = 200 + 100 # 0.06s (i.e. 0.02s after noise burst) te = 650 # 0.13s
    ## p40: noise burst from 0 ms (100th points) ts = 100 # 0.02s te = 650 # 0.13s
    tslice_audio = slice(ff(times_audio, ts), ff(times_audio, te)) ## +1 hack to make sure slice in audio and eeg are the same length
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
            
            ## plot the signal
            fig, axes = plt.subplots(
                3, 1            )

            axes = axes.flatten()
            axes[0].plot(times_audio[tslice_audio], stim)
            axes[1].plot(times_eeg[tslice_EEG],resp_z)
            axes[2].plot(lags_s,xcorr)
            axes[2].axvline(lag_min, color="grey", linestyle="--")
            axes[2].axvline(lag_max, color="grey", linestyle="--")

        return {
            "xcorr_max": np.array(xcorr_max),
            "xcorr_lag_ms": np.array(xcorr_lag)
        }
    else:
        raise ValueError("level must be 'group' or 'individual'")

def do_xcorr_sliding(audio, EEG, fs_audio, fs_eeg,
                     ts, te,
                     win_len_s,
                     step_s,
                     lag_window_s=(-0.014, -0.007)):

    # --- Downsample audio to EEG sampling rate ---
    num = int((len(audio) * fs_eeg) / fs_audio)
    audio_rs = signal.resample(audio, num, axis=0)

    times_audio = np.linspace(0, len(audio_rs)/fs_eeg, len(audio_rs))
    times_eeg = np.linspace(-0.02, 0.2, EEG.shape[1])

    tslice_audio = slice(ff(times_audio, ts), ff(times_audio, te))
    tslice_EEG   = slice(ff(times_eeg, ts), ff(times_eeg, te))

    stim = audio_rs[tslice_audio]
    resp = EEG[:, tslice_EEG]

    # --- Window parameters in samples ---
    win_len = int(win_len_s * fs_eeg)
    step = int(step_s * fs_eeg)

    n_samples = len(stim)
    n_windows = int((n_samples - win_len) / step) + 1

    # --- Lag setup ---
    lags = signal.correlation_lags(win_len, win_len)
    lags_s = lags / fs_eeg

    lag_min, lag_max = lag_window_s
    lag_mask = (lags_s >= lag_min) & (lags_s <= lag_max)

    # --- Storage ---
    xcorr_max = np.zeros((resp.shape[0], n_windows))
    xcorr_lag = np.zeros((resp.shape[0], n_windows))
    window_times = []

    # --- Sliding window loop ---
    for w in range(n_windows):
        start = w * step
        end = start + win_len

        stim_win = stim[start:end]
        stim_z = zscore_and_normalize(stim_win)

        window_times.append((start + end) / 2 / fs_eeg)

        for i, subj_resp in enumerate(resp):
            resp_win = subj_resp[start:end]
            resp_z = zscore_and_normalize(resp_win)

            xcorr = np.abs(signal.correlate(stim_z, resp_z, mode="full"))
            xcorr_win = xcorr[lag_mask]
            lag_win = lags_s[lag_mask]

            idx = np.argmax(xcorr_win)

            xcorr_max[i, w] = xcorr_win[idx]
            xcorr_lag[i, w] = lag_win[idx]

    return np.array(window_times), xcorr_max, xcorr_lag

def mask_with_fixed_noise(data, times, t_end, fixed_noise):
    data_out = data.copy()
    mask = times > t_end
    data_out[:, mask] = fixed_noise[:, mask]
    return data_out

def gen_noise(data,randseed):
    np.random.seed(randseed)
    # estimate noise scale from real data
    noise_std = np.std(data, axis=1, keepdims=True)
    noise_out = np.random.normal(
        loc=0,
        scale=noise_std,
        size=data.shape
    )
    return noise_out
    
#%%####################################### Set path   
root_path='/media/tzcheng/storage2/CBS/'
subjects_dir = '/media/tzcheng/storage2/subjects/'
stc1 = mne.read_source_estimate(root_path + 'cbs_A101/sss_fif/cbs_A101_pa_cabr_morph-vl.stc')
src = mne.read_source_spaces(subjects_dir + 'fsaverage/bem/fsaverage-vol-5-src.fif')
fname_aseg = subjects_dir + 'fsaverage/mri/aparc+aseg.mgz'
label_names = np.asarray(mne.get_volume_labels_from_aseg(fname_aseg))
times = np.linspace(-0.02, 0.2, 1101)

#%%####################################### load the data
file_type = 'morph_roi'
subject_type = 'adults'
fs,std_all = load_CBS_file(file_type, 'p10', subject_type)
fs,dev1_all = load_CBS_file(file_type, 'n40', subject_type)
fs,dev2_all = load_CBS_file(file_type, 'p40', subject_type)
    
## brainstem
file_type = 'sensor'
nfilter = '802000'
ntrial = 'all'
ntop = '0'
fs, p10_eng, n40_eng, p10_spa, n40_spa = load_brainstem_file(file_type, nfilter, ntrial, ntop)

#%%####################################### visualization
## average sensor data
sensor_data = p10_eng.mean(0)
evoked = mne.read_evokeds('/media/tzcheng/storage/Brainstem/brainstem_203/sss_fif/brainstem_203_n40_02_otp_raw_sss_proj_f802000_ntrial200_evoked_ffr.fif')
evoked[0].data = sensor_data
evoked[0].plot_topo()

## source data
source_data = p10_eng.mean(0)
stc1.data = source_data 
stc1.plot(src=src)

## plot individual FFRs: this is the order of how eeg subjects are saved. Refer to preprocessing_brainstem.py
subjects_eng=['113','124','107','110','121','118','126','129','108','106','111','112','133','104','123']
subjects_spa=['203','214','213','223','226','222','212','215','225','224','204','221','206','211','220','205'] ## 202 event code has some issues

subjects_eng_dict = dict(zip(subjects_eng, n40_eng))
subjects_spa_dict = dict(zip(subjects_spa, n40_spa))
n_cols = 3
plot_individuals(subjects_eng_dict,n_cols,times)
plot_individuals(subjects_spa_dict,n_cols,times)

## plot average FFRs between p10 vs. n40
plot_group_ffr(p10_eng, n40_eng, 'p10','n40', times)
plot_group_ffr(p10_spa, n40_spa, 'p10','n40', times)

## plot average FFRs between spa and eng
# will give error if two groups have different sample size
plot_group_ffr(p10_eng, p10_spa, 'eng','spa', times)
plot_group_ffr(n40_eng, n40_spa, 'eng','spa', times)

## plot the FFRs and audio on top of each other
fs_audio, p10_audio = load_CBS_file('audio', 'p10', 'adults')
fs_audio, n40_audio = load_CBS_file('audio', 'n40', 'adults')
plot_audio_ffr(times,p10_audio,fs_audio,p10_spa,0.1)
plot_audio_ffr(times,n40_audio,fs_audio,n40_spa,0.13)

#%%####################################### Subject-by-subject MEG decoding for each condition 
#%%# permuation test to compare p10/n40 vs. p10/p40 decoding in CBS eng

## reduce dimension
# get the mean
std = std_all.mean(axis=1)
dev1 = dev1_all.mean(axis=1)
dev2 = dev2_all.mean(axis=1)
p10_eng = p10_eng_all.mean(axis=1)
n40_eng = n40_eng_all.mean(axis=1)

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
std = std_all[:,rh_ROI_label[nROI],:] 
dev1 = dev1_all[:,rh_ROI_label[nROI],:] 
dev2 = dev2_all[:,rh_ROI_label[nROI],:]
p10_eng = p10_eng_all[:,rh_ROI_label[nROI],:] 
n40_eng = n40_eng_all[:,rh_ROI_label[nROI],:] 

## run decoding
ts = 0
te = 0.2
niter = 1000 # see how the random seed affects accuracy
shuffle = "keep pair"
randseed = 2
ncv = len(std)

## real difference between eng and spa decoding accuracy of p10 vs. n40 sounds
decoding_acc_p10n40 = do_subject_by_subject_decoding([p10_eng, n40_eng], times, ts, te, 14, shuffle, randseed)

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