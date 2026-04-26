#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 16:49:00 2024

Run analysis: SVM decoding, spectral, temporal analysis


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

def make_decode_fn(times, ts, te, ncv1, ncv2, shuffle, randseed):
    def decode_fn(group1, group2):
        acc1 = do_subject_by_subject_decoding(
            group1, times, ts, te, ncv1, shuffle, randseed
        )
        acc2 = do_subject_by_subject_decoding(
            group2, times, ts, te, ncv2, shuffle, randseed
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
    plot
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
subjects_dir = '/media/tzcheng/storage2/subjects/'
times = np.linspace(-0.02, 0.2, 1101)

#%%####################################### load the data
## CBS
file_type = 'EEG'
subject_type = 'adults'
fs,std = load_CBS_file(file_type, 'p10', subject_type)
fs,dev1 = load_CBS_file(file_type, 'n40', subject_type)
fs,dev2 = load_CBS_file(file_type, 'p40', subject_type)

## brainstem
file_type = 'EEG'
ntrial = 'all'
fs, p10_eng, n40_eng, p10_spa, n40_spa = load_brainstem_file(file_type, ntrial)
    
#%%####################################### visualize the data to examine
## plot individual FFRs: this is the order of how eeg subjects are saved. Refer to preprocessing_brainstem.py
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
# will give error if two groups have different sample size
plot_group_ffr(p10_eng, p10_spa, 'eng','spa', times)
plot_group_ffr(n40_eng, n40_spa, 'eng','spa', times)

## plot the FFRs and audio on top of each other
fs_audio, p10_audio = load_CBS_file('audio', 'p10', 'adults')
fs_audio, n40_audio = load_CBS_file('audio', 'n40', 'adults')
plot_audio_ffr(times,p10_audio,fs_audio,p10_spa,0.1)
plot_audio_ffr(times,n40_audio,fs_audio,n40_spa,0.13)

#%%####################################### trial-by-trial EEG decoding for each individual of brainstem dataset
root_path='/media/tzcheng/storage2/CBS/'
trial_by_trial_decoding_acc = do_CBS_trial_by_trial_decoding(root_path,n_trials=200)

root_path= '/media/tzcheng/storage/Brainstem/EEG/'
comparison = 'Eng/Spa' ## 'Eng/Spa' or 'p10/n40'
trial_by_trial_decoding_acc = do_brainstem_trial_by_trial_decoding(root_path)

#%%####################################### Subject-by-subject EEG decoding brainstem dataset
#%% One-shot decoding
## modify ts, te to do C and V section decoding [0, 40] ba C section
## for ba, 10 ms + 90 ms = 100 ms; for mba, 40 ms + 90 ms = 130 ms
## epoch length -20 to 200 ms with sampling rate at 5000 Hz
## Run with one random seed 2
ts = 0
te = 0.2
ncv = 15
randseed = 2

## brainstem
decoding_acc = do_subject_by_subject_decoding([p10_eng, n40_eng], times, ts, te, len(p10_eng), 'keep pair', randseed)
decoding_acc = do_subject_by_subject_decoding([p10_spa, n40_spa], times, ts, te, len(p10_spa), 'keep pair', randseed)
decoding_acc = do_subject_by_subject_decoding([p10_eng, p10_spa], times, ts, te, len(p10_eng), 'full', randseed)
decoding_acc = do_subject_by_subject_decoding([n40_eng, n40_spa], times, ts, te, len(p10_eng), 'full', randseed)

## CBS
# 3-way decoding: ba vs. pa vs. mba
scores_ba_mba_pa = do_subject_by_subject_decoding([std,dev1,dev2], times, ts, te, 18, 'keep pair', randseed)
# 2-way decoding: ba vs. pa, ba vs. mba, pa vs. mba
scores_ba_mba = do_subject_by_subject_decoding([std,dev1], times, ts, te, 18, 'keep pair', randseed)
scores_ba_pa = do_subject_by_subject_decoding([std,dev2], times, ts, te, 18, 'keep pair', randseed)
scores_mba_pa = do_subject_by_subject_decoding([dev1,dev2], times, ts, te, 18, 'keep pair', randseed)

#%% Random-seed decoding
# only work when the shuffle is set to "full", "keep pair" always generate the same decoding accuracy
ts = 0
te = 0.20
niter = 500 
shuffle = "keep pair"

# randomly select the same number of spanish speakers to match english speakers
p10_spa_match = random_select(p10_spa,len(p10_eng),1)
n40_spa_match = random_select(n40_spa,len(p10_eng),1)

condition_pairs = (
    [p10_eng, n40_eng],
    [p10_spa, n40_spa]
)
scores_all = run_random_seed_decoding(condition_pairs, times, ts, te, niter, shuffle, plot=True)

#%% Differential decoding
ts = 0
te = 0.20
niter = 500 
shuffle = "keep pair"
randseed = 2
ncv1 = 18
ncv2 = 18

condition_pairs = (
    [std, dev1],
    [std, dev2]
)

# condition_pairs = (
#     [p10_eng, n40_eng],
#     [p10_spa, n40_spa]
# )

decode_fn = make_decode_fn(times, ts, te, ncv1, ncv2, shuffle, randseed)
real_diff, diff_perm, fig, ax = permutation_decoding_diff(condition_pairs,decode_fn,niter, rng=None, plot=True, verbose=True)

#%% Sliding-time decoding
## if making the window very short then it become the point-by-point decoding
ts_global = 0
te_global = 0.2
window_length = 0.0002      # window
window_step = 0.0002         # ms step
shuffle = "keep pair"
randseed = 2
ncv_eng = 15
ncv_spa = 16

run_sliding_time_decoding(p10_eng,n40_eng,p10_spa,n40_spa,times,ts_global,te_global,window_length,window_step,ncv_eng,ncv_spa,shuffle,randseed,plot=True)

#%% Mask-peak decoding
# mask onset peaks of the signal to see decoding change
p10_eng_new = replace_onset_with_baseline_noise(p10_eng,times,onset_start=0.005,onset_end=0.04)
p10_spa_new = replace_onset_with_baseline_noise(p10_spa,times,onset_start=0.005,onset_end=0.04)

decoding_acc = do_subject_by_subject_decoding([p10_eng, n40_eng], times, ts, te, len(p10_eng), 'keep pair', randseed)
decoding_acc = do_subject_by_subject_decoding([p10_eng_new, n40_eng], times, ts, te, len(p10_eng), 'keep pair', randseed)
decoding_acc = do_subject_by_subject_decoding([p10_spa, n40_spa], times, ts, te, len(p10_eng), 'keep pair', randseed)
decoding_acc = do_subject_by_subject_decoding([p10_spa_new, n40_spa], times, ts, te, len(p10_eng), 'keep pair', randseed)

#%%# Increment decoding (increment in 5 ms) to test if the decoding accuracy jump or linear increase in eng vs. spa
window_step = 0.005

run_increment_decoding(
    std, dev1,
    std, dev2,
    # p10_eng, n40_eng,
    # p10_spa, n40_spa,
    times,
    window_step=window_step,
    ncv1=18,
    ncv2=18,
    shuffle="keep pair",
    randseed=2,
    do_permutation=None,
    niter=500,
    labels=("English", "Spanish"),
    plot = True
)

#%%# Increment decoding (increment in 5 ms) to test if the decoding accuracy jump or linear increase in p10n40 and p10p40
run_increment_decoding(
    std, dev1,
    std, dev2,
    times,
    ncv1=18,
    ncv2=18,
    shuffle="keep pair",
    randseed=2,
    do_permutation=None,
    niter=500,
    labels=("p10/n40", "p10/p40"),
    plot = True
)

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

#%%####################################### Spectrogram analysis for group
signal = p10_eng.mean(0)
signal = signal.astype(np.float32)
signal = signal/np.max(np.abs(signal))

nfft = fs
S = librosa.stft(
    signal,
    n_fft = nfft,
    hop_length = int(nfft*(0.04-0.039)), # Python hop_length ≠ MATLAB overlap
    win_length = int(nfft*0.04),
    window = 'hamm')
S_magnitude = np.abs(S)
S_db = librosa.amplitude_to_db(np.abs(S),ref=np.max)
freqs = librosa.fft_frequencies(sr=fs, n_fft=fs)
times = librosa.times_like(S, sr=fs, hop_length=int(nfft*(0.04-0.039)))

## Plot the spectrogram
fig, ax = plt.subplots(figsize=(10, 4))
im = librosa.display.specshow(
    S_db,
    sr=fs,
    hop_length=int(nfft*(0.04-0.039)),
    x_axis='time',
    y_axis='hz',
    cmap='jet',
    ax=ax,
    vmin=-20,
    vmax=0   
)
ax.set_title('Spectrogram')
ax.set_ylim([0,800])
fig.colorbar(im, ax=ax, format="%+2.0f dB")
plt.show()

f_min, f_max = 90, 110
freq_mask = (freqs >= f_min) & (freqs <= f_max)
S_band = S_db[freq_mask, :]

plt.figure()
plt.plot(times,S_magnitude[freq_mask, :].mean(0))

#%%####################################### Spectrogram analysis for individuals
EEG = p10_eng

nfft = fs
wl = 0.02 # window length
wo = 0.019 # window overlap
hop = int(nfft * (wl - wo))
win_len = int(nfft * wl)

all_S_mag = []
all_S_db = []

for subj in range(EEG.shape[0]):

    signal = EEG[subj].astype(np.float32)
    signal = signal / np.max(np.abs(signal))

    S = librosa.stft(
        signal,
        n_fft=nfft,
        hop_length=hop,
        win_length=win_len,
        window='hamm'
    )

    S_mag = np.abs(S)
    S_db  = librosa.amplitude_to_db(S_mag, ref=np.max)
    
    ## Plot the spectrogram for each individual
    # fig, ax = plt.subplots(figsize=(10, 4))
    # im = librosa.display.specshow(
    #     S_db,
    #     sr=fs,
    #     hop_length=int(nfft*(0.04-0.039)),
    #     x_axis='time',
    #     y_axis='hz',
    #     cmap='jet',
    #     ax=ax,
    #     vmin=-20,
    #     vmax=0   
    # )
    # ax.set_title('Spectrogram')
    # ax.set_ylim([0,800])
    # fig.colorbar(im, ax=ax, format="%+2.0f dB")
    # plt.show()
    
    all_S_mag.append(S_mag)
    all_S_db.append(S_db)

all_S_mag = np.array(all_S_mag)   # shape: (subjects, freqs, times)
all_S_db  = np.array(all_S_db)

freqs = librosa.fft_frequencies(sr=fs, n_fft=nfft)
times = librosa.times_like(S, sr=fs, hop_length=hop)

f_min, f_max = 90,110
freq_mask = (freqs >= f_min) & (freqs <= f_max)

band_power = all_S_mag[:, freq_mask, :].mean(axis=1)

fig, ax = plt.subplots(figsize=(10, 4))
im = librosa.display.specshow(
    all_S_mag.mean(0),
    sr=fs,
    hop_length=int(nfft*(0.04-0.039)),
    x_axis='time',
    y_axis='hz',
    cmap='jet',
    ax=ax
)
ax.set_title('Spectrogram')
ax.set_ylim([0,800])
fig.colorbar(im, ax=ax, format="%+2.0f dB")
plt.show()
    
#%% Alternatively just use hilbert for narrow band power extraction
EEG = p10_spa

# Amount of padding (e.g., 200 ms on each side)
n_pad =0.5 # in s
pad = int(n_pad * fs)   
# Reflect padding along time axis
EEG_padded = np.pad(
    EEG,
    pad_width=((0, 0), (pad, pad)),   # no pad on subjects, pad time
    mode='reflect'
)

f_min, f_max = 50,150
band_power_full = bandpower_hilbert(EEG_padded, fs, f_min, f_max)
times = np.linspace(-0.02, 0.2, 1101)

# Remove padding
band_power = band_power_full[:, pad:-pad]

#%% Visualization
mean_band = band_power.mean(axis=0)
sem_band  = band_power.std(axis=0) / np.sqrt(band_power.shape[0])

plt.figure(figsize=(8,5))

# --- Individual subjects ---
for subj in range(band_power.shape[0]):
    plt.plot(times, band_power[subj],
             color='magenta',
             alpha=0.3,
             linewidth=1)

# --- Group mean (THICK line) ---
plt.plot(times, mean_band,
         color='red',
         linewidth=3,
         label='Group Mean')

# --- Optional SEM shading ---
plt.fill_between(times,
                 mean_band - sem_band,
                 mean_band + sem_band,
                 color='red',
                 alpha=0.2,
                 label='±SEM')

plt.xlabel("Time (s)")
plt.ylabel("Magnitude")
plt.title(f"{f_min}-{f_max} Hz Band Magnitude")
plt.legend()
plt.tight_layout()
plt.show()

#%% Decode F0
f_min, f_max = 50,150
ts = 0
te = 0.2
niter = 500 
shuffle = "keep pair"
randseed = 2

band_power_p10_eng = bandpower_hilbert(p10_eng, fs, f_min, f_max)
band_power_n40_eng = bandpower_hilbert(n40_eng, fs, f_min, f_max)
band_power_p10_spa = bandpower_hilbert(p10_spa, fs, f_min, f_max)
band_power_n40_spa = bandpower_hilbert(n40_spa, fs, f_min, f_max)

decoding_acc_eng = do_subject_by_subject_decoding([band_power_p10_eng, band_power_n40_eng], times, ts, te, 15, "keep pair", None)
decoding_acc_spa = do_subject_by_subject_decoding([band_power_p10_spa, band_power_n40_spa], times, ts, te, 16, "keep pair", None)
decoding_acc_p10 = do_subject_by_subject_decoding([band_power_p10_eng, band_power_p10_spa], times, ts, te, 15, "full", None)
decoding_acc_n40 = do_subject_by_subject_decoding([band_power_n40_eng, band_power_n40_spa], times, ts, te, 15, "full", None)

condition_pairs = (
    [band_power_p10_eng, band_power_n40_eng],
    [band_power_p10_spa, band_power_n40_spa]
)

decode_fn = make_decode_fn(times, ts, te,len(condition_pairs[0][0]), len(condition_pairs[1][0]), shuffle, randseed)
real_diff, diff_perm, fig, ax = permutation_decoding_diff(condition_pairs,decode_fn,niter, rng=None, plot=True, verbose=True)

window_step = 0.005

run_increment_decoding(
    band_power_p10_eng, band_power_n40_eng,
    band_power_p10_spa, band_power_n40_spa,
    times,
    window_step=window_step,
    ncv1=15,
    ncv2=16,
    shuffle="keep pair",
    randseed=2,
    do_permutation=None,
    niter=500,
    labels=("English", "Spanish"),
    plot = True
)

#%% Decode spectrogram
nfft = fs
wl = 0.02 # window length
wo = 0.019 # window overlap
hop = int(nfft * (wl - wo))
win_len = int(nfft * wl)

## calculate spectrogram
p10_eng_S_mag = []
p10_eng_S_db = []
n40_eng_S_mag = []
n40_eng_S_db = []
p10_spa_S_mag = []
p10_spa_S_db = []
n40_spa_S_mag = []
n40_spa_S_db = []

## p10_eng
for subj in range(p10_eng.shape[0]):
    signal = p10_eng[subj].astype(np.float32)
    signal = signal / np.max(np.abs(signal))
    S = librosa.stft(
        signal,
        n_fft=nfft,
        hop_length=hop,
        win_length=win_len,
        window='hamm'
    )
    S_mag = np.abs(S)
    S_db  = librosa.amplitude_to_db(S_mag, ref=np.max)
    p10_eng_S_mag.append(S_mag)
    p10_eng_S_db.append(S_db)

## n40_eng
for subj in range(n40_eng.shape[0]):
    signal = n40_eng[subj].astype(np.float32)
    signal = signal / np.max(np.abs(signal))
    S = librosa.stft(
        signal,
        n_fft=nfft,
        hop_length=hop,
        win_length=win_len,
        window='hamm'
    )
    S_mag = np.abs(S)
    S_db  = librosa.amplitude_to_db(S_mag, ref=np.max)
    n40_eng_S_mag.append(S_mag)
    n40_eng_S_db.append(S_db)

## p10_spa   
for subj in range(p10_spa.shape[0]):
    signal = p10_spa[subj].astype(np.float32)
    signal = signal / np.max(np.abs(signal))
    S = librosa.stft(
        signal,
        n_fft=nfft,
        hop_length=hop,
        win_length=win_len,
        window='hamm'
    )
    S_mag = np.abs(S)
    S_db  = librosa.amplitude_to_db(S_mag, ref=np.max)
    p10_spa_S_mag.append(S_mag)
    p10_spa_S_db.append(S_db)

## n40_spa
for subj in range(n40_spa.shape[0]):
    signal = n40_spa[subj].astype(np.float32)
    signal = signal / np.max(np.abs(signal))
    S = librosa.stft(
        signal,
        n_fft=nfft,
        hop_length=hop,
        win_length=win_len,
        window='hamm'
    )
    S_mag = np.abs(S)
    S_db  = librosa.amplitude_to_db(S_mag, ref=np.max)
    n40_spa_S_mag.append(S_mag)
    n40_spa_S_db.append(S_db)

## 
p10_eng_S_mag = np.asarray(p10_eng_S_mag)
n40_eng_S_mag = np.asarray(n40_eng_S_mag)
p10_spa_S_mag = np.asarray(p10_spa_S_mag)
n40_spa_S_mag = np.asarray(n40_spa_S_mag)
p10_eng_flat = p10_eng_S_mag.reshape(p10_eng_S_mag.shape[0],-1)
n40_eng_flat = n40_eng_S_mag.reshape(n40_eng_S_mag.shape[0],-1)
p10_spa_flat = p10_spa_S_mag.reshape(p10_spa_S_mag.shape[0],-1)
n40_spa_flat = n40_spa_S_mag.reshape(n40_spa_S_mag.shape[0],-1)

decoding_acc = do_subject_by_subject_decoding([p10_eng_flat, n40_eng_flat], times, ts, te, len(p10_eng), 'keep pair', randseed)
decoding_acc = do_subject_by_subject_decoding([p10_spa_flat, n40_spa_flat], times, ts, te, len(p10_spa), 'keep pair', randseed)

#%%
window_step = 0.005

run_increment_decoding(
    p10_eng_flat, n40_eng_flat,
    p10_spa_flat, n40_spa_flat,
    times,
    window_step=window_step,
    ncv1=15,
    ncv2=16,
    shuffle="keep pair",
    randseed=2,
    do_permutation=None,
    niter=500,
    labels=("English", "Spanish"),
    plot = True
)    


#%%####################################### SNR analysis
level = 'individual' # 'group' (mean first then SNR) or 'individual' (SNR on individual level then mean)

data = n40_eng

ts = 0
te = 0.2 # 100 for ba and 130 for mba and pa
fmin = 50
fmax = 150

SNR_t, SNR_s = do_SNR(data,times,ts,te,fmin, fmax,level)
print('temporal SNR: ' + str(np.array(SNR_t).mean()) + '(' + str(np.array(SNR_t).std()/np.sqrt(len(data))) +')')
print('spectral SNR: ' + str(np.array(SNR_s).mean()) + '(' + str(np.array(SNR_s).std()/np.sqrt(len(data))) +')')

#%%####################################### xcorr analysis
level = 'individual'

# lh_ROI_label = [12, 72,76,74] # [subcortical] brainstem,[AC] STG, transversetemporal, [controls] frontal pole
# rh_ROI_label = [12, 108,112,110] # [subcortical] brainstem,[AC] STG, transversetemporal, [controls] frontal pole
# nROI = 2
# std = std_all[:,rh_ROI_label[nROI],:] 
# dev1 = dev1_all[:,rh_ROI_label[nROI],:] 
# dev2 = dev2_all[:,rh_ROI_label[nROI],:]

# p10_eng = p10_eng_all[:,rh_ROI_label[nROI],:] 
# n40_eng = n40_eng_all[:,rh_ROI_label[nROI],:] 

ts = 0 # 0.02 for ba and pa, 0.06 for mba
te = 0.04 # 0.1 for ba and 0.13 for mba and pa, this is hard cut off because audio files are this long

fs_audio, p10_audio = load_CBS_file('audio', 'p10', 'adults')
fs_audio, n40_audio = load_CBS_file('audio', 'n40', 'adults')
fs_audio, p40_audio = load_CBS_file('audio', 'p40', 'adults')
fs_eeg, p10_eng, n40_eng, p10_spa, n40_spa = load_brainstem_file(file_type, ntrial)

audio = n40_audio
EEG = dev1
audio = stats.zscore(audio)
EEG = stats.zscore(EEG,axis=-1)

xcorr = do_xcorr(audio,EEG,fs_audio,fs_eeg,ts,te,level)

print(np.mean(xcorr['xcorr_max']))
print(np.mean(xcorr['xcorr_lag_ms']))
xcorr3 = xcorr['xcorr_max']
lag3 = xcorr['xcorr_lag_ms']

# stats.ttest_rel(xcorr1,xcorr3)

## xcorr table
# print(xcorr)
print(np.mean(xcorr['xcorr_max']))
print(np.mean(xcorr['xcorr_lag_ms']))

## run stats ANOVA
import pingouin as pg
import pandas as pd

df = pd.read_csv('/home/tzcheng/Downloads/FFR_xcorr_window2 - Brainstem - 3000.csv')
df = pd.read_csv('/home/tzcheng/Downloads/FFR_xcorr_window2 - CBS.csv')

aov = pg.mixed_anova(
    dv='xcorr_lags',
    within='condition',
    between='group',
    subject='subject',
    data=df
)
print(aov)

## run stats paired and independent ttest
df.groupby(['condition', 'group'])['xcorr_coef'].mean()
df.groupby(['condition', 'group'])['xcorr_lags'].mean()*1000

condition1 = df.loc[(df["condition"] == 'p10') & (df["group"]== 'eng')]['xcorr_coef'].values
condition2 = df.loc[(df["condition"] == 'n40') & (df["group"]== 'eng')]['xcorr_coef'].values
diff_eng = condition2-condition1
stats.ttest_rel(condition1,condition2)
condition1 = df.loc[(df["condition"] == 'p10') & (df["group"]== 'spa')]['xcorr_coef'].values
condition2 = df.loc[(df["condition"] == 'n40') & (df["group"]== 'spa')]['xcorr_coef'].values
diff_spa = condition2-condition1

stats.ttest_rel(condition1,condition2)

group1 = df.loc[(df["condition"] == 'p10') & (df["group"]== 'eng')]['xcorr_lags'].values
group2 = df.loc[(df["condition"] == 'p10') & (df["group"]== 'spa')]['xcorr_lags'].values
stats.ttest_ind(group1,group2)
group1 = df.loc[(df["condition"] == 'n40') & (df["group"]== 'eng')]['xcorr_lags'].values 
group2 =  df.loc[(df["condition"] == 'n40') & (df["group"]== 'spa')]['xcorr_lags'].values
stats.ttest_ind(group1,group2)

## compare the diff of p10n40 between groups
group1 = df.loc[(df["condition"] == 'n40') & (df["group"]== 'eng')]['xcorr_coef'].values - df.loc[(df["condition"] == 'p10') & (df["group"]== 'eng')]['xcorr_coef'].values 
group2 =  df.loc[(df["condition"] == 'n40') & (df["group"]== 'spa')]['xcorr_coef'].values - df.loc[(df["condition"] == 'p10') & (df["group"]== 'spa')]['xcorr_coef'].values
stats.ttest_ind(group1,group2)

## implement permutation test
import numpy as np
from scipy.stats import permutation_test

def statistic(x, y, axis=0):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)

result = permutation_test((condition1, condition2), statistic, n_resamples=1000, alternative='two-sided')
print(f"condition1 Mean: {np.mean(condition1):.2f}")
print(f"condition2 Mean: {np.mean(condition2):.2f}")
print(f"Observed Difference: {result.statistic:.2f}")
print(f"P-value: {result.pvalue}")

#%% Verhulst modeled FFR analysis
p10_model_FFR = sp.io.loadmat('/home/tzcheng/Downloads/Verhulstetal2018Model-master/p10_FFR.mat')
p40_model_FFR = sp.io.loadmat('/home/tzcheng/Downloads/Verhulstetal2018Model-master/p40_FFR.mat')
n40_model_FFR = sp.io.loadmat('/home/tzcheng/Downloads/Verhulstetal2018Model-master/n40_FFR.mat')
p10_model_FFR = np.squeeze(p10_model_FFR["EFR"])
p40_model_FFR = np.squeeze(p40_model_FFR["EFR"])
n40_model_FFR = np.squeeze(n40_model_FFR["EFR"])
fs_eeg = 8820
t_model = np.linspace(0,len(n40_model_FFR)/fs_eeg,len(n40_model_FFR))

#### SVM decoding augmented data
def add_latency_jitter(signal, rng, max_jitter_samples=5):
    shift = rng.integers(-max_jitter_samples, max_jitter_samples + 1)
    return np.roll(signal, shift)
def add_amplitude_jitter(signal, rng, scale_std=0.1):
    scale = 1 + rng.normal(0, scale_std)
    return signal * scale
def pink_noise(n, rng):        
    X = rng.standard_normal(n)
    X = np.fft.rfft(X)
    
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1  # avoid divide by zero
    
    X /= np.sqrt(freqs)  # 1/f
    
    return np.fft.irfft(X, n)
def simulate_ffr_trial(EFR, rng, irregularities):
    # latency jitter
    trial = add_latency_jitter(EFR, max_jitter_samples=5, rng=rng)   
    # amplitude jitter
    trial = add_amplitude_jitter(trial, scale_std=0.1, rng=rng)
    # correlated noise
    noise = irregularities * np.std(EFR) * pink_noise(len(EFR), rng)
    return trial + noise

## prepare the 18 trials
n_trials = 18
trials_p10_EFR = np.zeros((n_trials, len(n40_model_FFR)))
trials_p40_EFR = np.zeros((n_trials, len(n40_model_FFR)))
trials_n40_EFR = np.zeros((n_trials, len(n40_model_FFR)))

rng = np.random.default_rng(2)
irregularities = 0.9

for i in range(n_trials):
    temp_p10 = simulate_ffr_trial(p10_model_FFR,rng,irregularities)
    trials_p10_EFR[i, :] = zero_pad(temp_p10,len(n40_model_FFR)) # pad the p10 to be the same length as p40 and n40
    trials_p40_EFR[i, :] = simulate_ffr_trial(p40_model_FFR,rng,irregularities)
    trials_n40_EFR[i, :] = simulate_ffr_trial(n40_model_FFR,rng,irregularities)

## visualize the trials
plt.figure()
plt.plot(t_model,trials_p10_EFR.transpose())
plt.plot(t_model,trials_p10_EFR.mean(0),linewidth = 2, color = 'k')
plt.title('p10 modeled FFR')
plt.figure()
plt.plot(t_model,trials_p40_EFR.transpose())
plt.plot(t_model,trials_p40_EFR.mean(0),linewidth = 2, color = 'k')
plt.title('p40 modeled FFR')
plt.figure()
plt.plot(t_model,trials_n40_EFR.transpose())
plt.plot(t_model,trials_n40_EFR.mean(0),linewidth = 2, color = 'k')
plt.title('n40 modeled FFR')

#%% Verhulst modeled FFR SVM decoding
ts = 0
te = 0.1

## One-shot decoding
decoding_acc_p10n40 = do_subject_by_subject_decoding([trials_p10_EFR, trials_n40_EFR], t_model, ts, te, n_trials, shuffle, None)
decoding_acc_p10p40 = do_subject_by_subject_decoding([trials_p10_EFR, trials_p40_EFR], t_model, ts, te, n_trials, shuffle, None)
 
## Random-seed decoding
shuffle = "full"
niter = 500 
randseed = 2
p10_spa_match = random_select(p10_spa,len(p10_eng),1)
n40_spa_match = random_select(n40_spa,len(p10_eng),1)

condition_pairs = (
    [p10_eng, n40_eng],
    [p10_spa, n40_spa]
)
scores_all = run_random_seed_decoding(condition_pairs, times, ts, te, niter, shuffle, plot=True)

## Differential decoding
ts = 0
te = 0.1
niter = 500
shuffle = "keep pair"
randseed = 2

condition_pairs = (
    [trials_p10_EFR, trials_n40_EFR],
    [trials_p10_EFR, trials_p40_EFR]
)

decode_fn = make_decode_fn(t_model, ts, te, n_trials,n_trials, shuffle, randseed)
real_diff, diff_perm, fig, ax = permutation_decoding_diff(condition_pairs,decode_fn,niter, rng=None, plot=True, verbose=True)

## Increment decoding
window_step = 0.005

run_increment_decoding(
    trials_p10_EFR, trials_n40_EFR,
    trials_p10_EFR, trials_p40_EFR,
    t_model,
    ncv1=18,
    ncv2=18,
    shuffle="keep pair",
    randseed=2,
    do_permutation=True,
    niter=500,
    labels=("p10/n40", "p10/p40"),
    plot = True
)

#%% Verhulst modeled FFR xcorr
ts = 0.06 # 0.02 for ba and pa, 0.06 for mba
te = 0.13 # 0.1 for ba and 0.13 for mba and pa, this is hard cut off because audio files are this long
lag_window_s=(-0.012, -0.007) # make it even shorter than the real FFR xcorr (-0.014, -0.007)

fs_audio, p10_audio = load_CBS_file('audio', 'p10', 'adults')
fs_audio, n40_audio = load_CBS_file('audio', 'n40', 'adults')
fs_audio, p40_audio = load_CBS_file('audio', 'p40', 'adults')

audio = p40_audio
EEG = np.squeeze(p40_model_FFR["EFR"])
audio = stats.zscore(audio)
EEG = stats.zscore(EEG)

plt.figure()
plt.plot(t_model,EEG)
plt.ylim([-3e-7, 5e-7])

num = int((len(audio)*fs_eeg)/fs_audio)    
audio_rs = signal.resample(audio, num, t=None, axis=0, window=None)
t = np.linspace(0,len(audio_rs)/fs_eeg,len(audio_rs))

## p10: noise burst from 0 ms (100th points) ts = 100 # 0.02s (i.e. 0.02s after noise burst) te = 500 # 0.1s
## n40: noise burst from 40 ms (200th points) ts = 200 + 100 # 0.06s (i.e. 0.02s after noise burst) te = 650 # 0.13s
## p40: noise burst from 0 ms (100th points) ts = 100 # 0.02s te = 650 # 0.13s
tslice = slice(ff(t, ts), ff(t, te)) ## +1 hack to make sure slice in audio and eeg are the same length

stim = audio_rs[tslice]
resp = EEG[tslice]
lags = signal.correlation_lags(len(stim),len(resp))
lags_s = lags/fs_eeg

lag_min, lag_max = np.array(lag_window_s)
lag_mask = (lags_s >= lag_min) & (lags_s <= lag_max)

stim_z = zscore_and_normalize(stim)
resp_z = zscore_and_normalize(resp)
xcorr = signal.correlate(stim_z,resp_z,mode='full')
xcorr = abs(xcorr)

xcorr_win = xcorr[lag_mask]
lag_win = lags_s[lag_mask]
max_idx = np.argmax(xcorr_win)

## plot the signal
fig, axes = plt.subplots(
    3, 1)

axes = axes.flatten()
axes[0].plot(t[tslice], stim_z)
axes[1].plot(t[tslice],resp_z)
axes[2].plot(lags_s,xcorr)
axes[2].axvline(lag_min, color="grey", linestyle="--")
axes[2].axvline(lag_max, color="grey", linestyle="--")

print("xcorr_max: " + str(xcorr_win[max_idx]))
print("xcorr_lag_ms: " + str(lag_win[max_idx]))