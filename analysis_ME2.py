#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:14:44 2024

Streamline processing the MEG time series data to generate and save the SSEP, connectivity, ERSP, decoding results. 
Input: MEG time series on the sensor, ROI (see the redo_ROI to further select ROI), whole brain time series (.npy files) in the MEG data folder
Output: save the analysis results (.npy or mne formats) in the SSEP, ERSP, connectivity, decoding folders

Currently don't have the computing power to run ERSP and conn on the whole brain data, but could do SSEP and decoding.
 
@author: tzcheng
"""

#%%####################################### Import library  
import numpy as np
import random
import mne
from mne.decoding import cross_val_multiscore
from mne_connectivity import spectral_connectivity_time, read_connectivity
import os
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

#%%####################################### Define functions
def do_SSEP(data, f_name, fmin, fmax, MEG_fs):  
    ## compute psds from the time series with mne.time_frequency.psd_array_welch function
    # data: input MEG time series (subjects, sources, times) 
    # f_name: input name for saving the file
    # fmin: min frequency for the psd output
    # fmax: max frequency for the psd output
    # MEG_fs: MEG (re)sampling rate
    psds, freqs = mne.time_frequency.psd_array_welch(
    data,MEG_fs, # could replace with label time series
    n_fft=np.shape(data)[2],
    n_overlap=0,
    n_per_seg=None,
    fmin=fmin,
    fmax=fmax,)
    np.savez(root_path + 'SSEP/' + f_name + '_psds.npz', psds=psds,freqs=freqs)
    return psds

def do_ERSP(data, f_name, fmin, fmax, f_step, MEG_fs,n_cycles,baseline,output):  
    ## compute time-frequency decompositions from the time series with mne.time_frequency.tfr_array_morlet function
    # data: input MEG time series (subjects, sources, times), supposed to be trial-by-trial time series in order to compute induced responses instead of evoked responses
    # f_name: input name for saving the file
    # fmin: min frequency for the ERSP output
    # fmax: max frequency for the ERSP output
    # MEG_fs: MEG (re)sampling rate
    # n_cycles, baseline, output: check the tfr_array_morlet function for other parameters, otherwise use default
    times = np.linspace(-0.5,10.5,np.shape(data)[-1]) # might need to change if redo epoching
    freqs = np.arange(fmin,fmax,f_step)
    tfr = mne.time_frequency.tfr_array_morlet(data,MEG_fs,freqs=freqs,n_cycles=n_cycles,output=output)
    if baseline == 'ratio':
        tfr /= np.mean(tfr,axis=-1, keepdims=True)
    elif baseline == 'percent':
        tfr /= np.mean(tfr,axis=-1, keepdims=True)
        tfr -= 1 
    np.savez(root_path + 'ERSP/' + f_name + '_bc_' + baseline + '_' + output + '.npz', tfr=tfr,times=times,freqs=freqs)
    return tfr,times,freqs
    
def do_connectivity(data, f_name, fmin, fmax, f_step, MEG_fs, directional):  
    ## compute connectivity using spectral_connectivity_time function
    # data: input MEG time series (subjects, sources, times)
    # f_name: input name for saving the file
    # fmin: min frequency for the conn output
    # fmax: max frequency for the conn output
    # f_step: freq resolution between each freq band
    # MEG_fs: MEG (re)sampling rate
    # directional: directional (GC) and non-directional (PLV, PLI, Coh, etc.)
    freqs = np.linspace(fmin,fmax,f_step)
    if directional:
        con_methods = ["gc"]
        con = spectral_connectivity_time(
            data,
            # indices = (np.array([[2, 3], [2, 3], [7, 26], [8,27], [0, 1], [71, 107],[2, 3],[2, 3]]),  # seeds
            # np.array([[0, 1], [71, 107],[2, 3],[2, 3],[2, 3], [2, 3], [7, 26], [8,27]])),  # targets
            indices = (np.array([[0]]),np.array([[1]])),  # Auditory to Motor
            freqs = freqs,
            method=con_methods,
            mode="multitaper", # if using cwt_morlet, add cwt_freqs = nfreq = np.array([1,2,3,4,5])
            sfreq=MEG_fs,
            fmin=fmin,
            fmax=fmax,
            faverage=False,
            n_jobs=1,)
        con.save(root_path + 'connectivity/' + f_name + '_conn_GC_AM')
        con = spectral_connectivity_time(
            data,
            # indices = (np.array([[2, 3], [2, 3], [7, 26], [8,27], [0, 1], [71, 107],[2, 3],[2, 3]]),  # seeds
            # np.array([[0, 1], [71, 107],[2, 3],[2, 3],[2, 3], [2, 3], [7, 26], [8,27]])),  # targets
            indices = (np.array([[1]]),np.array([[0]])),  # Motor to Auditory
            freqs = freqs,
            method=con_methods,
            mode="multitaper", # if using cwt_morlet, add cwt_freqs = nfreq = np.array([1,2,3,4,5])
            sfreq=MEG_fs,
            fmin=fmin,
            fmax=fmax,
            faverage=False,
            n_jobs=1,)
        con.save(root_path + 'connectivity/' + f_name + '_conn_GC_MA')
    else:
        con_methods = ["plv","coh","pli"]
        con = spectral_connectivity_time( # Compute frequency- and time-frequency-domain connectivity measures
        data,
        freqs=freqs,
        method=con_methods,
        mode="multitaper", # if using cwt_morlet, add cwt_freqs = nfreq = np.array([1,2,3,4,5])
        sfreq=MEG_fs,
        fmin=fmin,
        fmax=fmax,
        faverage=False,
        n_jobs=1,)
        con[0].save(root_path + 'connectivity/' + f_name + '_conn_plv')
        con[1].save(root_path + 'connectivity/' + f_name + '_conn_coh')
        con[2].save(root_path + 'connectivity/' + f_name + '_conn_pli')
    return con

def do_decoding(X1, X2, ts, te, model, seed):  
    ## compute ML-based decoding using scikit learn functions
    # X1: condition 1 e.g. MMR1 or ME2 random condtion
    # X2: condition 2 e.g. MMR2 or ME2 duple condtion
    # ts: time start to be included in the decoding 
    # te: time end to be included in the decoding
    # model: ML model used for decoding
    # seed: random seed to randomize the initial X,y order for replication purpose
    all_score = []
    ## Two way classification using ovr
    X = np.concatenate((X1,X2),axis=0)
    y = np.concatenate((np.repeat(0,len(X1)),np.repeat(1,len(X2))))
    ncv = np.shape(X1)[0]
    del X1, X2
    rand_ind = np.arange(0,len(X))
    random.Random(seed).shuffle(rand_ind)
    X = X[rand_ind,:,ts:te]
    y = y[rand_ind]
    
    if model == 'SVM':
        clf = make_pipeline(
            StandardScaler(),  # z-score normalization
            SVC(kernel='rbf',gamma='auto',C=0.1))
    elif model =='LogReg':  
        clf = make_pipeline(
            StandardScaler(),  # z-score normalizations
            LogisticRegression(solver="liblinear"))
        
    for n in np.arange(0,np.shape(X)[1],1):
        scores = cross_val_multiscore(clf, X[:,n,:], y, cv=ncv,verbose = 'ERROR') # takes about 10 mins to run
        score = np.mean(scores, axis=0)
        print("Data " + str(n+1) + " Accuracy: %0.1f%%" % (100 * score,))
        all_score.append(score)
    return all_score

def redo_ROI(new_ROI): 
    ## Average or select ROIs from the 114 labels
    # Auditory (STG 72,108, HG 76, 112), Motor (precentral 66 102), Sensorimotor (postcentral 64 100), and between them is paracentral 59, 95
    # Basal ganglia group (7,8,9,16,26,27,28,31): out of all include caudate (7 26) and putamen (8 27) only based on Cannon & Patel 2020 TICS, putamen is most relevant 
    # Frontal IFG (60,61,62,96,97,98)
    # new_ROI = {"AuditoryL": [72,76],"AuditoryR": [108,112], "MotorL": [66],"MotorR": [102], "SensoryL": [59,64],"SensoryR": [95,100], "BGL": [7,8],"BGR": [26,27], "IFGL": [60,61,62], "IFGR": [96,97,98]}
    # new_ROI = {"Auditory": [72,76,108,112], "Motor": [66,102], "Sensory": [59,64,95,100], "BG": [7,8,26,27], "IFG": [60,61,62,96,97,98]}
    age = ['7mo','11mo','br']  
    run = ['_02','_03','_04'] # random, duple, triple
    
    for n_age in age:
        for n_run in run:
            f_name = n_age + '_group' + n_run + '_stc_rs_mne_mag6pT_roi' 
            MEG0 = np.load(root_path + 'data/' + f_name + '.npy')   
            f_name = f_name +'_redo' + str(len(new_ROI))
            MEG = np.zeros((np.shape(MEG0)[0],len(new_ROI),np.shape(MEG0)[2]))
            for index, ROI in enumerate(new_ROI):
                MEG[:,index,:] = MEG0[:,new_ROI[ROI],:].mean(axis=1)
                np.save(root_path + 'data/' + f_name + '.npy', MEG)

#%%####################################### Do the jobs           
if __name__ == '__main__':
    #%%####################################### Set path
    root_path = '/media/tzcheng/storage/ME2_MEG/Zoe_analyses/me2_meg_analysis/'
    subjects_dir = '/media/tzcheng/storage2/subjects/'

    #%% Parameters
    age = ['7mo','11mo','br']  
    run = ['_02','_03','_04'] # random, duple, triple
    which_data_type = ['_sensor','_roi','_roi_redo4','_morph']
    data_type = which_data_type[2]
    MEG_fs = 250

    #%% Redo ROI if needed
    new_ROI = {"Auditory": [72,76, 108,112], "Motor": [66,102], "Sensory": [59,64,95,100], "BG": [7,8,26,27], "IFG": [60,61,62,96,97,98]}
    new_ROI = {"Auditory": [72,76, 108,112], "SensoriMotor": [66,102,59,64,95,100], "BG": [7,8,26,27], "IFG": [60,61,62,96,97,98]}
    redo_ROI(new_ROI)
    
    #%%####################################### Run the psds, tfr, conn
    ## Can run psds, tfr and conn for sensor, ROI and wholebrain. The research interests are: 
    ## 1. psds of the sensor, ROI, whole brain 
    ## 2. conn between the ROIs
    ## Load each condition one by one
    for n_age in age:
        for n_run in run:
            if data_type == '_sensor':
                f_name = n_age + '_group' + n_run + '_rs_mag6pT' + data_type 
            else:
                f_name = n_age + '_group' + n_run + '_stc_rs_mne_mag6pT' + data_type 
            MEG = np.load(root_path + 'data/' + f_name + '.npy') 
            psds = do_SSEP(MEG, f_name, fmin=0.5, fmax=5, MEG_fs=MEG_fs)
            # tfr,times,freqs = do_ERSP(MEG, f_name, fmin=5, fmax=35, f_step=1, MEG_fs=MEG_fs,n_cycles=15,baseline='percent',output='power')
            con = do_connectivity(MEG, f_name, fmin=1, fmax=35, f_step=200, MEG_fs=MEG_fs, directional=False)
            del MEG

    #%%####################################### Run the decoding
    ## Can run decoding on sensor, ROI and wholebrain. The research interests are on the wholebrain decoding results
    ## Decode the duple vs. random and triple vs. random, need to load at least two conditions
    #%% Decode across subjects
    for n_age in age:
        MEG_random = np.load(root_path + 'data/' + n_age + '_group_02_stc_rs_mne_mag6pT' + data_type + '.npy')
        MEG_duple = np.load(root_path + 'data/' + n_age + '_group_03_stc_rs_mne_mag6pT' + data_type + '.npy')
        MEG_triple = np.load(root_path + 'data/' + n_age + '_group_04_stc_rs_mne_mag6pT' + data_type + '.npy')
        acc_duple = do_decoding(MEG_duple, MEG_random, ts=0, te=2350, model='SVM', seed=15, nperm=100, criteria = 95) # outside the run loop
        acc_triple = do_decoding(MEG_triple, MEG_random, ts=0, te=2350, model='SVM', seed=15, nperm=100, criteria = 95) # outside the run loop
        np.save(root_path + 'decoding/' + n_age + data_type +'_decodingACC_duple.npy',acc_duple)
        np.save(root_path + 'decoding/' + n_age + data_type +'_decodingACC_triple.npy',acc_triple)

    #%% Decode across trials for each subject
    all_score_duple = []
    all_score_triple = []
    subj_path=['/media/tzcheng/storage/ME2_MEG/Zoe_analyses/7mo/' ,
               '/media/tzcheng/storage/ME2_MEG/Zoe_analyses/11mo/',
               '/media/tzcheng/storage/BabyRhythm/']
    for age_group in subj_path:
        subj = [] 
        for file in os.listdir(age_group):
            if file.startswith('br_') or file.startswith('me2_'):
                subj.append(file)
                
        for s in subj:
            file_in = age_group + s + '/sss_fif/' + s
            MEG_random = np.load(file_in + '_02_stc_mne_epoch_rs100_mag6pT'  + '.npy')
            MEG_duple = np.load(file_in + '_03_stc_mne_epoch_rs100_mag6pT'  + '.npy')
            MEG_triple = np.load(file_in + '_04_stc_mne_epoch_rs100_mag6pT'  + '.npy')
            acc_duple = do_decoding(MEG_duple, MEG_random, ts=0, te=2350, model='SVM', seed=15, nperm=100, criteria = 95) # outside the run loop
            acc_triple = do_decoding(MEG_triple, MEG_random, ts=0, te=2350, model='SVM', seed=15, nperm=100, criteria = 95) # outside the run loop
            np.save(root_path + 'decoding/by_subjects/' + s + '_wholebrain_decodingACC_duple.npy',acc_duple)
            np.save(root_path + 'decoding/by_subjects/' + s + '_wholebrain_decodingACC_triple.npy',acc_triple)
            all_score_duple.append(acc_duple)
            all_score_triple.append(acc_triple)
        np.save(root_path + 'decoding/by_subjects/' + 'br_wholebrain_decodingACC_trial_duple.npy', np.asarray(all_score_duple))
        np.save(root_path + 'decoding/by_subjects/' + 'br_wholebrain_decodingACC_trial_triple.npy', np.asarray(all_score_triple))