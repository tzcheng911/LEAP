#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:14:44 2024

Streamline processing the following tasks and save the output data. Only include the formal analyses 
that will be included in the paper main results. 
Input: .npy files in the data folder
Output: .npy or mne formats in the SSEP, ERSP, decoding, connectivity folders

currently not able to run ERSP and conn on the wholebrain data, but could do SSEP and ecoding on the wholebrain data
 
@author: tzcheng
"""

#%%####################################### Import library  
import numpy as np
import time
import random
import copy
import scipy.stats as stats
from scipy import stats,signal
from scipy.io import wavfile
import mne
from mne import spatial_src_adjacency
from mne.stats import spatio_temporal_cluster_1samp_test, summarize_clusters_stc
from mne.time_frequency import tfr_morlet, tfr_multitaper, tfr_stockwell, AverageTFRArray
from mne_connectivity import spectral_connectivity_epochs, spectral_connectivity_time,read_connectivity
from mne_connectivity.viz import plot_connectivity_circle
from mne.viz import circular_layout
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
import os
import sklearn 
from sklearn.decomposition import PCA, FastICA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
import matplotlib.pyplot as plt 

#%%####################################### Set path
root_path = '/media/tzcheng/storage/ME2_MEG/Zoe_analyses/me2_meg_analysis/'
subjects_dir = '/media/tzcheng/storage2/subjects/'

#%%####################################### Define functions
def do_SSEP(data, f_name, fmin, fmax, MEG_fs):  
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
    ## check the tfr_array_morlet function for options
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
    freqs = np.linspace(fmin,fmax,f_step)
    if directional:
        con_methods = ["gc"]
        con = spectral_connectivity_time(
            data,
            indices = (np.array([[2, 3], [2, 3], [7, 26], [8,27], [0, 1], [71, 107],[2, 3],[2, 3]]),  # seeds
            np.array([[0, 1], [71, 107],[2, 3],[2, 3],[2, 3], [2, 3], [7, 26], [8,27]])),  # targets
            freqs = freqs,
            method=con_methods,
            mode="multitaper", # if using cwt_morlet, add cwt_freqs = nfreq = np.array([1,2,3,4,5])
            sfreq=MEG_fs,
            fmin=fmin,
            fmax=fmax,
            faverage=False,
            n_jobs=1,)
        con.save(root_path + 'connectivity/' + f_name + 'GC')
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

def do_decoding(X1, X2, ts, te, model, seed, nperm,criteria):  
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
    elif model =='LogReg'      :  
        clf = make_pipeline(
            StandardScaler(),  # z-score normalizations
            LogisticRegression(solver="liblinear"))
        
    for n in np.arange(0,np.shape(X)[1],1):
        scores = cross_val_multiscore(clf, X[:,n,:], y, cv=ncv,verbose = 'ERROR') # takes about 10 mins to run
        score = np.mean(scores, axis=0)
        print("Data " + str(n+1) + " Accuracy: %0.1f%%" % (100 * score,))
        all_score.append(score)
    return all_score

def redo_ROI(age,run,new_ROI):
    ## age = ['7mo','11mo','br']  
    ## run = ['_02','_03','_04'] # random, duple, triple
    ## new_ROI = {"Auditory": [72,76, 108,112], "Motor": [66,102], "Sensory": [59,64,95,100], "BG": [7,8,26,27], "IFG": [60,61,62,96,97,98]}
    
    # Auditory (STG 72,108, HG 76, 112), Motor (precentral 66 102), Sensorimotor (postcentral 64 100), and between them is paracentral 59, 95
    # Basal ganglia group (7,8,9,16,26,27,28,31): out of all include caudate (7 26) and putamen (8 27) only based on Cannon & Patel 2020 TICS, putamen is most relevant 
    # Frontal IFG (60,61,62,96,97,98)
    for n_age in age:
        for n_run in run:
            f_name = n_age + '_group' + n_run + '_stc_rs_mne_mag6pT_roi' 
            MEG0 = np.load(root_path + 'data/' + f_name + '.npy')   
            f_name = f_name +'_redo'
            new_ROI = {"AuditoryL": [72,76],"AuditoryR": [108,112], "MotorL": [66],"MotorR": [102], "SensoryL": [59,64],"SensoryR": [95,100], "BGL": [7,8],"BGR": [26,27], "IFGL": [60,61,62], "IFGR": [96,97,98]}
            MEG = np.zeros((np.shape(MEG0)[0],len(new_ROI),np.shape(MEG0)[2]))
            for index, ROI in enumerate(new_ROI):
                MEG[:,index,:] = MEG0[:,new_ROI[ROI],:].mean(axis=1)
                np.save(root_path + 'data/' + f_name + '.npy', MEG)
                
#%%####################################### Run the psds, tfr, con
if __name__ == '__main__':
#%% Parameters
    age = ['7mo','11mo','br']  # 
    run = ['_02','_03','_04'] # random, duple, triple # 
    which_data_type = ['_sensor','_roi','_roi_redo','_morph'] ## currently not able to run ERSP and conn on the wholebrain data
    data_type = which_data_type[3]
    MEG_fs = 250
    decoding_acc = []

    for n_age in age:
        for n_run in run:
            if data_type == '_sensor':
                f_name = n_age + '_group' + n_run + '_rs_mag6pT' + data_type 
            else:
                f_name = n_age + '_group' + n_run + '_stc_rs_mne_mag6pT' + data_type 
            MEG = np.load(root_path + 'data/' + f_name + '.npy') 
            
            psds = do_SSEP(MEG, f_name, fmin=0.5, fmax=5, MEG_fs=MEG_fs)
            tfr,times,freqs = do_ERSP(MEG, f_name, fmin=5, fmax=35, f_step=1, MEG_fs=MEG_fs,n_cycles=15,baseline='percent',output='power')
            con = do_connectivity(MEG, f_name, fmin=1, fmax=35, f_step=200, MEG_fs=MEG_fs, directional=False)
            
            del MEG

    #%%####################################### Run the decoding
    #%% Parameters
    age = ['7mo','11mo','br']  
    which_data_type = ['_roi','_roi_redo','_morph'] ## currently not able to run ERSP and conn on the wholebrain data
    data_type = which_data_type[2]
    all_score_all = []

    ## decode across subjects
    for n_age in age:
        MEG_random = np.load(root_path + 'data/' + n_age + '_group_02_stc_rs_mne_mag6pT' + data_type + '.npy')
        # MEG_duple = np.load(root_path + 'data/' + n_age + '_group_03_stc_rs_mne_mag6pT' + data_type + '.npy')
        MEG_triple = np.load(root_path + 'data/' + n_age + '_group_04_stc_rs_mne_mag6pT' + data_type + '.npy')
        # acc_duple = do_decoding(MEG_duple, MEG_random, ts=0, te=2350, model='SVM', seed=15, nperm=100, criteria = 95) # outside the run loop
        acc_triple = do_decoding(MEG_triple, MEG_random, ts=0, te=2350, model='SVM', seed=15, nperm=100, criteria = 95) # outside the run loop
        # np.save(root_path + 'decoding/' + n_age + data_type +'_decodingACC_duple.npy',acc_duple)
        np.save(root_path + 'decoding/' + n_age + data_type +'_decodingACC_triple.npy',acc_triple)

    #%% decode across trials for each subject 
    subj_path='/media/tzcheng/storage/ME2_MEG/Zoe_analyses/7mo/' # change to 11mo 
    # subj_path = '/media/tzcheng/storage/BabyRhythm/'
    subj = [] 
    for file in os.listdir(subj_path):
        # if file.startswith('br_'):
        if file.startswith('me2_'):
            subj.append(file)
            
    for s in subj:
        file_in = subj_path + s + '/sss_fif/' + s
        MEG_random = np.load(file_in + '_02_stc_mne_epoch_rs100_mag6pT.npy')
        MEG_duple = np.load(file_in + '_03_stc_mne_epoch_rs100_mag6pT.npy')
        MEG_triple = np.load(file_in + '_04_stc_mne_epoch_rs100_mag6pT.npy')
        # acc_duple_triple = do_decoding(MEG_duple, MEG_triple, ts=0, te=2350, model='SVM', seed=15, nperm=100, criteria = 95) # outside the run loop
        acc_duple = do_decoding(MEG_duple, MEG_random, ts=0, te=2350, model='SVM', seed=15, nperm=100, criteria = 95) # outside the run loop
        acc_triple = do_decoding(MEG_triple, MEG_random, ts=0, te=2350, model='SVM', seed=15, nperm=100, criteria = 95) # outside the run loop
        # np.save(root_path + 'decoding/by_subjects/' + s + '_wholebrain_decodingACC_DT.npy',acc_duple_triple)
        np.save(root_path + 'decoding/by_subjects/' + s + '_wholebrain_decodingACC_duple.npy',acc_duple)
        np.save(root_path + 'decoding/by_subjects/' + s + '_wholebrain_decodingACC_triple.npy',acc_triple)