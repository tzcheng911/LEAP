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
from fooof import FOOOF
from fooof.sim.gen import gen_aperiodic
from fooof.plts.spectra import plot_spectra
from mne.decoding import cross_val_multiscore
from mne_connectivity import spectral_connectivity_time, read_connectivity, envelope_correlation
import os
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogistiscRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt 
from scipy import stats 

#%%####################################### Define functions
def do_SSEP(data, f_name, fmin, fmax, MEG_fs,fooof, width,n_peaks,min_peak_height, data_type):  
    ## compute psds from the time series with mne.time_frequency.psd_array_welch function
    # data: input MEG time series (subjects, sources, times) 
    # f_name: input name for saving the file
    # fmin: min frequency for the psd output
    # fmax: max frequency for the psd output
    # MEG_fs: MEG (re)sampling rate
    # data_type : str
    # 'sensor' -> average sensors before FOOOF
    # 'source' -> run FOOOF separately for each ROI
    psds, freqs = mne.time_frequency.psd_array_welch(
    data,MEG_fs, # could replace with label time series
    n_fft=np.shape(data)[2],
    n_overlap=0,
    n_per_seg=None,
    fmin=fmin,
    fmax=fmax,)

    init_flat_spec = None

    if fooof:
        print("Doing FOOOF")
        flat_psds = []

        # --------------------------------------------------
        # SENSOR DATA: average sensors
        # (subj, sensor, freq) -> (subj, freq)
        # --------------------------------------------------
        if data_type == "_sensor":

            sensor_mean_psds = psds.mean(axis=1)

            for nsub in range(sensor_mean_psds.shape[0]):

                fm = FOOOF(
                    peak_width_limits=width,
                    max_n_peaks=n_peaks,
                    min_peak_height=min_peak_height,
                )

                fm.fit(freqs, sensor_mean_psds[nsub], width)

                init_ap_fit = gen_aperiodic(
                    fm.freqs,
                    fm._robust_ap_fit(fm.freqs, fm.power_spectrum),
                )

                init_flat_spec = fm.power_spectrum - init_ap_fit

                flat_psds.append(init_flat_spec)

            # (subj, 1, freq)
            init_flat_spec = np.array(flat_psds)[:, np.newaxis, :]

        # --------------------------------------------------
        # SOURCE DATA: preserve ROI dimension
        # (subj, ROI, freq)
        # --------------------------------------------------
        elif data_type == "_roi_redo4" or data_type == "_roi" or data_type == "_morph":

            for nsub in range(psds.shape[0]):

                roi_specs = []

                for nroi in range(psds.shape[1]):
                    # print("Now doing v " + str(nroi))
                    fm = FOOOF(
                        peak_width_limits=width,
                        max_n_peaks=n_peaks,
                        min_peak_height=min_peak_height,
                    )
                    
                    ## Replace zeros with a small epsilon for FOOOF
                    eps = np.finfo(float).eps
                    psd_safe = np.maximum(psds[nsub, nroi], eps)

                    fm.fit(freqs, psd_safe, width)

                    init_ap_fit = gen_aperiodic(
                        fm.freqs,
                        fm._robust_ap_fit(fm.freqs, fm.power_spectrum),
                    )

                    init_flat = fm.power_spectrum - init_ap_fit

                    roi_specs.append(init_flat)

                flat_psds.append(roi_specs)

            # (subj, ROI, freq)
            init_flat_spec = np.array(flat_psds)

        else:
            raise ValueError("data_type must be '_sensor' or '_roi_redo4' or '_morph'")
            
        np.savez(
            root_path + 'SSEP/' + f_name + '_fpsds.npz',
            psds=init_flat_spec,
            freqs=freqs,
        )

    else:

        np.savez(
            root_path + 'SSEP/' + f_name + '_psds.npz',
            psds=psds,
            freqs=freqs,
        )

    return psds, init_flat_spec

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
        con_methods = ["plv","coh","pli","wpli"]
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
        con[3].save(root_path + 'connectivity/' + f_name + '_conn_wpli')
        
        ## implement AEP https://mne.tools/mne-connectivity/stable/auto_examples/mne_inverse_envelope_correlation.html
        bp_MEG = mne.filter.filter_data(data, MEG_fs, fmin, fmax, verbose=None)
        corr_obj = envelope_correlation(bp_MEG, orthogonalize="pairwise",verbose=None)
        corr = corr_obj.get_data(output="dense")[:,:,:,0]
        print("AEP between SM-A ROIs: " + str(corr.mean(0)[1][0]))
        print("AEP between SM-B ROIs: " + str(corr.mean(0)[2][1]))
        np.save(root_path + 'connectivity/' + f_name + '_conn_aec',corr)
    return con,corr

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

def redo_ROI(new_ROI,f_name): 
    ## Average or select ROIs from the 114 labels
    # Auditory (STG 72,108, HG 76, 112), Motor (precentral 66 102), Sensorimotor (postcentral 64 100), and between them is paracentral 59, 95
    # Basal ganglia group (7,8,9,16,26,27,28,31): out of all include caudate (7 26) and putamen (8 27) only based on Cannon & Patel 2020 TICS, putamen is most relevant 
    # Frontal IFG (60,61,62,96,97,98)
    # new_ROI = {"AuditoryL": [72,76],"AuditoryR": [108,112], "MotorL": [66],"MotorR": [102], "SensoryL": [59,64],"SensoryR": [95,100], "BGL": [7,8],"BGR": [26,27], "IFGL": [60,61,62], "IFGR": [96,97,98]}
    # new_ROI = {"Auditory": [72,76,108,112], "Motor": [66,102], "Sensory": [59,64,95,100], "BG": [7,8,26,27], "IFG": [60,61,62,96,97,98]}

    MEG0 = np.load(root_path + 'data/' + f_name + '.npy')   
    f_name = f_name +'_redo' + str(len(new_ROI))
    MEG = np.zeros((np.shape(MEG0)[0],len(new_ROI),np.shape(MEG0)[2]))
    for index, ROI in enumerate(new_ROI):
        MEG[:,index,:] = MEG0[:,new_ROI[ROI],:].mean(axis=1)
        np.save(root_path + 'data/' + f_name + '.npy', MEG)

#%%####################################### Do the jobs           

#%%####################################### Set path
root_path = '/media/tzcheng/storage/ME2_MEG/Zoe_analyses/me2_meg_analysis/'
subjects_dir = '/media/tzcheng/storage2/subjects/'

#%% Parameters
age = ['7mo','11mo','br']  
run = ['_02','_03','_04'] # random, duple, triple
# random duple and triple seperately 
randomDT = ['', '_randduple','_randtriple'] # use randomDT[1] and randomDT[2] with run = '02', otherwise use randomDT[0] 
which_data_type = ['_sensor','_roi','_roi_redo4','_morph']
data_type = which_data_type[2]
MEG_fs = 250
fooof = False
width = [0.5,5]
n_peaks=3
min_peak_height=5e-26
fmin = 0.5
fmax = 5

#%% Redo ROI if needed
## fsaverage 114 labels 
new_ROI = {"Auditory": [72,76, 108,112], "Motor": [66,102], "Sensory": [59,64,95,100], "BG": [7,8,26,27], "IFG": [60,61,62,96,97,98]}
new_ROI = {"Auditory": [72,76, 108,112], "SensoriMotor": [66,102,59,64,95,100], "BG": [7,8,26,27], "IFG": [60,61,62,96,97,98]}

## ANTS6-0Months3T 101 labels 
# new_ROI = {"Auditory": [61,65, 95,99], "SensoriMotor": [48,53,55,82,87,89], "BG": [6,7,20,21]}

data_type = which_data_type[1]
for n_age in age:
    for n_run in run:
        f_name = n_age + '_group' + n_run + '_stc_rs_mne_nonmorph_mag6pT' + randomDT[0] + data_type 
        redo_ROI(new_ROI,f_name)

#%%####################################### Run the psds, tfr, conn
## Can run psds, tfr and conn for sensor, ROI and wholebrain. The research interests are: 
## 1. psds of the sensor, ROI, whole brain 
## 2. conn between the ROIs
## Load each condition one by one
for n_age in age:
    print("Doing age " + n_age)
    for n_run in run:
        print("Doing run " + n_run)
        if data_type == '_sensor':
            f_name = n_age + '_group' + n_run + '_rs_mag6pT' + randomDT[0] +  data_type 
        else:
            f_name = n_age + '_group' + n_run + '_stc_rs_mne_nonmorph_mag6pT' + randomDT[0] + data_type 
        MEG = np.load(root_path + 'data/' + f_name + '.npy') 
        [psds,init_flat_spec] = do_SSEP(MEG, f_name, fmin, fmax, MEG_fs, fooof, width,n_peaks,min_peak_height,data_type)
        # tfr,times,freqs = do_ERSP(MEG, f_name, fmin=5, fmax=35, f_step=1, MEG_fs=MEG_fs,n_cycles=15,baseline='percent',output='power')
        # con = do_connectivity(MEG, f_name, fmin=1, fmax=35, f_step=200, MEG_fs=MEG_fs, directional=False)
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

#%%####################################### Address reviewer requests
#%% Reviewer 1: cHPI head movement
root_path='/media/tzcheng/storage/ME2_MEG/Zoe_analyses/7mo/'
os.chdir(root_path)
subjects = []
runs = ['_02','_03','_04'] 
head_pos_all = []

for file in os.listdir():
    if file.startswith('me2_'): 
        subjects.append(file)
for s in subjects:
    print(s)      
    for run in runs:
        print(run)
        filename = root_path + s + '/raw_fif/' + s + run + '_otp_raw.fif'
        raw = mne.io.read_raw_fif(filename, allow_maxshield="yes")
        chpi_freqs, ch_idx, chpi_codes = mne.chpi.get_chpi_info(info=raw.info)
        chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(raw)
        chpi_locs = mne.chpi.compute_chpi_locs(raw.info, chpi_amplitudes)
        head_pos = mne.chpi.compute_head_pos(raw.info, chpi_locs, verbose=True) #  The columns correspond to [t, q1, q2, q3, x, y, z, gof, err, v] for each time point
        mne.viz.plot_head_positions(head_pos, mode="traces", totals=True)
        plt.figure()
        plt.plot(abs(head_pos[:,4])+abs(head_pos[:,5])+abs(head_pos[:,6]))
        head_pos_all.append({
            "subject": s,
            "run": run,
            "head_pos": head_pos,
        })

## calculate the total distance across conditions
metric_by_run = {}

for run in runs:
    subject_means = []
    plt.figure()
    for record in head_pos_all:
        if record["run"] == run:
            hp = record["head_pos"]

            # total displacement time series
            total_dist = np.abs(hp[:, 4]) + np.abs(hp[:, 5]) + np.abs(hp[:, 6])
            plt.plot(total_dist)
            # one value for this subject/run
            subject_means.append(np.mean(total_dist))
            
            
    metric_by_run[run] = {
        "mean": np.mean(subject_means),
        "std": np.std(subject_means, ddof=1),
        "n": len(subject_means),
        "values": subject_means
    }

for run in runs:
    print(
        f"{run}: n={metric_by_run[run]['n']}, "
        f"mean={metric_by_run[run]['mean']:.4f}, "
        f"std={metric_by_run[run]['std']:.4f}"
    )

## does not survive multiple comparison correction
# pvalue=np.float64(0.5618139504972308)
stats.ttest_1samp(
    np.asarray(metric_by_run['_03']['values']) -
    np.asarray(metric_by_run['_02']['values']),
    0
)
# pvalue=np.float64(0.01978672416754115)
stats.ttest_1samp(
    np.asarray(metric_by_run['_04']['values']) -
    np.asarray(metric_by_run['_02']['values']),
    0
)
# pvalue=np.float64(0.03952887777581683)
stats.ttest_1samp(
    np.asarray(metric_by_run['_03']['values']) -
    np.asarray(metric_by_run['_04']['values']),
    0
)

#%% Reviewer 2: Verification of source localization

#%% Goodness-of-fit: plot the residual on top of evoked and see the print out of the vairance explained
subjects_dir = '/media/tzcheng/storage2/subjects/'
root_path='/media/tzcheng/storage/ME2_MEG/Zoe_analyses/7mo/' # change to 11mo and /media/tzcheng/storage/BabyRhythm/
os.chdir(root_path)
rfs=250
lambda2 = 0.1111111111111111

runs = ['_02','_03','_04']
subj = [] 
for file in os.listdir():
    if file.startswith('me2_'):
        subj.append(file)
for s in subj:
    for run in runs:
        file_in = root_path + s + '/sss_fif/' + s
        fwd = mne.read_forward_solution(file_in + '-fwd.fif')
        epoch = mne.read_epochs(file_in + run + '_otp_raw_sss_proj_fil50_mag6pT_epoch.fif').resample(sfreq = 100)
        noise_cov = mne.read_cov(file_in + run + '_erm_otp_raw_sss_proj_fil50-cov.fif')
        data_cov = mne.compute_covariance(epoch, tmin=0, tmax=None)
        evoked = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_fil50_mag6pT_evoked.fif')[0].resample(sfreq = rfs)
        
        inverse_operator = mne.minimum_norm.make_inverse_operator(epoch.info, fwd, noise_cov,loose=1,depth=0.8)
        stc_mne, residual = mne.minimum_norm.apply_inverse((evoked), inverse_operator, return_residual=True, pick_ori = None)
        
        fig, axes = plt.subplots(2, 1)
        evoked.plot(axes=axes)
        for ax in axes:
            for text in list(ax.texts):
                text.remove()
            for line in ax.lines:
                line.set_color("#98df81")
        residual.plot(axes=axes)

#%% simulation - this is for one subject now

#### complete version of simulation
## place the source point or label in the vol model
## simulate the raw data from the vol stc, can add chpi, ecg, eog, noise on this step
## run the preprocessing pipeline
## run the src pipeline to compare estimated and simulated source activity 

#### simplified version of simulation
## place the source point or label in the vol model
## simulate the evoked data from the vol stc
## run the src pipeline to compare estimated and simulated source activity 
root_path = '/media/tzcheng/storage/ME2_MEG/Zoe_analyses/7mo/'
subjects_dir = '/media/tzcheng/storage2/subjects/'
subject = 'me2_101_7m'
label_list = ['Left-Putamen','Left-Caudate','Right-Putamen','Right-Caudate']

fname_raw = root_path + subject + '/raw_fif/' + subject + '_03_raw.fif'
fname_evoked= subjects_dir + subject + '/sss_fif/' + subject + '_04_otp_raw_sss_proj_fil50_mag6pT_evoked.fif'
fname_trans = root_path + subject + '/sss_fif/' + subject + '_trans.fif'
fname_bem = subjects_dir + subject + '/bem/' + subject + '-5120-5120-5120-bem-sol.fif'
fname_src = subjects_dir + subject + '/bem/' + subject + '-vol-5-src.fif'
fname_aseg= subjects_dir + subject + '/mri/aparc+aseg.mgz'

trans=mne.read_trans(fname_trans)
bem=mne.read_bem_solution(fname_bem)
fwd=mne.make_forward_solution(raw.info,trans,src_putamen,bem,meg=True,eeg=False)

print('Creating a source')

src_BG = mne.setup_volume_source_space(
    subject, mri=fname_aseg, pos=5.0, bem=bem_fname,
    volume_label= label_list, add_interpolator=False)

rr = np.mean(src_BG[0]['rr'], axis=0, keepdims=True)
nn = np.array([[0., -1., -1.]])  # backward and up (this is probably wrong)
nn /= np.linalg.norm(nn, axis=-1)

src_sim = mne.setup_volume_source_space(subject, pos=dict(rr=rr, nn=nn))
vertices = [src_sim[0]['vertno']]
tmin, tmax = -0.01, 0.05
n_samp = int(round((tmax - tmin) * raw.info['sfreq'])) + 1
data = np.zeros([sum(len(v) for v in vertices), n_samp])
t_activation = 0.01
activation = np.hanning(int(round(raw.info['sfreq'] * t_activation)))
activation *= 5e-9  # nAm (this is way too high)
t_offset = 0.0
n_offset = int(round((t_offset - tmin) * raw.info['sfreq']))
data[:, n_offset:n_offset + len(activation)] = activation
stc = mne.VolSourceEstimate(data, vertices[0], tmin=tmin,
                            tstep=1. / raw.info['sfreq'])



stc_gen = mne.simulation.SourceSimulator(src, tstep=tstep)
    events = np.array([[0, 0, 1]])  # single event at t=0; adapt if you need more
    stc_gen.add_data(active_label, waveform, events)
    stc_true = stc_gen.get_stc()

    sim_evoked = mne.simulation.simulate_evoked(
        fwd, stc_true, info, cov=noise_cov, nave=nave if nave else 30
    )

    stc_est = apply_inverse(sim_evoked, inverse_operator, lambda2=1.0 / 9.0,
                             method=method)

    recovered_at_target = np.abs(stc_est.data[readout_indices]).mean(axis=0)
    return stc_true, stc_est, recovered_at_target


# --- (a) positive control: signal only in basal ganglia ---
bg_idx = get_volume_label_indices(fwd['src'], 'Left-Putamen')           
waveform_a = np.sin(2*np.pi*3.33*t) * envelope
waveform = source_time_series = np.sin(2.0 * np.pi * 18.0 * np.arange(100) * tstep) * 10e-9
stc_true_a, stc_est_a, recovered_bg = run_ground_truth_simulation(
    fwd, fwd['src'], evoked.info, noise_cov, inverse_operator,
    active_label='Left-Putamen', readout_indices=bg_idx,
    waveform=waveform_a, tstep=1.0 / evoked.info['sfreq'])
# Report: peak localization error (distance in mm between true voxel and
# the peak of stc_est restricted to the volume portion of src), and the
# ratio of leaked amplitude at cortical vertices vs. the true basal ganglia
# amplitude (should be small).

# --- (b) leakage test: signal only in nearby cortex, NONE in basal ganglia ---
ctx_label = mne.read_labels_from_annot(subject, ...)[SOME_LABEL]         # [ADAPT]
waveform_b = ...  # matched in amplitude/SNR to your real cortical effect  # [ADAPT]
_, stc_est_b, leaked_into_bg = run_ground_truth_simulation(
    fwd, fwd['src'], evoked.info, noise_cov, inverse_operator,
    active_label=ctx_label, readout_indices=bg_idx,
    waveform=waveform_b, tstep=1.0 / evoked.info['sfreq'])
# Report: leaked_into_bg as a fraction of your REAL, observed basal ganglia
# effect size. If the leaked amplitude is small relative to your observed
# effect, that's your strongest single piece of evidence against leakage.

#%% point-spread functions for the whole-brain
# compute for each individual (see ROI_lookup.py)
# quick test on fsaverage as a sand model
inverse_operator = mne.minimum_norm.make_inverse_operator(epoch.info, fwd, noise_cov,loose=1,depth=0.8)

# compute resolution matrix for dSPM
rm_lor_vol = mne.minimum_norm.make_inverse_resolution_matrix(fwd, inverse_operator, pick_ori = None, method="dSPM", lambda2=lambda2)

# prepare the label
label_v_ind = np.load('/media/tzcheng/storage/scripts_zoe/ROI_lookup_ANTS6-0Months3T.npy', allow_pickle=True)
fname_aseg = subjects_dir + s + '/mri/aparc+aseg.mgz'
label_names = mne.get_volume_labels_from_aseg(fname_aseg)

sources_vol = label_v_ind[21][0] # Right-Putamen

stc_psf_vol = mne.minimum_norm.get_point_spread(rm_lor_vol, fwd["src"], sources_vol, mode = 'mean', norm=True)
stc_ctf_vol = mne.minimum_norm.get_cross_talk(rm_lor_vol, fwd["src"], sources_vol, mode = 'mean', norm=True)

n = 5981

rm_lor_vol_reduced = np.linalg.norm(
    rm_lor_vol.reshape(n, 3, n, 3),
    axis=(1, 3)
)

print(rm_lor_vol.shape)
print(rm_lor_vol_reduced.shape)

locations = np.linalg.norm(np.vstack([s['rr'][s['vertno']] for s in fwd["src"]])*1000,axis=1)  # -> mm
resmax = np.abs(rm_lor_vol_reduced).argmax(axis=0)
diffloc = locations - locations[resmax]
np.linalg.norm(diffloc, axis=1) 

def _get_all_src_locations(src):
"""Generalized replacement for MNE's internal _get_src_locations,
which is hardcoded to exactly two surface source spaces. This works
for surface, volume, or mixed source spaces by concatenating each
sub-source-space's used vertex coordinates in stacking order —
matching the row/column order of the resolution matrix."""
return np.vstack([s['rr'][s['vertno']] for s in src])  # (n_dipoles, 3), meters

def peak_localization_error(resmat, src, function='ctf'):
"""Same formula as mne.minimum_norm.resolution_metrics(metric='peak_err'),
generalized to mixed/volume source spaces."""
locations = 1000 * _get_all_src_locations(src)  # -> mm
m = resmat.T if function == 'ctf' else resmat
resmax = np.abs(m).argmax(axis=0)
diffloc = locations - locations[resmax]
return np.linalg.norm(diffloc, axis=1)  # per-vertex, mm

def spatial_deviation(resmat, src, function='ctf'):
"""Same formula as metric='sd_ext' (Molins et al., 2008, eq. 12),
generalized to mixed/volume source spaces."""
locations = 1000 * _get_all_src_locations(src)
m = resmat.T if function == 'ctf' else resmat
width = np.empty(m.shape[1])
for i in range(m.shape[1]):
sq_dist = np.sum((locations - locations[i]) ** 2, axis=1)
w = np.abs(m[:, i]) ** 2
width[i] = np.sqrt(np.sum(sq_dist * w) / np.sum(w))
return width





src_vol = fwd["src"]
verttrue_vol = src_vol[0]["vertno"][sources_vol]

# find vertex with maximum in PSF
max_vert_idx, _ = np.unravel_index(stc_psf_vol.data.argmax(), stc_psf_vol.data.shape)
vert_max_ctf_vol = src_vol[0]["vertno"][[max_vert_idx]]

# plot them
brain_psf_vol = stc_psf_vol.plot_3d(
    "me2_215_7m",
    src=fwd["src"],
    views="ven",
    subjects_dir=subjects_dir,
    volume_options=dict(alpha=0.5),
)
brain_psf_vol.add_text(0.1, 0.9, "Volumetric sLORETA PSF", "title", font_size=16)
brain_psf_vol.add_foci(
    verttrue_vol, coords_as_verts=True, scale_factor=1, hemi="vol", color="green"
)
brain_psf_vol.add_foci(
    vert_max_ctf_vol,
    coords_as_verts=True,
    scale_factor=1.25,
    hemi="vol",
    color="black",
    alpha=0.3,
)

## point-spread functions for the ROI label
