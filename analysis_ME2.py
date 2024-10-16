#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 16:53:55 2024 
1. Frequency Analysis
2. Time-Frequency Analysis
3. Connectivity Analysis
4. ML decoding Analysis (using whole time series to distinguish duple vs. triple)
@author: tzcheng
"""
#%%####################################### Import library  
import matplotlib.pyplot as plt
import numpy as np
import random
import os 

import mne
from mne.datasets import somato
from mne.time_frequency import tfr_morlet, tfr_multitaper, tfr_stockwell, AverageTFRArray
from scipy.io import wavfile
from scipy import stats,signal
from mne_connectivity import spectral_connectivity_epochs
from mne_connectivity.viz import plot_connectivity_circle
from mne.viz import circular_layout

from sklearn.decomposition import PCA, FastICA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC

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

def plot_err(group_data,color,t):
    group_avg=np.mean(group_data,axis=0)
    err=np.std(group_data,axis=0)/np.sqrt(group_data.shape[0])
    up=group_avg+err
    lw=group_avg-err
    #t=np.linspace(-100,600,3501)
    plt.plot(t,group_avg,color=color)
    plt.fill_between(t,up,lw,color=color,alpha=0.5)
    
#%%####################################### Load the files
age = 'br' # '7mo' (or '7mo_0_15' or '7mo_15_32' for MEG_v), '11mo', 'br' for adults
run = '_03' # '_01','_02','_03','_04' silence, random, duple, triple

subjects_dir = '/media/tzcheng/storage2/subjects/'
root_path='/media/tzcheng/storage/ME2_MEG/Zoe_analyses/'
fname_aseg = subjects_dir + 'fsaverage/mri/aparc+aseg.mgz'
label_names = np.asarray(mne.get_volume_labels_from_aseg(fname_aseg))
nROI = [72,108,66,102,59,95,7,8,9,16,26,27,28,31] # Auditory (STG 72,108), Motor (precentral 66 102 and paracentral 59, 95), 
# Basal ganglia group (7,8,9,16,26,27,28,31) (left and then right)
nV = 10020 # need to manually look up from the whole-brain plot

fs, audio = wavfile.read(root_path + 'Stimuli/Random.wav') # Random, Duple300, Triple300
MEG_sensor = np.load(root_path + 'me2_meg_analysis/' + age + '_group' + run + '_sensor.npy') # 01,02,03,04
MEG_v = np.load(root_path + 'me2_meg_analysis/' + age + '_group' + run + '_stc_mne.npy') # 01,02,03,04
MEG_roi = np.load(root_path + 'me2_meg_analysis/' + age + '_group' + run + '_stc_mne_roi.npy') # 01,02,03,04

stc1 = mne.read_source_estimate('/media/tzcheng/storage/BabyRhythm/br_03/sss_fif/br_03_01_stc_lcmv_morph-vl.stc')
times = stc1.times
src = mne.read_source_spaces(subjects_dir + 'fsaverage/bem/fsaverage-vol-5-src.fif')

# stc1.data = MEG_v.mean(axis=0)
# stc1.plot(src = src,clim=dict(kind="value",lims=[5,5.5,8]))

#%%####################################### Frequency analysis
# Are there peak amplitude in the beat and meter rates? which ROI, which source?
fmin = 0.5
fmax = 5
MEG_fs = 1000
n_lines = 10
n_freq = [33] # [6,7] 1.1 Hz, [12, 13] 1.6 Hz, [33] 3.33 Hz 

#%% Frequency spectrum of the audio 
psds, freqs = mne.time_frequency.psd_array_welch(
audio,fs, 
n_fft=len(audio),
n_overlap=0,
n_per_seg=None,
fmin=fmin,
fmax=fmax,)

plt.figure()
plt.plot(freqs,psds)

#%% Frequency spectrum of the MEG
## sensor
## option 1
evokeds = mne.read_evokeds(root_path + '7mo/me2_205_7m/sss_fif/me2_205_7m_01_otp_raw_sss_proj_fil50_evoked.fif')[0]
evokeds.data = MEG_sensor.mean(0)
evo_spectrum = evokeds.compute_psd('welch', fmin = fmin, fmax=40)
psds, freqs = evo_spectrum.get_data(return_freqs=True)
evo_spectrum.plot()
evo_spectrum.plot_topomap(ch_type = "grad")

## option 2
psds, freqs = mne.time_frequency.psd_array_welch(
MEG_sensor,MEG_fs, # could replace with label time series
n_fft=np.shape(MEG_sensor)[2],
n_overlap=0,
n_per_seg=None,
fmin=fmin,
fmax=fmax,)

plt.figure()
plot_err(psds[:,nROI,:],'k',freqs)
plt.title(label_names[nROI])

## ROI
psds, freqs = mne.time_frequency.psd_array_welch(
MEG_roi,MEG_fs, # could replace with label time series
n_fft=np.shape(MEG_roi)[2],
n_overlap=0,
n_per_seg=None,
fmin=fmin,
fmax=fmax,)

plt.figure()
plot_err(psds[:,nROI,:],'k',freqs)
plt.title(label_names[nROI])

## wholebrain
psds, freqs = mne.time_frequency.psd_array_welch(
MEG_v,MEG_fs, # take about 2 mins
n_fft=np.shape(MEG_v)[2],
n_overlap=0,
n_per_seg=None,
fmin=fmin,
fmax=fmax,)

plt.figure()
plot_err(psds[:,nV,:],'k',freqs)

## Whole-brain frequency spectrum
stc1.data = psds.mean(axis=0)
stc1.tmin = freqs[0] # hack into the time with freqs
stc1.tstep = np.diff(freqs)[0] # hack into the time with freqs
stc1.plot(src = src,clim=dict(kind="value",lims=[1,3,5]))

#%%####################################### Time-Frequency analysis
# Are there stronger amplitude in the beat and meter rates? which ROI?
## sensor
evokeds = mne.read_evokeds(root_path + '7mo/me2_205_7m/sss_fif/me2_205_7m_01_otp_raw_sss_proj_fil50_evoked.fif')[0]
evokeds.data = MEG_sensor.mean(0)
evo_tfr = evokeds.compute_tfr('morlet', n_cycles=3,freqs=np.arange(fmin, 40, 2))from mne.decoding import (
    SlidingEstimator,
    GeneralizingEstimator,
    Scaler,
    cross_val_multiscore,
    LinearModel,
    get_coef,
    Vectorizer,
    CSP,
)
evo_tfr.plot_topo(baseline=(-0.5, 0), mode="logratio", title="Average power")
evo_tfr.plot(picks=[82], baseline=(-0.5, 0), mode="logratio", title=evo_tfr.ch_names[82])
tfr, freqs = evo_tfr.get_data(return_freqs=True)

## ROI
epochs = mne.read_epochs(root_path + '7mo/me2_205_7m/sss_fif/me2_205_7m_01_otp_raw_sss_proj_fil50_epoch.fif')
source_tfr = mne.time_frequency.tfr_array_morlet(MEG_roi,MEG_fs,freqs=np.arange(1, 40, 2),n_cycles=3,output='power')

plt.figure()
plt.imshow(source_tfr.mean(0)[nROI[0],:,:],interpolation='bilinear',
               origin='lower')

## wholebrain: too big to run
source_tfr = mne.time_frequency.tfr_array_morlet(MEG_v,MEG_fs,freqs=np.arange(1, 40, 2),n_cycles=3,output='power')

#%%####################################### Connectivity analysis
# How are ROI connected, which direction?
con_methods = ["pli", "plv", "coh"]
con = spectral_connectivity_epochs( # Compute frequency- and time-frequency-domain connectivity measures
    MEG_roi[:,nROI,:],
    method=con_methods,
    mode="multitaper",
    sfreq=MEG_fs,
    fmin=fmin,
    fmax=fmax,
    faverage=False,
    mt_adaptive=True,
    n_jobs=1,
)

## Extract the data
test = con[2].get_data(output="dense")[:, :, n_freq].mean(2)

con_res = dict()
for method, c in zip(con_methods, con):
    con_res[method] = c.get_data(output="dense")[:, :, n_freq].mean(2) # get the n freq
    
## visualization
label_names = label_names[nROI]
labels = mne.read_labels_from_annot("sample", parc="aparc", subjects_dir=subjects_dir)
label_colors = [label.color for label in labels]

node_order = list()
node_order.extend(label_names)  # reverse the order
node_angles = circular_layout(
    label_names, node_order, start_pos=90, group_boundaries=[0, len(label_names) / 2]
)

fig, ax = plt.subplots(figsize=(8, 8), facecolor="black", subplot_kw=dict(polar=True))
plot_connectivity_circle(
    con_res["coh"],
    label_names,
    n_lines=n_lines, # plot the top n lines
    node_angles=node_angles,
    node_colors=label_colors,
    title="All-to-All Connectivity Br random",
    ax=ax)
fig.tight_layout()

#%%####################################### Decoding analysis
age = '11mo' # '7mo' or '7mo_0_15' or '7mo_15_32' or '11mo' or 'br' for adults
subjects_dir = '/media/tzcheng/storage2/subjects/'
root_path='/media/tzcheng/storage/ME2_MEG/Zoe_analyses/'
fname_aseg = subjects_dir + 'fsaverage/mri/aparc+aseg.mgz'
label_names = np.asarray(mne.get_volume_labels_from_aseg(fname_aseg))
nROI = [72,108,66,102,59,95,7,8,9,16,26,27,28,31] # Auditory (STG 72,108), Motor (precentral 66 102 and paracentral 59, 95), 
# Basal ganglia group (7,8,9,16,26,27,28,31) (left and then right), Parietal ()
nV = 10020 # need to manually look up from the whole-brain plot

input_data = 'wholebrain'

if input_data == 'sensor':
    duple = np.load(root_path + 'me2_meg_analysis/' + age + '_group_03_sensor.npy',allow_pickle=True)
    triple = np.load(root_path + 'me2_meg_analysis/' + age + '_group_04_sensor.npy',allow_pickle=True)
elif input_data == 'ROI':
    duple = np.load(root_path + 'me2_meg_analysis/' + age + '_group_03_stc_mne_roi.npy',allow_pickle=True)
    triple = np.load(root_path + 'me2_meg_analysis/' + age + '_group_04_stc_mne_roi.npy',allow_pickle=True)
elif input_data == 'wholebrain':
    duple = np.load(root_path + 'me2_meg_analysis/' + age + '_group_03_stc_mne.npy',allow_pickle=True)
    triple = np.load(root_path + 'me2_meg_analysis/' + age + '_group_04_stc_mne.npy',allow_pickle=True)
else:
    print("Need to decide whether to use ROI or whole brain as feature.")
   
stc1 = mne.read_source_estimate('/media/tzcheng/storage/BabyRhythm/br_03/sss_fif/br_03_01_stc_lcmv_morph-vl.stc')
times = stc1.times
src = mne.read_source_spaces(subjects_dir + 'fsaverage/bem/fsaverage-vol-5-src.fif')

all_score = []
## Two way classification using ovr
X = np.concatenate((duple,triple),axis=0)
y = np.concatenate((np.repeat(0,len(duple)),np.repeat(1,len(triple))))

rand_ind = np.arange(0,len(X))
random.Random(15).shuffle(rand_ind)
X = X[rand_ind,:,:]
y = y[rand_ind]

clf = make_pipeline(
    StandardScaler(),  # z-score normalization
    LogisticRegression(solver="liblinear")  # liblinear is faster than lbfgs
)    
for n in np.arange(0,np.shape(X)[1],1):
        scores = cross_val_multiscore(clf, X[:,n,:], y, cv=5) # takes about 10 mins to run
        score = np.mean(scores, axis=0)
        print("Data " + str(n+1) + " Accuracy: %0.1f%%" % (100 * score,))
        all_score.append(score)
np.save(root_path + 'me2_meg_analysis/decoding/'+ age + '_decoding_accuracy_' + input_data +'.npy',all_score)

#%%##### Downsample
# Time-frequency analysis, decoding analysis are too computational heavy to run on wholebrain data 
fs_new = 1000
num = int((len(audio)*fs_new)/fs)        
audio = signal.resample(audio, num, t=None, axis=0, window=None)