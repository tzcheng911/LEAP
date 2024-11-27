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
import pandas as pd
import random
import os 
import time

import mne
from mne.datasets import somato
from mne.time_frequency import tfr_morlet, tfr_multitaper, tfr_stockwell, AverageTFRArray
from scipy.io import wavfile
from scipy import stats,signal
from scipy.stats import pearsonr
from mne_connectivity import spectral_connectivity_epochs, spectral_connectivity_time
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
run = '_04' # '_01','_02','_03','_04' silence, random, duple, triple

subjects_dir = '/media/tzcheng/storage2/subjects/'
root_path='/media/tzcheng/storage/ME2_MEG/Zoe_analyses/'
fname_aseg = subjects_dir + 'fsaverage/mri/aparc+aseg.mgz'
label_names = np.asarray(mne.get_volume_labels_from_aseg(fname_aseg))
nROI = [72,108,66,102,64,100,59,95,7,8,9,16,26,27,28,31,60,61,62,96,97,98,50,86] 
# Auditory (STG 72,108), Motor (precentral 66 102), Sensorimotor (postcentral 64 100), and between them is paracentral 59, 95
# Basal ganglia group (7,8,9,16,26,27,28,31): putamen is most relevant 8 27
# Frontal IFG (60,61,62,96,97,98)
# Parietal: inferior parietal (50 86), posterior parietal (??)
nV = 10020 # need to manually look up from the whole-brain plot

# fs, audio = wavfile.read(root_path + 'Stimuli/Random.wav') # Random, Duple300, Triple300
# MEG_sensor = np.load(root_path + 'me2_meg_analysis/' + age + '_group' + run + '_sensor.npy') # 01,02,03,04
# MEG_v = np.load(root_path + 'me2_meg_analysis/' + age + '_group' + run + '_stc_rs_mne.npy') # 01,02,03,04    
MEG_roi = np.load(root_path + 'me2_meg_analysis/' + age + '_group' + run + '_stc_mne_roi.npy') # 01,02,03,04

stc1 = mne.read_source_estimate('/media/tzcheng/storage/BabyRhythm/br_03/sss_fif/br_03_01_stc_lcmv_morph-vl.stc')
src = mne.read_source_spaces(subjects_dir + 'fsaverage/bem/fsaverage-vol-5-src.fif')
times = stc1.times

# stc1.data = MEG_v.mean(axis=0)
# stc1.plot(src = src,clim=dict(kind="value",lims=[5,5.5,8]))

#%%####################################### Frequency analysis
# Are there peak amplitude in the beat and meter rates? which ROI, which source?
fmin = 0.5
fmax = 5
n_lines = 10
n_freq = [33] # [6,7] 1.1 Hz, [12, 13] 1.6 Hz, [33] 3.33 Hz 
n_times = np.shape(MEG_v)[-1]
if n_times == 12001:
    MEG_fs = 1000 ## need a better way to write this, can mess up with the code later
elif n_times == 3000:
    MEG_fs = 250

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
evo_spectrum = evokeds.compute_psd('welch', fmin = fmin, fmax=5)
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
plot_err(psds.mean(0),'k',freqs)
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
evo_tfr = evokeds.compute_tfr('morlet', n_cycles=3,freqs=np.arange(fmin, 40, 2))
evo_tfr.plot_topo(baseline=(-0.5, 0), mode="logratio", title="Average power")
evo_tfr.plot(picks=[82], baseline=(-0.5, 0), mode="logratio", title=evo_tfr.ch_names[82])
tfr, freqs = evo_tfr.get_data(return_freqs=True)

## ROI
epochs = mne.read_epochs(root_path + '7mo/me2_205_7m/sss_fif/me2_205_7m_01_otp_raw_sss_proj_fil50_epoch.fif')
source_tfr = mne.time_frequency.tfr_array_morlet(MEG_roi,MEG_fs,freqs=np.arange(0.5, 5, 0.5),n_cycles=3,output='power')

# shitty imshow
fig, ax = plt.subplots(1,1)
im = plt.imshow(source_tfr.mean(0)[nROI[1],:,:],aspect = 'auto', origin='lower', cmap='jet')
ax.set_xticks(np.linspace(0,len(epochs.times),10))
ax.set_xticklabels(np.linspace(min(epochs.times),max(epochs.times),10,dtype = int), rotation=30)
ax.set_yticks(np.linspace(0,len(np.arange(0.5, 5, 0.5)),10))
ax.set_yticklabels(np.linspace(min(np.arange(0.5, 5, 0.5)),max(np.arange(0.5, 5, 0.5)),10), rotation=30)
plt.title(label_names[nROI[1]])

## wholebrain: too big to run
source_tfr = mne.time_frequency.tfr_array_morlet(MEG_v,MEG_fs,freqs=np.arange(1, 40, 2),n_cycles=3,output='power')

#%%####################################### Connectivity analysis
tic = time.time()
# How are ROI connected, which direction?
con_methods = ["pli", "dpli", "plv", "coh"]

# across subjects
con = spectral_connectivity_epochs( # Compute frequency- and time-frequency-domain connectivity measures
    MEG_roi[:,nROI,:],
    method=con_methods,
    mode="multitaper", # if using cwt_morlet, add cwt_freqs = nfreq = np.array([1,2,3,4,5])
    sfreq=MEG_fs,
    fmin=fmin,
    fmax=fmax,
    faverage=False,
    mt_adaptive=True,
    n_jobs=1,
)

# across time for each subject (this one makes more sense)
# con = spectral_connectivity_time(  # Compute frequency- and time-frequency-domain connectivity measures
#     MEG_roi[:,nROI,:],
#     method=con_methods,
#     # if using cwt_morlet, add cwt_freqs = nfreq = np.array([1,2,3,4,5])
#     mode="multitaper",
#     sfreq=MEG_fs,
#     fmin=fmin,
#     fmax=fmax,
#     freqs = np.arange(1,30,1),
#     faverage=False,
#     n_jobs=1,
# )
toc = time.time()
print('It takes ' + str((toc - tic)/60) + 'min to run wholebrain connectivity')


# Extract the data
con_res = dict()
for method, c in zip(con_methods, con):
    con_res[method] = c.get_data(output="dense").mean(axis = -1) # get the n freq
    
## visualization
ROI_names = label_names[nROI]
labels = mne.read_labels_from_annot("sample", parc="aparc", subjects_dir=subjects_dir)
label_colors = [label.color for label in labels]

node_order = list()
node_order.extend(ROI_names)  # reverse the order
node_angles = circular_layout(
    ROI_names, node_order, start_pos=90, group_boundaries=[0, len(ROI_names) / 2])

#%%
fig, ax = plt.subplots(figsize=(8, 8), facecolor="black", subplot_kw=dict(polar=True))
plot_connectivity_circle(
    con_res["plv"],
    ROI_names,
    n_lines=n_lines, # plot the top n lines
    node_angles=node_angles,
    node_colors=label_colors,
    title="All-to-All Connectivity Triple PLV",
    ax=ax)
fig.tight_layout()

#%%####################################### Decoding analysis
age = '11mo' # '7mo' or '7mo_0_15' or '7mo_15_32' or '11mo' or 'br' for adults
subjects_dir = '/media/tzcheng/storage2/subjects/'
root_path='/media/tzcheng/storage/ME2_MEG/Zoe_analyses/'
fname_aseg = subjects_dir + 'fsaverage/mri/aparc+aseg.mgz'
label_names = np.asarray(mne.get_volume_labels_from_aseg(fname_aseg))
nROI = [72,108,66,102,59,95,7,8,9,16,26,27,28,31] # Auditory (STG 72,108), Motor (precentral 66 102 and paracentral 59, 95), 
# Basal ganglia group (7,8,9,16,26,27,28,31) (left and then right), Parietal (), Frontal (IFG:  'ctx-rh-parsopercularis', 'ctx-rh-parsorbitalis',
# 'ctx-rh-parstriangularis)
nV = 10020 # need to manually look up from the whole-brain plot

input_data = 'wholebrain'

if input_data == 'sensor':
    duple = np.load(root_path + 'me2_meg_analysis/' + age + '_group_03_sensor.npy',allow_pickle=True)
    triple = np.load(root_path + 'me2_meg_analysis/' + age + '_group_04_sensor.npy',allow_pickle=True)
elif input_data == 'ROI':
    duple = np.load(root_path + 'me2_meg_analysis/' + age + '_group_03_stc_mne_roi.npy',allow_pickle=True)
    triple = np.load(root_path + 'me2_meg_analysis/' + age + '_group_04_stc_mne_roi.npy',allow_pickle=True)
elif input_data == 'wholebrain':
    duple = np.load(root_path + 'me2_meg_analysis/' + age + '_group_03_stc_rs_mne.npy',allow_pickle=True) # rs: resample data (fs = 250)
    triple = np.load(root_path + 'me2_meg_analysis/' + age + '_group_04_stc_rs_mne.npy',allow_pickle=True)
else:
    print("Need to decide whether to use ROI or whole brain as feature.")
   
stc1 = mne.read_source_estimate('/media/tzcheng/storage/BabyRhythm/br_03/sss_fif/br_03_01_stc_lcmv_morph-vl.stc')
times = stc1.times
src = mne.read_source_spaces(subjects_dir + 'fsaverage/bem/fsaverage-vol-5-src.fif')

all_score = []
## Two way classification using ovr
X = np.concatenate((duple,triple),axis=0)
y = np.concatenate((np.repeat(0,len(duple)),np.repeat(1,len(triple))))
del duple, triple
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
np.save(root_path + 'me2_meg_analysis/decoding/'+ age + '_decoding_accuracy_' + input_data +'_rs.npy',all_score)

## visualize the wholebrain decoding
acc = np.load(root_path + 'me2_meg_analysis/decoding/br_decoding_accuracy_wholebrain_rs.npy')
fake_data = np.zeros([len(acc),2])
fake_data[:,0] = acc
fake_data[:,1] = acc
stc1.data = fake_data
stc1.plot(src=src)

#%%##### Correlation analysis between neural responses and CDI
## Extract variables

## Check the subjects who have CDI 
CDI_WS = pd.read_excel(root_path + 'me2_meg_analysis/ME2_WG & WS Report_2023_09_07.xlsx',sheet_name=2)
corr_p = pearsonr(MEG, CDI_WS)
