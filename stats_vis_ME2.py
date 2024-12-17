#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 09:54:23 2024
Visualize the figures in the paper 
Figure 1: acoustic spectrum, sensor level spectrum for the four conditions ()
Figure 2
Figure 3
Figure 4
Figure 5
Figure 6
 
@author: tzcheng
"""

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

def plot_err(group_stc,color,t):
    group_avg=np.mean(group_stc,axis=0)
    err=np.std(group_stc,axis=0)/np.sqrt(group_stc.shape[0])
    up=group_avg+err
    lw=group_avg-err
    plt.plot(t,group_avg,color=color)
    plt.fill_between(t,up,lw,color=color,alpha=0.5)

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

root_path='/media/tzcheng/storage/ME2_MEG/Zoe_analyses/'
subjects_dir = '/media/tzcheng/storage2/subjects/'
stc1 = mne.read_source_estimate('/media/tzcheng/storage/BabyRhythm/br_03/sss_fif/br_03_01_stc_mne_morph_mag6pT-vl.stc')
src = mne.read_source_spaces(subjects_dir + 'fsaverage/bem/fsaverage-vol-5-src.fif')
epochs = mne.read_epochs(root_path + '7mo/me2_205_7m/sss_fif/me2_205_7m_01_otp_raw_sss_proj_fil50_mag6pT_epoch.fif')
epochs.resample(250)
times = epochs.times

fmin = 0.5
fmax = 5

#%%####################################### Load the audio files
fs, audio_duple = wavfile.read(root_path + 'Stimuli/Duple300.wav') # Random, Duple300, Triple300
fs, audio_triple = wavfile.read(root_path + 'Stimuli/Triple300.wav') # Random, Duple300, Triple300

plt.figure()
plt.plot(np.linspace(0,audio_duple.size/fs,audio_duple.size),audio_duple)

plt.figure()
plt.plot(np.linspace(0,audio_triple.size/fs,audio_triple.size),audio_triple)

psds, freqs = mne.time_frequency.psd_array_welch(
audio_duple,fs, 
n_fft=len(audio_duple),
n_overlap=0,
n_per_seg=None,
fmin=fmin,
fmax=fmax,)

plt.figure()
plt.plot(freqs,psds)
plt.title('Duple rhythm')

psds, freqs = mne.time_frequency.psd_array_welch(
audio_triple,fs, 
n_fft=len(audio_triple),
n_overlap=0,
n_per_seg=None,
fmin=fmin,
fmax=fmax,)

plt.figure()
plt.plot(freqs,psds)
plt.title('Triple rhythm')

#%%####################################### Load the sensor files
## SSEP

age = 'br' # '7mo', '11mo', 'br' for adults
MEG_fs = 1000
pool_fbin = 5

MEG_random = np.load(root_path + 'me2_meg_analysis/' + age + '_group_02_mag6pT_sensor.npy') # 01,02,03,04
MEG_random_duple = np.load(root_path + 'me2_meg_analysis/' + age + '_group_02_mag6pT_randduple_sensor.npy') # 01,02,03,04
MEG_random_triple = np.load(root_path + 'me2_meg_analysis/' + age + '_group_02_mag6pT_randtriple_sensor.npy') # 01,02,03,04
MEG_duple = np.load(root_path + 'me2_meg_analysis/' + age + '_group_03_mag6pT_sensor.npy') # 01,02,03,04
MEG_triple = np.load(root_path + 'me2_meg_analysis/' + age + '_group_04_mag6pT_sensor.npy') # 01,02,03,04

########################################## SSEP
psds_random, freqs = mne.time_frequency.psd_array_welch(
MEG_random_duple,MEG_fs, # could replace with label time series
n_fft=np.shape(MEG_random_duple)[2],
n_overlap=0,
n_per_seg=None,
fmin=fmin,
fmax=fmax,)

plt.figure()
plot_err(psds_random.mean(axis=1),'k',freqs)
plt.title('Random')

psds_random_duple, freqs = mne.time_frequency.psd_array_welch(
MEG_random_duple,MEG_fs, # could replace with label time series
n_fft=np.shape(MEG_random_duple)[2],
n_overlap=0,
n_per_seg=None,
fmin=fmin,
fmax=fmax,)

plt.figure()
plot_err(psds_random_duple.mean(axis=1),'k',freqs)
plt.title('Random duple')

psds_random_triple, freqs = mne.time_frequency.psd_array_welch(
MEG_random_triple,MEG_fs, # could replace with label time series
n_fft=np.shape(MEG_random_triple)[2],
n_overlap=0,
n_per_seg=None,
fmin=fmin,
fmax=fmax,)

plt.figure()
plot_err(psds_random_triple.mean(axis=1),'k',freqs)
plt.title('Random triple')

psds_duple, freqs = mne.time_frequency.psd_array_welch(
MEG_duple,MEG_fs, # could replace with label time series
n_fft=np.shape(MEG_duple)[2],
n_overlap=0,
n_per_seg=None,
fmin=fmin,
fmax=fmax,)

plt.figure()
plot_err(psds_duple.mean(axis=1),'k',freqs)
plt.title('Duple')

psds_triple, freqs = mne.time_frequency.psd_array_welch(
MEG_triple,MEG_fs, # could replace with label time series
n_fft=np.shape(MEG_triple)[2],
n_overlap=0,
n_per_seg=None,
fmin=fmin,
fmax=fmax,)

plt.figure()
plot_err(psds_triple.mean(axis=1),'k',freqs)
plt.title('Triple')

########################################## paramatric ttest test on random_duple vs. random_triple
FOI = [6,7, 12,13, 30,31] # beat rate, meter rate, mu, beta
X = psds_random_duple.mean(axis=1)-psds_random_triple.mean(axis=1)
t,p = stats.ttest_1samp(X[:,[6,7]].mean(axis=1),0) # duple vs. random in beat 3.3 Hz, meter 1.67 Hz
print(p)
t,p = stats.ttest_1samp(X[:,[12,13]].mean(axis=1),0) # duple vs. random in beat 3.3 Hz, meter 1.67 Hz
print(p)
t,p = stats.ttest_1samp(X[:,[30,31]].mean(axis=1),0) # duple vs. random in beat 3.3 Hz, meter 1.67 Hz
print(p)

########################################## non-paramatric permutation test on random_duple vs. random_triple
T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(X, seed = 0)
good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
for i in np.arange(0,len(good_cluster_inds),1):
    print("The " + str(i+1) + "st significant cluster")
    print(clusters[good_cluster_inds[i]])
    print(freqs[clusters[good_cluster_inds[i]]])
    
########################################## paramatric ttest test on meter vs. random ***
X = psds_duple.mean(axis=1)-psds_random.mean(axis=1)
t,p = stats.ttest_1samp(X[:,[6,7]].mean(axis=1),0) # duple vs. random in beat 3.3 Hz, meter 1.67 Hz
print(p)
t,p = stats.ttest_1samp(X[:,[12,13]].mean(axis=1),0) # duple vs. random in beat 3.3 Hz, meter 1.67 Hz
print(p)
t,p = stats.ttest_1samp(X[:,[30,31]].mean(axis=1),0) # duple vs. random in beat 3.3 Hz, meter 1.67 Hz
print(p)

X = psds_triple.mean(axis=1)-psds_random.mean(axis=1) 
t,p = stats.ttest_1samp(X[:,[6,7]].mean(axis=1),0) # triple vs. random in beat 3.3 Hz, meter 1.1 Hz
print(p)
t,p = stats.ttest_1samp(X[:,[12,13]].mean(axis=1),0) # triple vs. random in beat 3.3 Hz, meter 1.1 Hz
print(p)
t,p = stats.ttest_1samp(X[:,[30,31]].mean(axis=1),0) # triple vs. random in beat 3.3 Hz, meter 1.1 Hz
print(p)

########################################## non-paramatric permutation test ***
X = psds_duple.mean(axis=1)-psds_random.mean(axis=1)
T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(X, seed = 0)
good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
for i in np.arange(0,len(good_cluster_inds),1):
    print("The " + str(i+1) + "st significant cluster")
    print(clusters[good_cluster_inds[i]])
    print(freqs[clusters[good_cluster_inds[i]]])

X = psds_triple.mean(axis=1)-psds_random.mean(axis=1)
T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(X, seed = 0)
good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
for i in np.arange(0,len(good_cluster_inds),1):
    print("The " + str(i+1) + "st significant cluster")
    print(clusters[good_cluster_inds[i]])
    print(freqs[clusters[good_cluster_inds[i]]])

#%%####################################### Load the source ROI files
## SSEP
## ERSP
## decoding
## Conn

age = 'br' # '7mo', '11mo', 'br' for adults
MEG_fs = 250

fname_aseg = subjects_dir + 'fsaverage/mri/aparc+aseg.mgz'
label_names = np.asarray(mne.get_volume_labels_from_aseg(fname_aseg))
nROI = [72,108,66,102,64,100,59,95,7,8,26,27,60,61,62,96,97,98,50,86,71,107] 
# Auditory (STG 72,108), Motor (precentral 66 102), Sensorimotor (postcentral 64 100), and between them is paracentral 59, 95
# Basal ganglia group (7,8,9,16,26,27,28,31): out of all include caudate (7 26) and putamen (8 27) only based on Cannon & Patel 2020 TICS, putamen is most relevant 
# Frontal IFG (60,61,62,96,97,98)
# Posterior Parietal: inferior parietal (50 86),  superior parietal (71 107)
MEG_random = np.load(root_path + 'me2_meg_analysis/' + age + '_group_02_stc_rs_mne_mag6pT_roi.npy') # 01,02,03,04
MEG_random_duple = np.load(root_path + 'me2_meg_analysis/' + age + '_group_02_stc_rs_mne_mag6pT_randduple_roi.npy') # 01,02,03,04
MEG_random_triple = np.load(root_path + 'me2_meg_analysis/' + age + '_group_02_stc_rs_mne_mag6pT_randtriple_roi.npy') # 01,02,03,04
MEG_duple = np.load(root_path + 'me2_meg_analysis/' + age + '_group_03_stc_rs_mne_mag6pT_roi.npy') # 01,02,03,04
MEG_triple = np.load(root_path + 'me2_meg_analysis/' + age + '_group_04_stc_rs_mne_mag6pT_roi.npy') # 01,02,03,04

#%%######################################### SSEP
psds_random, freqs = mne.time_frequency.psd_array_welch(
MEG_random,MEG_fs, # could replace with label time series
n_fft=np.shape(MEG_random)[2],
n_overlap=0,
n_per_seg=None,
fmin=fmin,
fmax=fmax,)

psds_random_duple, freqs = mne.time_frequency.psd_array_welch(
MEG_random_duple,MEG_fs, # could replace with label time series
n_fft=np.shape(MEG_random_duple)[2],
n_overlap=0,
n_per_seg=None,
fmin=fmin,
fmax=fmax,)

psds_random_triple, freqs = mne.time_frequency.psd_array_welch(
MEG_random_triple,MEG_fs, # could replace with label time series
n_fft=np.shape(MEG_random_triple)[2],
n_overlap=0,
n_per_seg=None,
fmin=fmin,
fmax=fmax,)

psds_duple, freqs = mne.time_frequency.psd_array_welch(
MEG_duple,MEG_fs, # could replace with label time series
n_fft=np.shape(MEG_duple)[2],
n_overlap=0,
n_per_seg=None,
fmin=fmin,
fmax=fmax,)

psds_triple, freqs = mne.time_frequency.psd_array_welch(
MEG_triple,MEG_fs, # could replace with label time series
n_fft=np.shape(MEG_triple)[2],
n_overlap=0,
n_per_seg=None,
fmin=fmin,
fmax=fmax,)

########################################## paramatric ttest test on random_duple vs. random_triple
FOI = [6,7, 12,13, 30,31] # beat rate, meter rate, mu, beta
X = psds_random_duple.mean(axis=1)-psds_random_triple.mean(axis=1)
t,p = stats.ttest_1samp(X[:,[6,7]].mean(axis=1),0) # duple vs. random in beat 3.3 Hz, meter 1.67 Hz
idx = np.where(p < 0.05)
print(label_names[idx])
t,p = stats.ttest_1samp(X[:,[12,13]].mean(axis=1),0) # duple vs. random in beat 3.3 Hz, meter 1.67 Hz
idx = np.where(p < 0.05)
print(label_names[idx])
t,p = stats.ttest_1samp(X[:,[30,31]].mean(axis=1),0) # duple vs. random in beat 3.3 Hz, meter 1.67 Hz
idx = np.where(p < 0.05)
print(label_names[idx])

#%%######################################### paramatric ttest test on meter vs. random ***
FOI = [6,7, 12,13, 30,31] # beat rate, meter rate, mu, beta

X = np.squeeze(psds_duple[:,[nROI],:])-np.squeeze(psds_random[:,[nROI],:])
t,p = stats.ttest_1samp(X[:,:,[6,7]].mean(axis=2),0) # duple vs. random in beat 3.3 Hz, meter 1.67 Hz
idx = np.where(p < 0.05)
print(label_names[np.array(nROI)[idx]])
t,p = stats.ttest_1samp(X[:,:,[12,13]].mean(axis=2),0) # duple vs. random in beat 3.3 Hz, meter 1.67 Hz
idx = np.where(p < 0.05)
print(label_names[np.array(nROI)[idx]])
t,p = stats.ttest_1samp(X[:,:,[30,31]].mean(axis=2),0) # duple vs. random in beat 3.3 Hz, meter 1.67 Hz
idx = np.where(p < 0.05)
print(label_names[np.array(nROI)[idx]])

X = np.squeeze(psds_triple[:,[nROI],:])-np.squeeze(psds_random[:,[nROI],:])
t,p = stats.ttest_1samp(X[:,:,[6,7]].mean(axis=2),0) # duple vs. random in beat 3.3 Hz, meter 1.67 Hz
idx = np.where(p < 0.05)
print(label_names[np.array(nROI)[idx]])
t,p = stats.ttest_1samp(X[:,:,[12,13]].mean(axis=2),0) # duple vs. random in beat 3.3 Hz, meter 1.67 Hz
idx = np.where(p < 0.05)
print(label_names[np.array(nROI)[idx]])
t,p = stats.ttest_1samp(X[:,:,[30,31]].mean(axis=2),0) # duple vs. random in beat 3.3 Hz, meter 1.67 Hz
idx = np.where(p < 0.05)
print(label_names[np.array(nROI)[idx]])

########################################## non-paramatric permutation test ***
X = psds_duple-psds_random
for n in nROI:
    print("cluster " + label_names[n])
    T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(X[:,n,:], seed = 0)
    good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
    print("Find " + str(len(good_cluster_inds)) + " significant cluster")

    for i in np.arange(0,len(good_cluster_inds),1):
        print("The " + str(i+1) + "st significant cluster")
        print(clusters[good_cluster_inds[i]])
        print(freqs[clusters[good_cluster_inds[i]]])

X = psds_triple-psds_random
for n in nROI:
    print("cluster " + label_names[n])
    T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(X[:,n,:], seed = 0)
    good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
    print("Find " + str(len(good_cluster_inds)) + " significant cluster")

    for i in np.arange(0,len(good_cluster_inds),1):
        print("The " + str(i+1) + "st significant cluster")
        print(clusters[good_cluster_inds[i]])
        print(freqs[clusters[good_cluster_inds[i]]])

#%%######################################### ERSP
vmin, vmax = -1,1
freqs=np.arange(5, 35, 1)

epochs = mne.read_epochs(root_path + '7mo/me2_205_7m/sss_fif/me2_205_7m_01_otp_raw_sss_proj_fil50_mag6pT_epoch.fif')
epochs.resample(MEG_fs)
times = epochs.times
epochs.drop_channels(epochs.info["ch_names"][0:192]) # hack into the epochs
power_random = mne.time_frequency.tfr_array_morlet(MEG_random,MEG_fs,freqs=freqs,n_cycles=15,output='power')
power_duple = mne.time_frequency.tfr_array_morlet(MEG_duple,MEG_fs,freqs=freqs,n_cycles=15,output='power')
power_triple = mne.time_frequency.tfr_array_morlet(MEG_triple,MEG_fs,freqs=freqs,n_cycles=15,output='power')

tfr_random = AverageTFRArray(
    info=epochs.info, data=power_random.mean(axis=0), times=epochs.times, freqs=freqs, nave=np.shape(power_random)[0])
tfr_duple = AverageTFRArray(
    info=epochs.info, data=power_duple.mean(axis=0), times=epochs.times, freqs=freqs, nave=np.shape(power_duple)[0])
tfr_triple = AverageTFRArray(
    info=epochs.info, data=power_triple.mean(axis=0), times=epochs.times, freqs=freqs, nave=np.shape(power_triple)[0])

tfr_random.apply_baseline(mode='logratio', baseline=(0,None))
tfr_duple.apply_baseline(mode='logratio', baseline=(0,None))
tfr_triple.apply_baseline(mode='logratio', baseline=(0,None))

for n in nROI: 
    # tfr_random.plot(
    #     picks=[n],
    #     vlim = (vmin, vmax),
    #     title="Random TFR of ROI " + label_names[n])
    # tfr_duple.plot(
    #     picks=[n],
    #     vlim = (vmin, vmax),
    #     title="Duple TFR of ROI " + label_names[n])
    # tfr_triple.plot(
    #     baseline=(0,None),
    #     picks=[n],
    #     vlim = (vmin, vmax),
    #     title="Triple TFR of ROI " + label_names[n])
    plt.figure() # plot the power across time in frequency of interest 
    plot_err(power_duple[:,n,15:30,:].mean(axis = 1),'b',times)
    plot_err(power_triple[:,n,15:30,:].mean(axis = 1),'r',times)
    plot_err(power_random[:,n,15:30,:].mean(axis = 1),'k',times)
    plt.title(label_names[n])
    plt.xlim([-0.5,10])
    plt.ylim([0,20])
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
########################################## a-priori paramatric window ttest alpha (8-12 Hz), beta (15-30 Hz) 
# alpha 8-12 Hz *** only caudate and putamen reach marginal p = ~0.05
X = power_duple[:,nROI,3:7,:].mean(axis=2).mean(axis=2) - power_random[:,nROI,3:7,:].mean(axis=2).mean(axis=2)
t,p = stats.ttest_1samp(X,0) # alpha 8-12 Hz
idx = np.where(p < 0.055)
print(label_names[np.array(nROI)[idx]])

# beta 15-30 Hz
X = power_duple[:,nROI,15:30,:].mean(axis=2).mean(axis=2) - power_random[:,nROI,15:30,:].mean(axis=2).mean(axis=2)
t,p = stats.ttest_1samp(X,0) # alpha 8-12 Hz
idx = np.where(p < 0.055)
print(label_names[np.array(nROI)[idx]])

# alpha 8-12 Hz 
X = power_triple[:,nROI,3:7,:].mean(axis=2).mean(axis=2) - power_random[:,nROI,3:7,:].mean(axis=2).mean(axis=2)
t,p = stats.ttest_1samp(X,0) # alpha 8-12 Hz
idx = np.where(p < 0.055)
print(label_names[np.array(nROI)[idx]])

# beta 15-30 Hz
X = power_triple[:,nROI,15:30,:].mean(axis=2).mean(axis=2) - power_random[:,nROI,15:30,:].mean(axis=2).mean(axis=2)
t,p = stats.ttest_1samp(X,0) # alpha 8-12 Hz
idx = np.where(p < 0.07)
print(label_names[np.array(nROI)[idx]])

########################################## non-paramatric permutation test 
# Duple: None really as epected (Right-Putamen has lots of significnace in low freuency 5 or 6 Hz) sig_ROI = 108, 102, 100, 7, 8, 26, 27, 60
# Triple: Better, some effects on 7-10 Hz sig_ROI = 108, 100, 60, 96, 71
X = power_duple - power_random
X = power_triple - power_random
sig_ROI = []
for n in nROI:
    print("cluster " + label_names[n])
    T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(X[:,n,:,:], seed = 0)
    good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
    print("Find " + str(len(good_cluster_inds)) + " significant cluster")
    
    for i in np.arange(0,len(good_cluster_inds),1):
        print("The " + str(i+1) + "st significant cluster")
        print(freqs[clusters[good_cluster_inds[i]][0]]) # frequency window of significance
        print(times[clusters[good_cluster_inds[i]][1]]) # time window of significance
        
    if len(good_cluster_inds)>0:
        sig_ROI.append(n)
        
#%%######################################### Decoding None really is significant (95th percentile)
ts = 0 # onset of sound
te = 2350 # select the first 8.9s to include the 30 beats for both duple and triple meters
all_score = []
## Two way classification using ovr
X = np.concatenate((MEG_duple[:,nROI,:],MEG_triple[:,nROI,:]),axis=0)
y = np.concatenate((np.repeat(0,len(MEG_duple)),np.repeat(1,len(MEG_triple))))

rand_ind = np.arange(0,len(X))
random.Random(15).shuffle(rand_ind)
X = X[rand_ind,:,ts:te]
y = y[rand_ind]

clf = make_pipeline(
    StandardScaler(),  # z-score normalization
    SVC(kernel='rbf',gamma='auto',C=0.1)  
)
    
    # clf = make_pipeline(
    #     StandardScaler(),  # z-score normalization
    #     SVC(kernel='linear',gamma='auto')  
    # )
    
for n in np.arange(0,np.shape(X)[1],1):
    scores = cross_val_multiscore(clf, X[:,n,:], y, cv=np.shape(MEG_triple)[0]) # takes about 10 mins to run
    score = np.mean(scores, axis=0)
    print("Data " + str(n+1) + " Accuracy: %0.1f%%" % (100 * score,))
    all_score.append(score)
    
    ## Run permutation
    n_perm=1000
    scores_perm=[]
    for i in range(n_perm):
        yp = copy.deepcopy(y)
        random.shuffle(yp)
    
        # Run cross-validated decoding analyses:
        scores = cross_val_multiscore(clf, X[:,n,:], yp, cv=np.shape(MEG_triple)[0], n_jobs=None) # X can be MMR or cABR
        scores_perm.append(np.mean(scores,axis=0))
        # print("Iteration " + str(i))
    scores_perm_array=np.asarray(scores_perm)
    
    plt.figure()
    plt.hist(scores_perm_array,bins=15,color='grey')
    plt.vlines(score,ymin=0,ymax=1000,color='r',linewidth=2)
    # plt.vlines(np.percentile(scores_perm_array,90),ymin=0,ymax=1000,color='',linewidth=2)
    plt.vlines(np.percentile(scores_perm_array,95),ymin=0,ymax=1000,color='grey',linewidth=2)
    plt.ylabel('Count',fontsize=20)
    plt.xlabel('Accuracy',fontsize=20)
    plt.title(label_names[nROI[n]])
    print('Accuracy: ' + str(score))
    print('95%: ' + str(np.percentile(scores_perm_array,95)))

#%%######################################### Connectivity
fmin = 5
fmax = 35
freqs = np.linspace(fmin,fmax,380)
Freq_Bands = {"theta": [4.0, 8.0], "alpha": [8.0, 12.0], "beta": [12.0, 30.0]}

## gc
con_methods = ["gc"]
con = spectral_connectivity_time( # Compute frequency- and time-frequency-domain connectivity measures
    MEG_duple,
    indices = (np.array([[2, 3], [2, 3], [7, 26], [8,27], [0, 1], [71, 107],[2, 3],[2, 3]]),  # seeds
           np.array([[0, 1], [71, 107],[2, 3],[2, 3],[2, 3], [2, 3], [7, 26], [8,27]])),  # targets
    # testing motor -> auditory, motor -> parietal, caudate -> motor, putamen -> motor
    # and the other way aorund 
    freqs = freqs,
    method=con_methods,
    mode="multitaper", # if using cwt_morlet, add cwt_freqs = nfreq = np.array([1,2,3,4,5])
    sfreq=MEG_fs,
    fmin=fmin,
    fmax=fmax,
    faverage=False,
    n_jobs=1,
)
con.save('GC_ASAP')

con_res = con.get_data()
plot_err(con_res[:,0,:],'k',freqs)
plot_err(con_res[:,1,:],'r',freqs)

## non-directional connectivity
con_methods = ["plv","coh","pli"]
con = spectral_connectivity_time( # Compute frequency- and time-frequency-domain connectivity measures
    MEG_random[:,nROI,:],
    freqs=freqs,
    method=con_methods,
    mode="multitaper", # if using cwt_morlet, add cwt_freqs = nfreq = np.array([1,2,3,4,5])
    sfreq=MEG_fs,
    fmin=fmin,
    fmax=fmax,
    faverage=False,
    n_jobs=1,
)
con[0].save('br_rand_conn_plv')
con[1].save('br_rand_conn_coho')
con[2].save('br_rand_conn_pli')

# Extract the data
con_res = dict()
for method, c in zip(con_methods, con):
    con_res[method] = c.get_data(output="dense") # get the n freq
    
## visualization
ROI_names = label_names[nROI]
labels = mne.read_labels_from_annot("sample", parc="aparc", subjects_dir=subjects_dir)
label_colors = [label.color for label in labels]

node_order = list()
node_order.extend(ROI_names)  # reverse the order
node_angles = circular_layout(
    ROI_names, node_order, start_pos=90, group_boundaries=[0, len(ROI_names) / 2])

fig, ax = plt.subplots(figsize=(8, 8), facecolor="black", subplot_kw=dict(polar=True))
plot_connectivity_circle(
    con_res["plv"].mean(axis=0)[:,:,127:325].mean(axis=2), # change to the freqs of interest
    ROI_names,
    n_lines=20, # plot the top n lines
    node_angles=node_angles,
    node_colors=label_colors,
    title="All-to-All Connectivity Duple PLV Beta band",
    ax=ax)
fig.tight_layout()

temp_conn_duple = np.squeeze(con_res["plv"][:,:,:,127:325].mean(axis=3)) 
conn_duple = temp_conn_duple[:,1,0]
temp_conn_triple = np.squeeze(con_res["plv"][:,:,:,127:325].mean(axis=3)) 
conn_triple = temp_conn_triple[:,1,0]
temp_conn_random = np.squeeze(con_res["plv"][:,:,:,127:325].mean(axis=3)) 
conn_random = temp_conn_random[:,1,0]

    temp_conn = np.squeeze(con_res["plv"][:,-2:,:-2,:]) # [tg/bg,channel,freq]

test = np.where(p<0.05)

fig, ax = plt.subplots()
conn_contrast = np.squeeze(temp_conn[:,:,:10].mean(1)) 
im, cbar = heatmap(conn_contrast, label_names, con[0].freqs[:10], ax=ax,
                   cmap="jet_r", cbarlabel="EEG env Conn",aspect = 'auto',
                   vmin=-0.1,vmax=0.1)

########################################## paramatric ttest test on duple vs. random
X = psds_random_duple.mean(axis=1)-psds_random_triple.mean(axis=1)
t,p = stats.ttest_1samp(X[:,[6,7]].mean(axis=1),0) # duple vs. random in beat 3.3 Hz, meter 1.67 Hz
print(p)
t,p = stats.ttest_1samp(X[:,[12,13]].mean(axis=1),0) # duple vs. random in beat 3.3 Hz, meter 1.67 Hz
print(p)
t,p = stats.ttest_1samp(X[:,[30,31]].mean(axis=1),0) # duple vs. random in beat 3.3 Hz, meter 1.67 Hz
print(p)

########################################## non-paramatric permutation test on random_duple vs. random_triple
T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(X, seed = 0)
good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
for i in np.arange(0,len(good_cluster_inds),1):
    print("The " + str(i+1) + "st significant cluster")
    print(clusters[good_cluster_inds[i]])
    print(freqs[clusters[good_cluster_inds[i]]])


#%%####################################### Load the source wholebrain files
## SSEP
## ERSP
## decoding
## Conn

age = 'br' # '7mo', '11mo', 'br' for adults
MEG_fs = 250

MEG_random = np.load(root_path + 'me2_meg_analysis/' + age + '_group_02_stc_rs_mne_mag6pT_morph.npy') # 01,02,03,04    
MEG_random_duple = np.load(root_path + 'me2_meg_analysis/' + age + '_group_02_stc_rs_mne_mag6pT_randduple_morph.npy') # 01,02,03,04    
MEG_random_triple = np.load(root_path + 'me2_meg_analysis/' + age + '_group_02_stc_rs_mne_mag6pT_randtriple_morph.npy') # 01,02,03,04    
MEG_duple = np.load(root_path + 'me2_meg_analysis/' + age + '_group_03_stc_rs_mne_mag6pT_morph.npy') # 01,02,03,04    
MEG_triple = np.load(root_path + 'me2_meg_analysis/' + age + '_group_04_stc_rs_mne_mag6pT_morph.npy') # 01,02,03,04    

psds_random, freqs = mne.time_frequency.psd_array_welch(
MEG_random,MEG_fs, # could replace with label time series
n_fft=np.shape(MEG_random)[2],
n_overlap=0,
n_per_seg=None,
fmin=fmin,
fmax=fmax,)

psds_random_duple, freqs = mne.time_frequency.psd_array_welch(
MEG_random_duple,MEG_fs, # could replace with label time series
n_fft=np.shape(MEG_random_duple)[2],
n_overlap=0,
n_per_seg=None,
fmin=fmin,
fmax=fmax,)

psds_random_triple, freqs = mne.time_frequency.psd_array_welch(
MEG_random_triple,MEG_fs, # could replace with label time series
n_fft=np.shape(MEG_random_triple)[2],
n_overlap=0,
n_per_seg=None,
fmin=fmin,
fmax=fmax,)

psds_duple, freqs = mne.time_frequency.psd_array_welch(
MEG_duple,MEG_fs, # could replace with label time series
n_fft=np.shape(MEG_duple)[2],
n_overlap=0,
n_per_seg=None,
fmin=fmin,
fmax=fmax,)

psds_triple, freqs = mne.time_frequency.psd_array_welch(
MEG_triple,MEG_fs, # could replace with label time series
n_fft=np.shape(MEG_triple)[2],
n_overlap=0,
n_per_seg=None,
fmin=fmin,
fmax=fmax,)

#%%######################################### paramatric ttest test on meter vs. random 
FOI = [6,7, 12,13, 30,31] # beat rate, meter rate, mu, beta
print("Computing adjacency.")
adjacency = mne.spatial_src_adjacency(src)
p_threshold = 0.001
df = np.shape(MEG_duple)[0] - 1  # degrees of freedom for the test
t_threshold = stats.distributions.t.ppf(1 - p_threshold / 2, df=df)

print('Clustering.')
########################################## non-paramatric permutation test ***
fsave_vertices = [s["vertno"] for s in src]
X = psds_duple-psds_random
X = np.transpose(X,(0,2,1)) # subj, time, space

T_obs, clusters, cluster_p_values, H0 = clu = mne.stats.spatio_temporal_cluster_1samp_test(
    X,
    seed=0,
    adjacency=adjacency,
    n_jobs=None,
    threshold=t_threshold,
    buffer_size=None,
    verbose=True,
)

good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
good_clusters = [clusters[idx] for idx in good_cluster_inds]

stc_all_cluster_vis = summarize_clusters_stc(
    clu, p_thresh = p_threshold, tstep = np.diff(freqs)[0], tmin = freqs[0], vertices=fsave_vertices, subject="fsaverage"
)
stc_all_cluster_vis = stc_all_cluster_vis.plot(src=src) ## what is this actually plotting

stc1.tmin = freqs[0] # hack into the time with freqs
stc1.tstep = np.diff(freqs)[0] # hack into the time with freqs
stc1.plot(src = src)

X = psds_triple-psds_random
X = np.transpose(X,(0,2,1)) # subj, time, space

T_obs, clusters, cluster_p_values, H0 = clu = mne.stats.spatio_temporal_cluster_1samp_test(
    X,
    seed=0,
    adjacency=adjacency,
    n_jobs=None,
    threshold=t_threshold,
    buffer_size=None,
    verbose=True,
)

good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
good_clusters = [clusters[idx] for idx in good_cluster_inds]


