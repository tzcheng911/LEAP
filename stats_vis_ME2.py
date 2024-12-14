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
import mne
import scipy.stats as stats
from scipy import stats,signal
from mne import spatial_src_adjacency
from mne.stats import spatio_temporal_cluster_1samp_test, summarize_clusters_stc
from mne.time_frequency import tfr_morlet, tfr_multitaper, tfr_stockwell, AverageTFRArray
from mne_connectivity import spectral_connectivity_epochs, spectral_connectivity_time
from mne_connectivity.viz import plot_connectivity_circle
from mne.viz import circular_layout

import sklearn 
import matplotlib.pyplot as plt 
from scipy.io import wavfile

def plot_err(group_stc,color,t):
    group_avg=np.mean(group_stc,axis=0)
    err=np.std(group_stc,axis=0)/np.sqrt(group_stc.shape[0])
    up=group_avg+err
    lw=group_avg-err
    plt.plot(t,group_avg,color=color)
    plt.fill_between(t,up,lw,color=color,alpha=0.5)
    
root_path='/media/tzcheng/storage/ME2_MEG/Zoe_analyses/'
subjects_dir = '/media/tzcheng/storage2/subjects/'
stc1 = mne.read_source_estimate('/media/tzcheng/storage/BabyRhythm/br_03/sss_fif/br_03_01_stc_lcmv_morph-vl.stc')
src = mne.read_source_spaces(subjects_dir + 'fsaverage/bem/fsaverage-vol-5-src.fif')
times = stc1.times

fmin = 0.5
fmax = 5

#%%####################################### Load the audio files
fs, audio_duple = wavfile.read(root_path + 'Stimuli/Duple300.wav') # Random, Duple300, Triple300
fs, audio_triple = wavfile.read(root_path + 'Stimuli/Triple300.wav') # Random, Duple300, Triple300

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
nROI = [72,108,66,102,64,100,59,95,7,8,9,16,26,27,28,31,60,61,62,96,97,98,50,86] 
# Auditory (STG 72,108), Motor (precentral 66 102), Sensorimotor (postcentral 64 100), and between them is paracentral 59, 95
# Basal ganglia group (7,8,9,16,26,27,28,31): putamen is most relevant 8 27
# Frontal IFG (60,61,62,96,97,98)
# Parietal: inferior parietal (50 86), posterior parietal (??)
MEG_random = np.load(root_path + 'me2_meg_analysis/' + age + '_group_02_stc_rs_mne_mag6pT_roi.npy') # 01,02,03,04
MEG_random_duple = np.load(root_path + 'me2_meg_analysis/' + age + '_group_02_stc_rs_mne_mag6pT_randduple_roi.npy') # 01,02,03,04
MEG_random_triple = np.load(root_path + 'me2_meg_analysis/' + age + '_group_02_stc_rs_mne_mag6pT_randtriple_roi.npy') # 01,02,03,04
MEG_duple = np.load(root_path + 'me2_meg_analysis/' + age + '_group_03_stc_rs_mne_mag6pT_roi.npy') # 01,02,03,04
MEG_triple = np.load(root_path + 'me2_meg_analysis/' + age + '_group_04_stc_rs_mne_mag6pT_roi.npy') # 01,02,03,04

########################################## SSEP
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
X = psds_random_duple-psds_random_triple
t,p = stats.ttest_1samp(X[:,:,[6,7]].mean(axis=2),0) # duple vs. random in beat 3.3 Hz, meter 1.67 Hz
idx = np.where(p < 0.05)
print(label_names[idx])
t,p = stats.ttest_1samp(X[:,:,[12,13]].mean(axis=2),0) # duple vs. random in beat 3.3 Hz, meter 1.67 Hz
idx = np.where(p < 0.05)
print(label_names[idx])
t,p = stats.ttest_1samp(X[:,:,[30,31]].mean(axis=2),0) # duple vs. random in beat 3.3 Hz, meter 1.67 Hz
idx = np.where(p < 0.05)
print(label_names[idx])

########################################## non-paramatric permutation test on random_duple vs. random_triple
for n in nROI:
    print("cluster " + label_names[n])
    T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(X[:,n,:], seed = 0)
    good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
    print("Find " + str(len(good_cluster_inds)) + " significant cluster")
    for i in np.arange(0,len(good_cluster_inds),1):
        print("The " + str(i+1) + "st significant cluster")
        print(clusters[good_cluster_inds[i]])
        print(freqs[clusters[good_cluster_inds[i]]])

    
#%%######################################### paramatric ttest test on meter vs. random ***
FOI = [6,7, 12,13, 30,31] # beat rate, meter rate, mu, beta

X = psds_duple-psds_random
t,p = stats.ttest_1samp(X[:,:,[6,7]].mean(axis=2),0) # duple vs. random in beat 3.3 Hz, meter 1.67 Hz
idx = np.where(p < 0.05)
print(label_names[idx])
t,p = stats.ttest_1samp(X[:,:,[12,13]].mean(axis=2),0) # duple vs. random in beat 3.3 Hz, meter 1.67 Hz
idx = np.where(p < 0.05)
print(label_names[idx])
t,p = stats.ttest_1samp(X[:,:,[30,31]].mean(axis=2),0) # duple vs. random in beat 3.3 Hz, meter 1.67 Hz
idx = np.where(p < 0.05)
print(label_names[idx])

X = psds_triple-psds_random
t,p = stats.ttest_1samp(X[:,:,[6,7]].mean(axis=2),0) # duple vs. random in beat 3.3 Hz, meter 1.67 Hz
idx = np.where(p < 0.05)
print(label_names[idx])
t,p = stats.ttest_1samp(X[:,:,[12,13]].mean(axis=2),0) # duple vs. random in beat 3.3 Hz, meter 1.67 Hz
idx = np.where(p < 0.05)
print(label_names[idx])
t,p = stats.ttest_1samp(X[:,:,[30,31]].mean(axis=2),0) # duple vs. random in beat 3.3 Hz, meter 1.67 Hz
idx = np.where(p < 0.05)
print(label_names[idx])

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

########################################## ERSP
vmin, vmax = -3.0,3.0
freqs=np.arange(5, 35, 1)

epochs = mne.read_epochs(root_path + '7mo/me2_205_7m/sss_fif/me2_205_7m_01_otp_raw_sss_proj_fil50_epoch.fif')
epochs.resample(MEG_fs)
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

X = power_duple - power_random
tfr_duple_random = AverageTFRArray(
    info=epochs.info, data=X.mean(axis=0), times=epochs.times, freqs=freqs, nave=np.shape(power_triple)[0])

for n in nROI: 
    # tfr_random.plot(
    #     baseline=(-0.5, 0),
    #     picks=[n],
    #     mode="percent",
    #     vlim = (vmin, vmax),
    #     title="Random TFR of ROI " + label_names[n])
    # tfr_duple.plot(
    #     baseline=(-0.5, 0),
    #     picks=[n],
    #     mode="percent",
    #     vlim = (vmin, vmax),
    #     title="Duple TFR of ROI " + label_names[n])
    # tfr_duple_random.plot(
    #     baseline=(-0.5, 0),
    #     picks=[n],
    #     mode="percent",
    #     vlim = (-100, 100),
    #     title="Duple - Random TFR of ROI " + label_names[n])
########################################## a-priori paramatric window ttest alpha (8-12 Hz), beta (15-30 Hz)
    t,p = stats.ttest_1samp(X[:,n,3:7,:].mean(axis=1).mean(axis=1),0) # alpha 8-12 Hz
    if p < 0.05: 
        print("Alpha band p-value of ROI " + label_names[n] + ": " + str(p))
    t,p = stats.ttest_1samp(X[:,n,15:30,:].mean(axis=1).mean(axis=1),0) # beta 15-30 Hz
    if p < 0.05: 
        print("Beta band p-value of ROI " + label_names[n] + ": " + str(p))

########################################## non-paramatric permutation test
    print('Clustering.')
    p_threshold = 0.001
    df = np.shape(MEG_duple)[0] - 1  # degrees of freedom for the test
    t_threshold = stats.distributions.t.ppf(1 - p_threshold / 2, df=df)
    
########################################## non-paramatric permutation test ***
    X = np.transpose(X,(0,3,2,1)) # subj, time, space
    T_obs, clusters, cluster_p_values, H0 = clu = mne.stats.spatio_temporal_cluster_1samp_test(
        X,
        seed=0,
        n_jobs=None,
        threshold=t_threshold,
        buffer_size=None,
        verbose=True,)

########################################## Connectivity
con_methods = ["pli", "dpli", "plv", "coh"]
con = spectral_connectivity_epochs( # Compute frequency- and time-frequency-domain connectivity measures
    MEG_duple[:,nROI,:],
    method=con_methods,
    mode="multitaper", # if using cwt_morlet, add cwt_freqs = nfreq = np.array([1,2,3,4,5])
    sfreq=MEG_fs,
    fmin=fmin,
    fmax=fmax,
    faverage=False,
    mt_adaptive=True,
    n_jobs=1,
)
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

fig, ax = plt.subplots(figsize=(8, 8), facecolor="black", subplot_kw=dict(polar=True))
plot_connectivity_circle(
    con_res["dpli"],
    ROI_names,
    n_lines=10, # plot the top n lines
    node_angles=node_angles,
    node_colors=label_colors,
    title="All-to-All Connectivity Triple PLV",
    ax=ax)
fig.tight_layout()

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


