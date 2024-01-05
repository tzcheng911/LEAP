#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 11:13:47 2023
Statistical test on MMR1 vs. MMR2
1. traditional window method t-test
2. non-parametric cluster-based permutation t-test
3. decoding method see decoding_zoe.py 

@author: tzcheng
"""

import numpy as np
import time
import mne
import scipy.stats as stats
from scipy import stats,signal
from mne import spatial_src_adjacency
from mne.stats import spatio_temporal_cluster_1samp_test
import sklearn 
import matplotlib.pyplot as plt 
from scipy.io import wavfile

def plot_err(group_stc,color,t):
    group_avg=np.mean(group_stc,axis=0)
   #plt.figure()
    err=np.std(group_stc,axis=0)/np.sqrt(group_stc.shape[0])
    up=group_avg+err
    lw=group_avg-err
    t=np.linspace(-100,600,3501)
    plt.plot(t,group_avg,color=color)
    plt.fill_between(t,up,lw,color=color,alpha=0.5)

root_path='/media/tzcheng/storage/CBS/'
subjects_dir = '/media/tzcheng/storage2/subjects/'

ts = 1250 
te = 2000 

#%% paramatric permutation test on the time window 100 - 250 ms for MMR
## Load the MEG MMR data
# change to condition (traditional, new method etc.), methods (vector vs. magnitude vs. eeg), spatial (sensor, ROIs, whole brain) 
mmr1 = np.load(root_path + 'cbsA_meeg_analysis/MEG/magnitude_method/group_mmr1_mba_None_morph.npy') 
mmr2 = np.load(root_path + 'cbsA_meeg_analysis/MEG/magnitude_method/group_mmr2_pa_None_morph.npy') 
mmr1 = np.load(root_path + 'cbsA_meeg_analysis/MEG/vector_method/group_mmr1_mba_vector_morph.npy') 
mmr2 = np.load(root_path + 'cbsA_meeg_analysis/MEG/vector_method/group_mmr2_pa_vector_morph.npy') 

X = mmr2-mmr1
X = X[:,:,ts:te].mean(axis=1).mean(axis=1)
stats.ttest_1samp(X,0)

## Load the EEG MMR data: only the new method showed a significance
mmr1 = np.load(root_path + 'cbsA_meeg_analysis/EEG/group_mmr1_eeg.npy') 
mmr2 = np.load(root_path + 'cbsA_meeg_analysis/EEG/group_mmr2_eeg.npy') 
mmr1 = np.load(root_path + 'cbsA_meeg_analysis/EEG/group_mmr1_mba_eeg.npy')
mmr2 = np.load(root_path + 'cbsA_meeg_analysis/EEG/group_mmr2_pa_eeg.npy')

X = mmr2-mmr1
X = X[:,ts:te].mean(axis=1)
stats.ttest_1samp(X,0)

#%% non-paramatric permutation test on EEG
root_path='/media/tzcheng/storage/CBS/'
times = np.linspace(-0.1,0.6,3501) # For MMR
times = np.linspace(-0.02,0.2,1101) # For FFR
ts = 500
te = 1750

std = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_std_eeg.npy')
dev1 = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_dev1_eeg.npy')
dev2 = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_dev2_eeg.npy')
std = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_std_cabr_eeg_200.npy')
dev1 = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_dev1_cabr_eeg_200.npy')
dev2 = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_dev2_cabr_eeg_200.npy')

MMR1 = dev1 - std
MMR2 = dev2 - std

X = MMR1-MMR2
X = X[:,ts:te]

## FFR 
X = dev2 - dev1
X = dev2 - std

T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(X)

good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
for i in np.arange(0,len(good_cluster_inds),1):
    print("The " + str(i+1) + "st significant cluster")
    print(times[clusters[good_cluster_inds[i]]])
times[ts:te][clusters[good_cluster_inds]]

#%% non-paramatric permutation test on the whole brain
tic = time.time()
stc1 = mne.read_source_estimate(root_path + 'cbs_A101/sss_fif/cbs_A101_mmr2_morph-vl.stc')
src=mne.read_source_spaces('/media/tzcheng/storage2/subjects/fsaverage/bem/fsaverage-vol-5-src.fif')

times = stc1.times

mmr1 = np.load(root_path + 'cbsA_meeg_analysis/MEG/vector_method/group_mmr1_vector_morph.npy')
mmr2 = np.load(root_path + 'cbsA_meeg_analysis/MEG/vector_method/group_mmr2_vector_morph.npy')

X = mmr1-mmr2
Xt = np.transpose(X,[0,2,1])

print('Computing adjecency')
adjecency = spatial_src_adjacency(src)

#    Now let's actually do the clustering. This can take a long time...
#    Here we set the threshold quite high to reduce computation.
# p_threshold = 0.01
# t_threshold=-stats.distributions.t.ppf(p_threshold/2.,35-1)
print('Clustering.')

# T_obs, clusters, cluster_p_values, H0 = clu =\
#     spatio_temporal_cluster_1samp_test(Xt, adjacency=adjecency, n_jobs=4,threshold=None, buffer_size=None,n_permutations=512)

## will kill the kernal
T_obs, clusters, cluster_p_values, H0 = clu =\
    spatio_temporal_cluster_1samp_test(Xt, adjacency=adjecency, n_jobs=4,threshold=dict(start=0,step=0.5), buffer_size=None,n_permutations=512)
#    Now select the clusters that are sig. at p < 0.05 (note that this value
#    is multiple-comparisons corrected).
good_cluster_inds = np.where(cluster_p_values < 0.05)[0]

toc = time.time()
print('it takes ' +str((toc-tic)/60) +'mins')
np.save('/media/tzcheng/storage/CBS/cbsA_meeg_analysis/tfce_p_values',cluster_p_values)
np.save('/media/tzcheng/storage/CBS/cbsA_meeg_analysis/tfce_t',T_obs)
np.save('/media/tzcheng/storage/CBS/cbsA_meeg_analysis/tfce_h0',H0)

#%% visualize clusters
stc=mne.read_source_estimate('/CBS/cbs_A101/sss_fif/cbs_A101_mmr1-vl.stc')
cluster_p_values=-np.log10(cluster_p_values)
stc.data=cluster_p_values.reshape(3501,14629).T

#lims=[0, 0.025, 0.05]
lims=[1.5, 2, 2.5]
kwargs=dict(src=src, subject='fsaverage',subjects_dir=subjects_dir)
#stc_to.data=1-cluster_p_values.reshape(701,14629).T
brain=stc.plot_3d(clim=dict(kind='value',pos_lims=lims),hemi='both',views=['axial'],size=(600,300),view_layout='horizontal',show_traces=0.5,**kwargs)

#%% ROIs
#label info
atlas = 'aparc' # aparc, aparc.a2009s
fname_aseg = '/mnt/subjects/fsaverage/mri/'+atlas+'+aseg.mgz'
label_names = mne.get_volume_labels_from_aseg(fname_aseg)

l_stg=label_names.index('ctx-lh-superiortemporal')
r_stg=label_names.index('ctx-rh-superiortemporal')
l_ifg=[label_names.index('ctx-lh-parsopercularis'),label_names.index('ctx-lh-parsorbitalis'),label_names.index('ctx-lh-parstriangularis')]
r_ifg=[label_names.index('ctx-rh-parsopercularis'),label_names.index('ctx-rh-parsorbitalis'),label_names.index('ctx-rh-parstriangularis')]

mmr1_roi=np.load('/mnt/CBS/meg_mmr_analysis/group_mmr1_roi.npy')
mmr2_roi=np.load('/mnt/CBS/meg_mmr_analysis/group_mmr2_roi.npy')

#%% plot ROI by contrast
plt.figure(figsize=(12,12))
plt.subplot(2,2,1)
plot_err(mmr1_roi[:,l_stg,:],'b')
plot_err(mmr2_roi[:,l_stg,:],'r')
plt.xlabel('Time(ms)',fontsize=18)
plt.ylabel('dSPM Value',fontsize=18)
plt.ylim((-0,10))
#plt.legend(['Nonnative','Native'],prop={'size':20},loc='lower right')
plt.title('LEFT Temporal',fontsize=20)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)


plt.subplot(2,2,3)
plot_err(np.mean(mmr1_roi[:,l_ifg,:],axis=1),'b')
plot_err(np.mean(mmr2_roi[:,l_ifg,:],axis=1),'r')
plt.xlabel('Time(ms)',fontsize=18)
plt.ylabel('dSPM Value',fontsize=18)
plt.ylim((-0,10))
#plt.legend(['Nonnative','Native'],prop={'size':20},loc='lower right')
plt.title('LEFT Frontal',fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.subplot(2,2,2)
plot_err(mmr1_roi[:,r_stg,:],'b')
plot_err(mmr2_roi[:,r_stg,:],'r')
plt.xlabel('Time(ms)',fontsize=18)
plt.ylabel('dSPM Value',fontsize=18)
plt.ylim((-0,10))
#plt.legend(['Nonnative','Native'],prop={'size':20},loc='lower right')
plt.title('RIGHT Temporal',fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.subplot(2,2,4)
plot_err(np.mean(mmr1_roi[:,r_ifg,:],axis=1),'b')
plot_err(np.mean(mmr2_roi[:,r_ifg,:],axis=1),'r')
plt.xlabel('Time(ms)',fontsize=18)
plt.ylabel('dSPM Value',fontsize=18)
plt.ylim((-0,10))
#plt.legend(['Nonnative','Native'],prop={'size':20},loc='lower right')
plt.title('RIGHT Frontal',fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

#%% bootstraping the EEG xcorr between FFR and audio
root_path='/media/tzcheng/storage/CBS/'

## Load FFR from 0
std = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_std_cabr_eeg_200.npy')[:,100:]
dev1 = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_dev1_cabr_eeg_200.npy')[:,100:]
dev2 = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_dev2_cabr_eeg_200.npy')[:,100:]

## Load real audio
fs, std_audio = wavfile.read('/media/tzcheng/storage/CBS/stimuli/+10.wav')
fs, dev1_audio = wavfile.read('/media/tzcheng/storage/CBS/stimuli/-40.wav')
fs, dev2_audio = wavfile.read('/media/tzcheng/storage/CBS/stimuli/+40.wav')
# Downsample
fs_new = 5000
num_std = int((len(std_audio)*fs_new)/fs)
num_dev = int((len(dev2_audio)*fs_new)/fs)  # #sample_new/fs_new=#sample/fs find number of samples in the resampled data
            
std_audio = signal.resample(std_audio, num_std, t=None, axis=0, window=None)
dev1_audio = signal.resample(dev1_audio, num_dev, t=None, axis=0, window=None)
dev2_audio = signal.resample(dev2_audio, num_dev, t=None, axis=0, window=None)

## choose among std, dev1, dev2 and run the corresponding session below
audio0 = dev2_audio
EEG0 = dev2
times0_audio = np.linspace(0,len(audio0)/5000,len(audio0))
times0_eeg = np.linspace(0,len(EEG0[0])/5000,len(EEG0[0]))

## dev 1: noise burst from 40 ms (200th points)
ts = 200 + 100
te = 650
audio = audio0[ts:te] # try 0.042 to 0.13 s for dev2
EEG = EEG0[:,ts:te]
times_audio = times0_audio[ts:te]
times_eeg = times0_eeg[ts:te]
lags = signal.correlation_lags(len(audio),len(EEG[0]))
lags_s = lags/5000

## dev 2: noise burst from 0 ms (100th points)
ts = 100
te = 650
audio = audio0[ts:te] # try 0.02 to 0.13 s for dev2
EEG = EEG0[:,ts:te]
times_audio = times0_audio[ts:te]
times_eeg = times0_eeg[ts:te]
lags = signal.correlation_lags(len(audio),len(EEG[0]))
lags_s = lags/5000

## std: noise burst from 0 ms (100th points)
ts = 100
te = 500
audio = audio0[ts:te] # try 0.02 to 0.1 s for std
EEG = EEG0[:,ts:te]
times_audio = times0_audio[ts:te]
times_eeg = times0_eeg[ts:te]
lags = signal.correlation_lags(len(audio),len(EEG[0]))
lags_s = lags/5000

## xcorr from averaged FFR
a = (audio - np.mean(audio))/np.std(audio)
a = a / np.linalg.norm(a)
b = (EEG.mean(axis=0) - np.mean(EEG.mean(axis=0)))/np.std(EEG.mean(axis=0))
b = b / np.linalg.norm(b)
xcorr = signal.correlate(a,b,mode='full')
xcorr = abs(xcorr)
xcorr_max = max(xcorr)
xcorr_maxlag = np.argmax(xcorr)

## sample with replacement 
n_boot = 1000
boot_xcorr_max = []
boot_xcorr_maxlag = []
ind = np.arange(0,np.shape(EEG)[0],1)

for n in np.arange(0,n_boot,1):
    boot_ind = sklearn.utils.resample(ind)
    boot_EEG = EEG[boot_ind,:]
    b = (boot_EEG.mean(axis=0) - np.mean(boot_EEG.mean(axis=0)))/np.std(boot_EEG.mean(axis=0))
    b = b / np.linalg.norm(b)
    xcorr = signal.correlate(a,b,mode='full')
    xcorr = abs(xcorr)
    xcorr_max = max(xcorr)
    xcorr_maxlag = np.argmax(xcorr)
    boot_xcorr_max.append(xcorr_max)
    boot_xcorr_maxlag.append(xcorr_maxlag)
boot_xcorr_maxlag = lags_s[boot_xcorr_maxlag]*1000
boot_xcorr_max=np.asarray(boot_xcorr_max)
boot_xcorr_maxlag=np.asarray(boot_xcorr_maxlag)

plt.figure()
plt.hist(boot_xcorr_max,bins=30,color='k')
plt.vlines(xcorr_max,ymin=0,ymax=12,color='r',linewidth=2)
plt.vlines(np.percentile(boot_xcorr_max,97.5),ymin=0,ymax=12,color='grey',linewidth=2)
plt.ylabel('Count',fontsize=20)
plt.xlabel('Xcorr coef',fontsize=20)
plt.title('Xcorr coef compared to 97.5 percentile of n = 1000 bootstrap distribution')
