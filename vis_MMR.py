#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 11:31:16 2023

Used to visualize MMR subplots 

@author: tzcheng
"""

import mne
import matplotlib
import matplotlib.pyplot as plt 
import numpy as np
import os
from nilearn.plotting import plot_glass_brain
from scipy import stats,signal

def plot_err(group_stc,color,t):
    group_avg=np.mean(group_stc,axis=0)
   #plt.figure()
    err=np.std(group_stc,axis=0)/np.sqrt(group_stc.shape[0])
    up=group_avg+err
    lw=group_avg-err
    plt.plot(t,group_avg,color=color)
    plt.fill_between(t,up,lw,color=color,alpha=0.5)

root_path='/media/tzcheng/storage2/CBS/'
subjects_dir = '/media/tzcheng/storage2/subjects/'
stc1 = mne.read_source_estimate(root_path + 'cbs_A101/sss_fif/cbs_A101_mmr2_morph-vl.stc')

#%%####################################### Traditional or new direction
## Load vertex: traditional method 
# adults 
MEG_mmr1_v = np.load(root_path + 'cbsA_meeg_analysis/MEG/vector_method/group_mmr1_vector_morph.npy') # with the mag or vector method
MEG_mmr2_v = np.load(root_path + 'cbsA_meeg_analysis/MEG/vector_method/group_mmr2_vector_morph.npy') # with the mag or vector method
EEG_mmr1 = np.load(root_path + 'cbsA_meeg_analysis/EEG/group_mmr1_eeg.npy')
EEG_mmr2 = np.load(root_path + 'cbsA_meeg_analysis/EEG/group_mmr2_eeg.npy')

# infants
MEG_mmr1_v = np.load(root_path + 'cbsb_meg_analysis/MEG/vector_method/group_mmr1_vector_morph.npy') # with the mag or vector method
MEG_mmr2_v = np.load(root_path + 'cbsb_meg_analysis/MEG/vector_method/group_mmr2_vector_morph.npy') # with the mag or vector method

## Load vertex: new method
MEG_mmr1_v = np.load(root_path + 'cbsA_meeg_analysis/MEG/vector_method/group_mmr1_mba_vector_morph.npy') # with the mag or vector method
MEG_mmr2_v = np.load(root_path + 'cbsA_meeg_analysis/MEG/vector_method/group_mmr2_pa_vector_morph.npy') # with the mag or vector method
EEG_mmr1 = np.load(root_path + 'cbsA_meeg_analysis/EEG/group_mmr1_mba_eeg.npy')
EEG_mmr2 = np.load(root_path + 'cbsA_meeg_analysis/EEG/group_mmr2_pa_eeg.npy')

subject = 'fsaverage'
src = mne.read_source_spaces(subjects_dir + subject + '/bem/fsaverage-vol-5-src.fif')
fname_aseg = subjects_dir + subject + '/mri/aparc+aseg.mgz'
label_names = np.asarray(mne.get_volume_labels_from_aseg(fname_aseg))
label_names
lh_ROI_label = [60,61,62,72] # STG and IFG (parsopercularis, parsorbitalis, parstriangularis)
rh_ROI_label = [96,97,98,108] # STG and IFG (parsopercularis, parsorbitalis, parstriangularis)

## visualization average same-plot
plt.figure()
plot_err(stats.zscore(EEG_mmr1,axis=1),'k',stc1.times)
plot_err(stats.zscore(MEG_mmr1_m.mean(axis=1),axis=1),'b',stc1.times)
plot_err(stats.zscore(MEG_mmr1_v.mean(axis=1),axis=1),'r',stc1.times)
plot_err(stats.zscore(MEG_mmr1_c.mean(axis=1),axis=1),'orange',stc1.times)

plt.legend(['EEG','','MEG vector',''])
plt.xlabel('Time (s)')
plt.ylabel('zscore')
plt.title('Traditional method MMR1')
plt.xlim([-0.05, 0.45])

plt.figure()
plot_err(stats.zscore(EEG_mmr2,axis=1),'k',stc1.times)
plot_err(stats.zscore(MEG_mmr2_m.mean(axis=1),axis=1),'b',stc1.times)
plot_err(stats.zscore(MEG_mmr2_v.mean(axis=1),axis=1),'r',stc1.times)
plot_err(stats.zscore(MEG_mmr2_c.mean(axis=1),axis=1),'orange',stc1.times)
plt.legend(['EEG','','MEG mag','','MEG vector',''])
plt.xlabel('Time (ms)')
plt.ylabel('zscore')
plt.title('Traditional method MMR2')
plt.xlim([-0.05, 0.45])

## visualization average sub-plot
times = stc1.times
plt.figure()
plt.subplot(311)
plot_err(EEG_mmr1,'grey',stc1.times)
plot_err(EEG_mmr2,'k',stc1.times)
plt.title('Traditional direction')
plt.xlim([-100,600])

plt.subplot(312)
plot_err(MEG_mmr1_m.mean(axis=1),'c',stc1.times)
plot_err(MEG_mmr2_m.mean(axis=1),'b',stc1.times)
plt.xlim([-100,600])
plt.ylabel('Amplitude')

plt.subplot(313)
plot_err(MEG_mmr1_v.mean(axis=1),'m',stc1.times)
plot_err(MEG_mmr2_v.mean(axis=1),'r',stc1.times)
plt.xlim([-100,600])
plt.xlabel('Time (ms)')

## visualization ROI
plt.figure()
plt.subplot(311)
plot_err(EEG_mmr1,'grey',stc1.times)
plot_err(EEG_mmr2,'k',stc1.times)
plt.title('left ROI: First last mba and pa')
plt.xlim([-100,600])

plt.subplot(312)
plot_err(MEG_mmr1_roi_m[:,lh_ROI_label,:].mean(axis=1),'c',stc1.times)
plot_err(MEG_mmr2_roi_m[:,lh_ROI_label,:].mean(axis=1),'b',stc1.times)
plt.xlim([-100,600])
plt.ylabel('Amplitude')

plt.subplot(313)
plot_err(MEG_mmr1_roi_v[:,lh_ROI_label,:].mean(axis=1),'m',stc1.times)
plot_err(MEG_mmr2_roi_v[:,lh_ROI_label,:].mean(axis=1),'r',stc1.times)
plt.xlim([-100,600])
plt.xlabel('Time (ms)')

# time series of EEG, MEG_v, MEG_m averaged across all sources for each individual
fig,axs = plt.subplots(1,len(MEG_mmr1_m),sharex=True,sharey=True)
fig.suptitle('MMR2')

for s in np.arange(0,len(MEG_mmr1_m),1):
    axs[s].plot(times,stats.zscore(EEG_mmr2[s,:],axis=0),'k')
    axs[s].plot(times,stats.zscore(MEG_mmr2_m[s,:,:].mean(axis =0),axis=0),'b')
    axs[s].plot(times,stats.zscore(MEG_mmr2_v[s,:,:].mean(axis =0),axis=0),'r')
    axs[s].set_xlim([0.05,0.3])
    axs[s].set_title('subj' + str(s+1))
    
#%%####################################### check the evoked for fist and last /ba/, /pa/ and /mba/
## Note that the result could be slightly different because /ba/ was random sampled
root_path='/media/tzcheng/storage/CBS/'
stc1 = mne.read_source_estimate(root_path + 'cbs_A101/sss_fif/cbs_A101_mmr2_morph-vl.stc')
times = stc1.times

EEG_1pa = np.load(root_path + 'cbsA_meeg_analysis/EEG/group_dev2_eeg.npy')
EEG_1mba = np.load(root_path + 'cbsA_meeg_analysis/EEG/group_dev1_eeg.npy')
EEG_1ba = np.load(root_path + 'cbsA_meeg_analysis/EEG/group_dev_reverse_eeg.npy')

EEG_endpa = np.load(root_path + 'cbsA_meeg_analysis/EEG/group_std2_reverse_eeg.npy')
EEG_endmba = np.load(root_path + 'cbsA_meeg_analysis/EEG/group_stdMEG_mmr1_v = np.load(root_path + 'cbsA_meeg_analysis/MEG/vector_method/group_mmr1_vector_morph.npy')
MEG_mmr2_v = np.load(root_path + 'cbsA_meeg_analysis/MEG/vector_method/group_mmr2_vector_morph.npy')
1_reverse_eeg.npy')
EEG_endba = np.load(root_path + 'cbsA_meeg_analysis/EEG/group_std_eeg.npy')

plt.figure()
plot_err(EEG_1ba,'r',stc1.times)
plot_err(EEG_1mba,'g',stc1.times)
plot_err(EEG_1pa,'b',stc1.times)
plt.title('EEG Evoked response for ba, mba and pa')
plt.legend(['1st ba','','1st mba','','1st pa',''])
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.xlim([-100,600])
scores_observed = np.load('/media/tzcheng/storage/CBS/cbsA_meeg_analysis/decoding/roc_auc_None_morph_kall.npy')
patterns = np.load('/media/tzcheng/storage/CBS/cbsA_meeg_analysis/decoding/patterns_None_morph_kall.npy')
plt.figure()
plot_err(EEG_endba,'r',stc1.times)
plot_err(EEG_endmba,'g',stc1.times)
plot_err(EEG_endpa,'b',stc1.times)
plt.title('EEG Evoked response for ba, mba and pa')
plt.legend(['last ba','','last mba','','last pa',''])
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.xlim([-100,600])

plt.figure()
plot_err(EEG_endba,'r',stc1.times)
plot_err(EEG_1pa,'b',stc1.times)
plot_err(EEG_1pa-EEG_endba,'k',stc1.times)
plt.legend(['std','','dev','','MMR',''])
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.xlim([-100,600])

plt.figure()
plot_err(EEG_1pa,'b',stc1.times)
plot_err(EEG_endpa,'r',stc1.times)
plt.legend(['1st pa','','last pa',''])
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.xlim([-100,600])

plt.figure()
plot_err(EEG_1mba - EEG_endmba,'b',stc1.times)
plot_err(EEG_1pa - EEG_endpa,'r',stc1.times)
plt.legend(['mmr mba','','mmr pa',''])
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.xlim([-100,600])

#%%####################################### check the source activity for first and last /ba/, /pa/ and /mba/
## Note that the result could be slightly different because /ba/ was random sampled
root_path='/media/tzcheng/storage/CBS/'
stc1 = mne.read_source_estimate(root_path + 'cbs_A101/sss_fif/cbs_A101_mmr2_morph-vl.stc')
times = stc1.times

MEG_mmr1_m = np.load(root_path + 'cbsA_meeg_analysis/MEG/magnitude_method/group_mmr1_mba_None_morph.npy')
MEG_mmr2_m = np.load(root_path + 'cbsA_meeg_analysis/MEG/magnitude_method/group_mmr2_pa_None_morph.npy')

MEG_mmr1_v = np.load(root_path + 'cbsA_meeg_analysis/MEG/vector_method/group_mmr1_mba_vector_morph.npy')
MEG_mmr2_v = np.load(root_path + 'cbsA_meeg_analysis/MEG/vector_method/group_mmr2_pa_vector_morph.npy')

plt.figure()
plot_err(MEG_mmr1_m.mean(axis = 1),'b',stc1.times)
plot_err(MEG_mmr2_m.mean(axis = 1),'r',stc1.times)
plt.legend(['mmr mba','','mmr pa',''])
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.title('Mag method')
plt.xlim([-100,600])

plt.figure()
plot_err(MEG_mmr1_v.mean(axis = 1),'b',stc1.times)
plot_err(MEG_mmr2_v.mean(axis = 1),'r',stc1.times)
plt.legend(['mmr mba','','mmr pa',''])
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.title('vector method')
plt.xlim([-100,600])
scores_observed = np.load('/media/tzcheng/storage/CBS/cbsA_meeg_analysis/decoding/roc_auc_None_morph_kall.npy')
patterns = np.load('/media/tzcheng/storage/CBS/cbsA_meeg_analysis/decoding/patterns_None_morph_kall.npy')

## visualization average
times = stc1.times
plt.figure()
plt.subplot(311)
plot_err(EEG_1mba - EEG_endmba,'grey',stc1.times)
plot_err(EEG_1pa - EEG_endpa,'k',stc1.times)
plt.title('First last mba and pa')
plt.xlim([-100,600])

plt.subplot(312)
plot_err(MEG_mmr1_m.mean(axis=1),'c',stc1.times)
plot_err(MEG_mmr2_m.mean(axis=1),'b',stc1.times)
plt.xlim([-100,600])
plt.ylabel('Amplitude')

plt.subplot(313)
plot_err(MEG_mmr1_v.mean(axis=1),'m',stc1.times)
plot_err(MEG_mmr2_v.mean(axis=1),'r',stc1.times)
plt.xlim([-100,600])
plt.xlabel('Time (ms)')

#%%####################################### check the std, dev, mmr for first and last /ba/, /pa/ and /mba/
MEG_mmr1_m = np.load(root_path + 'cbsA_meeg_analysis/MEG/magnitude_method/group_mmr1_mba_None_morph.npy') # with the mag or vector method
MEG_mmr2_m = np.load(root_path + 'cbsA_meeg_analysis/MEG/magnitude_method/group_mmr2_pa_None_morph.npy') # with the mag or vector method
MEG_std1_m = np.load(root_path + 'cbsA_meeg_analysis/MEG/magnitude_method/group_std1_reverse_None_morph.npy') # with the mag or vector method
MEG_std2_m = np.load(root_path + 'cbsA_meeg_analysis/MEG/magnitude_method/group_std2_reverse_None_morph.npy') # with the mag or vector method
MEG_dev1_m = np.load(root_path + 'cbsA_meeg_analysis/MEG/magnitude_method/group_dev1_None_morph.npy') # with the mag or vector method
MEG_dev2_m = np.load(root_path + 'cbsA_meeg_analysis/MEG/magnitude_method/group_dev2_None_morph.npy') # with the mag or vector method

plt.figure()
plot_err(MEG_std1_m.mean(axis = 1),'r',stc1.times)
plot_err(MEG_dev1_m.mean(axis = 1),'b',stc1.times)
plot_err(MEG_dev1_m.mean(axis = 1) - MEG_std1_m.mean(axis = 1),'k',stc1.times)
plt.legend(['std','','dev','','MMR1',''])
plt.xlabel('Time (ms)')and the 
plt.ylabel('Amplitude')
plt.xlim([-100,600])

plt.figure()
plot_err(MEG_std2_m.mean(axis = 1),'r',stc1.times)
plot_err(MEG_dev2_m.mean(axis = 1),'b',stc1.times)
plot_err(MEG_dev2_m.mean(axis = 1) - MEG_std2_m.mean(axis = 1),'k',stc1.times)
plt.legend(['std','','dev','','MMR2',''])
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.xlim([-100,600])

#%%####################################### MMR result whole brain
## Traditional direction: ba to mba vs. ba to pa
## Adults
MEG_mmr1_m = np.load(root_path + 'cbsA_meeg_analysis/MEG/magnitude_method/group_mmr1_None_morph.npy')
MEG_mmr2_m = np.load(root_path + 'cbsA_meeg_analysis/MEG/magnitude_method/group_mmr2_None_morph.npy')
MEG_mmr1_v = np.load(root_path + 'cbsA_meeg_analysis/MEG/vector_method/group_mmr1_vector_morph.npy')
MEG_mmr2_v = np.load(root_path + 'cbsA_meeg_analysis/MEG/vector_method/group_mmr2_vector_morph.npy')
## Babies
MEG_mmr1_m = np.load(root_path + 'cbsb_meg_analysis/group_mmr1_None_morph.npy')
MEG_mmr2_m = np.load(root_path + 'cbsb_meg_analysis/group_mmr2_None_morph.npy')
MEG_mmr1_v = np.load(root_path + 'cbsb_meg_analysis/group_mmr1_vector_morph.npy')
MEG_mmr2_v = np.load(root_path + 'cbsb_meg_analysis/group_mmr2_vector_morph.npy')

## New method: first - last mba vs. first pa - last pascores_observed = np.load(root_path + '/cbsA_meeg_analysis/decoding/adult_roc_auc_vector_morph_kall.npy')
MEG_mmr1_m = np.load(root_path + 'cbsA_meeg_analysis/MEG/magnitude_method/group_mmr1_mba_None_morph.npy')
MEG_mmr2_m = np.load(root_path + 'cbsA_meeg_analysis/MEG/magnitude_method/group_mmr2_pa_None_morph.npy')
MEG_mmr1_v = np.load(root_path + 'cbsA_meeg_analysis/MEG/vector_method/group_mmr1_mba_vector_morph.npy')
MEG_mmr2_v = np.load(root_path + 'cbsA_meeg_analysis/MEG/vector_method/group_mmr2_pa_vector_morph.npy')

subjects_dir = '/media/tzcheng/storage2/subjects/'

stc1 = mne.read_source_estimate(root_path + 'cbs_A101/sss_fif/cbs_A101_mmr2_morph-vl.stc')
times = stc1.times
src = mne.read_source_spaces(subjects_dir + 'fsaverage/bem/fsaverage-vol-5-src.fif')

stc1.data = MEG_mmr2_v.mean(axis=0) - MEG_mmr1_v.mean(axis=0)
stc1_crop = stc1.copy().crop(tmin= -0.05, tmax=0.45)
stc1_crop.plot(src, clim=dict(kind="percent",pos_lims=[90,95,99]), subject='fsaverage', subjects_dir=subjects_dir)


stc1.plot(src, clim=dict(kind="percent",pos_lims=[90,95,99]), subject='fsaverage', subjects_dir=subjects_dir)
stc1.plot(src,clim=dict(kind="value",pos_lims=[0,12,16]), subject='fsaverage', subjects_dir=subjects_dir)

#%%####################################### decoding result
## Traditional direction: ba to mba vs. ba to pa
## Adults
scores_observed = np.load(root_path + '/cbsA_meeg_analysis/decoding/adult_roc_auc_vector_morph_kall.npy')
patterns = np.load(root_path + 'cbsA_meeg_analysis/decoding/adult_patterns_vector_morph_kall.npy')
scores_permute = np.load(root_path + '/cbsA_meeg_analysis/decoding/adult_vector_scores_100perm_kall_tradition.npz')

## Babies
scores_observed = np.load(root_path + 'cbsb_meg_analysis/decoding/baby_roc_auc_vector_morph_kall.npy')
patterns = np.load(root_path + 'cbsb_meg_analysis/decoding/baby_patterns_vector_morph_kall.npy')
scores_permute = np.load(root_path + '/cbsb_meg_analysis/decoding/baby_vector_scores_100perm_kall_tradition.npz')

## New method: first - last mba vs. first pa - last pa
scores_observed = np.load(root_path + 'cbsA_meeg_analysis/decoding/adult_roc_auc_vector_morph_kall_mba_pa.npy')
patterns = np.load(root_path + 'cbsA_meeg_analysis/decoding/adult_patterns_vector_morph_kall_mba_pa.npy')
scores_permute = np.load(root_path + '/cbsA_meeg_analysis/decoding/adult_vector_scores_100perm_kall_new.npz')

scores_observed = np.load(root_path + 'cbsb_meg_analysis/decoding/baby_roc_auc_vector_morph_kall_mba_pa.npy')
patterns = np.load(root_path + 'cbsb_meg_analysis/decoding/baby_patterns_vector_morph_kall_mba_pa.npy')
scores_permute = np.load(root_path + '/cbsb_meg_analysis/decoding/baby_vector_scores_100perm_kall_new.npz')


## Plot acc across time
fig, ax = plt.subplots(1)
ax.plot(stc1.times, scores_observed.mean(0), label="score")
ax.plot(scores_permute['peaks_time'],np.percentile(scores_permute['scores_perm_array'],95,axis=0),'g.')
ax.axhline(0.5, color="k", linestyle="--", label="chance")
ax.axhline(np.percentile(scores_observed.mean(0),q = 97.5), color="grey", linestyle="--", label="95 percentile")
ax.axvline(0, color="k")
plt.xlabel('Time (s)')
plt.title('Decoding accuracy baby new')
plt.xlim([-0.05,0.45])
plt.ylim([0,1])

## Plot patterns
src = mne.read_source_spaces(subjects_dir + 'fsaverage/bem/fsaverage-vol-5-src.fif')
stc1 = mne.read_source_estimate(root_path + 'cbs_A101/sss_fif/cbs_A101_mmr2_morph-vl.stc')
stc1.data = patterns
stc1_crop = stc1.copy().crop(tmin= -0.05, tmax=0.45)
# Plot patterns across sources
stc1_crop.plot(src, clim=dict(kind="percent",pos_lims=[90,95,99]), subject='fsaverage', subjects_dir=subjects_dir)

stc1.plot(src, clim=dict(kind="percent",pos_lims=[90,95,99]), subject='fsaverage', subjects_dir=subjects_dir)

stc1.plot(src, clim=dict(kind="value",pos_lims=[0,4,8], subject='fsaverage', subjects_dir=subjects_dir)
stc1_crop.plot(src, subject='fsaverage', subjects_dir=subjects_dir)
stc1_crop.plot(src,c, subject='fsaverage', subjects_dir=subjects_dir)

