import mne
import matplotlib.pyplot as plt 
import numpy as np
import os
from scipy import stats,signal
from numpy import dot
from numpy.linalg import norm
from scipy.stats import pearsonr
import scipy as sp
import os
import seaborn as sns
import pandas as pd
import scipy.stats as stats
from scipy.io import wavfile
import time


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
stc1 = mne.read_source_estimate(root_path + 'cbs_A101/sss_fif/cbs_A101_pa_cabr_morph-vl.stc')
src = mne.read_source_spaces(subjects_dir + 'fsaverage/bem/fsaverage-vol-5-src.fif')

#%%####################################### Load audio and cABR
#%%####################################### Load audio and cABR
## audio 
fs, ba_audio = wavfile.read('/media/tzcheng/storage2/CBS/stimuli/+10.wav')
fs, mba_audio = wavfile.read('/media/tzcheng/storage2/CBS/stimuli/-40.wav')
fs, pa_audio = wavfile.read('/media/tzcheng/storage2/CBS/stimuli/+40.wav')

# Downsample
fs_new = 5000
num_std = int((len(ba_audio)*fs_new)/fs)
num_dev = int((len(pa_audio)*fs_new)/fs)  # #sample_new/fs_new=#sample/fs find number of samples in the resampled data
audio_ba = signal.resample(ba_audio, num_std, t=None, axis=0, window=None)
audio_mba = signal.resample(mba_audio, num_dev, t=None, axis=0, window=None)
audio_pa = signal.resample(pa_audio, num_dev, t=None, axis=0, window=None)

## EEG
EEG_ba_cABR = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_std_cabr_eeg_200.npy')
EEG_mba_cABR = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_dev1_cabr_eeg_200.npy')
EEG_pa_cABR = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_dev2_cabr_eeg_200.npy')

## MEG source vertices: more than 200 trials for now
# adults
MEG_ba_cABR = np.load(root_path + 'cbsA_meeg_analysis/MEG/cABR/' + 'group_ba_cabr_morph.npy')
MEG_mba_cABR = np.load(root_path + 'cbsA_meeg_analysis/MEG/cABR/' + 'group_mba_cabr_morph.npy')
MEG_pa_cABR = np.load(root_path + 'cbsA_meeg_analysis/MEG/cABR/' + 'group_pa_cabr_morph.npy')

# infants 
MEG_ba_cABR = np.load(root_path + 'cbsb_meg_analysis/MEG/cABR/' + 'group_ba_cabr_morph.npy')
MEG_mba_cABR = np.load(root_path + 'cbsb_meg_analysis/MEG/cABR/' + 'group_mba_cabr_morph.npy')
MEG_pa_cABR = np.load(root_path + 'cbsb_meg_analysis/MEG/cABR/' + 'group_pa_cabr_morph.npy')

## MEG source ROI: more than 200 trials for now
# adults
MEG_ba_cABR = np.load(root_path + 'cbsA_meeg_analysis/MEG/cABR/' + 'group_ba_cabr_f80450_morph_roi.npy')
MEG_mba_cABR = np.load(root_path + 'cbsA_meeg_analysis/MEG/cABR/' + 'group_mba_cabr_f80450_morph_roi.npy')
MEG_pa_cABR = np.load(root_path + 'cbsA_meeg_analysis/MEG/cABR/' + 'group_pa_cabr_f80450_morph_roi.npy')

## MEG sensor: more than 200 trials for now
# adults
MEG_ba_cABR = np.load(root_path + 'cbsA_meeg_analysis/MEG/cABR/' + 'group_ba_sensor.npy')
MEG_mba_cABR = np.load(root_path + 'cbsA_meeg_analysis/MEG/cABR/' + 'group_mba_sensor.npy')
MEG_pa_cABR = np.load(root_path + 'cbsA_meeg_analysis/MEG/cABR/' + 'group_pa_sensor.npy')

#%%####################################### visualize cABR of ba, mba and pa
plt.figure()
plt.subplot(311)
plot_err(EEG_ba_cABR,'k',stc1.times)
plt.xlim([-0.02,0.2])

plt.subplot(312)
plot_err(EEG_mba_cABR,'b',stc1.times)
plt.xlim([-0.02,0.2])
plt.ylabel('Amplitude')

plt.subplot(313)
plot_err(EEG_pa_cABR,'r',stc1.times)
plt.xlim([-0.02,0.2])
plt.xlabel('Time (s)')

## visualize cABR of ba, mba and pa in MEG sensor
plt.figure()
plt.subplot(311)
plot_err(MEG_ba_cABR.mean(axis=1),'k',stc1.times)
plt.xlim([-0.02,0.2])

plt.subplot(312)
plot_err(MEG_mba_cABR.mean(axis=1),'b',stc1.times)
plt.xlim([-0.02,0.2])
plt.ylabel('Amplitude')

plt.subplot(313)
plot_err(MEG_pa_cABR.mean(axis=1),'r',stc1.times)
plt.xlim([-0.02,0.2])
plt.xlabel('Time (s)') 

## visualize cABR of ba, mba and pa in MEG source 
plt.figure()
plt.subplot(311)
plot_err(MEG_ba_cABR.mean(axis=1),'k',stc1.times)
plt.xlim([-0.02,0.2])

plt.subplot(312)
plot_err(MEG_mba_cABR.mean(axis=1),'b',stc1.times)
plt.xlim([-0.02,0.2])
plt.ylabel('Amplitude')

plt.subplot(313)
plot_err(MEG_pa_cABR.mean(axis=1),'r',stc1.times)
plt.xlim([-0.02,0.2])
plt.xlabel('Time (s)')

## visualize sensor activity based on the cross-correlation
df_MEGEEG = pd.read_pickle(root_path + 'cbsA_meeg_analysis/correlation/cabr_df_xcorr_MEGEEG_sensor_s.pkl')
df_MEGaudio = pd.read_pickle(root_path + 'cbsA_meeg_analysis/correlation/cabr_df_xcorr_MEGaudio_sensor_s.pkl')

max_xcorr_ba = df_MEGEEG.groupby('Subject')['abs XCorr MEG & ba'].max()
tmp_idx = df_MEGEEG.groupby('Subject')['abs XCorr MEG & ba'].idxmax()
max_xcorr_sensor_ba = df_MEGEEG['Vertno'][tmp_idx].tolist()
max_xcorr_mba = df_MEGEEG.groupby('Subject')['abs XCorr MEG & mba'].max()
tmp_idx = df_MEGEEG.groupby('Subject')['abs XCorr MEG & mba'].idxmax()
max_xcorr_sensor_mba = df_MEGEEG['Vertno'][tmp_idx].tolist()
max_xcorr_pa = df_MEGEEG.groupby('Subject')['abs XCorr MEG & pa'].max()
tmp_idx = df_MEGEEG.groupby('Subject')['abs XCorr MEG & pa'].idxmax()
max_xcorr_sensor_pa = df_MEGEEG['Vertno'][tmp_idx].tolist()

subj_ba_array = []
subj_mba_array = []
subj_pa_array = []

for i in np.arange(0,18,1):
    subj_ba_array.append(MEG_ba_cABR[i,max_xcorr_sensor_ba[i],:])    
    subj_mba_array.append(MEG_mba_cABR[i,max_xcorr_sensor_mba[i],:])    
    subj_pa_array.append(MEG_pa_cABR[i,max_xcorr_sensor_pa[i],:])    
plt.figure()
plt.subplot(211)
plot_err(EEG_ba_cABR,'k',stc1.times)
plt.title('ba')
plt.xlim([-0.02,0.2])
plt.subplot(212)
plt.plot(stc1.times,np.array(subj_ba_array).mean(0))
plt.xlim([-0.02,0.2])

plt.figure()
plt.subplot(211)
plt.title('mba')
plot_err(EEG_mba_cABR,'k',stc1.times)
plt.xlim([-0.02,0.2])
plt.subplot(212)
plt.plot(stc1.times,np.array(subj_mba_array).mean(0))
plt.xlim([-0.02,0.2])

plt.figure()
plt.subplot(211)
plt.title('pa')
plot_err(EEG_pa_cABR,'k',stc1.times)
plt.xlim([-0.02,0.2])
plt.subplot(212)
plt.plot(stc1.times,np.array(subj_pa_array).mean(0))
plt.xlim([-0.02,0.2])

## visualize MEG source activity
stc1.data = MEG_mba_cABR.mean(axis=0)
stc1.plot(src, clim=dict(kind="percent",pos_lims=[90,95,99]), subject='fsaverage', subjects_dir=subjects_dir)

## visualize source activity based on the cross-correlation
df = pd.read_pickle(root_path + 'cbsA_meeg_analysis/correlation/cabr_df_xcorr_MEGEEG_roi.pkl')
v_hack = pd.concat([df['abs XCorr MEG & audio_ba'],df['abs XCorr MEG & audio_mba'],df['abs XCorr MEG & audio_pa']],axis=1)
stc1.data = v_hack
stc1.plot(src, clim=dict(kind="percent",pos_lims=[90,95,99]), subject='fsaverage', subjects_dir=subjects_dir)

ind = np.where(stc1.vertices[0] == 18493)
plot_err(np.squeeze(MEG_ba_cABR[:,ind[0],:]),'k',np.linspace(-0.02,0.2,1101))
plt.xlim([0,0.2])

## visualize roi
roi_ind = 66
plt.figure()
plt.subplot(311)
plot_err(MEG_ba_cABR[:,roi_ind,:],'k',stc1.times)
plt.xlim([-0.02,0.2])

plt.subplot(312)
plot_err(MEG_mba_cABR[:,roi_ind,:],'b',stc1.times)
plt.xlim([-0.02,0.2])
plt.ylabel('Amplitude')

plt.subplot(313)
plot_err(MEG_pa_cABR[:,roi_ind,:],'r',stc1.times)
plt.xlim([-0.02,0.2])
plt.xlabel('Time (s)') 

## visualize sensor
evoked = mne.read_evokeds(root_path + 'cbs_A123/sss_fif/cbs_A123_01_otp_raw_sss_proj_f_evoked_substd_cabr.fif')[0]
evoked.plot_sensors(kind = '3d')

sensor_ind = np.where(np.array(evoked.ch_names) == 'MEG0731')

plt.figure()
plt.subplot(311)
plot_err(np.squeeze(MEG_ba_cABR[:,sensor_ind,:]),'k',stc1.times)
plt.xlim([-0.02,0.2])

plt.subplot(312)
plot_err(np.squeeze(MEG_mba_cABR[:,sensor_ind,:]),'b',stc1.times)
plt.xlim([-0.02,0.2])
plt.ylabel('Amplitude')

plt.subplot(313)
plot_err(np.squeeze(MEG_pa_cABR[:,sensor_ind,:]),'r',stc1.times)
plt.xlim([-0.02,0.2])
plt.xlabel('Time (s)') 
