#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 11:22:13 2024

Used to visualize FFR subplots 

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

#%%####################################### load data
root_path='/media/tzcheng/storage2/CBS/'
subjects_dir = '/media/tzcheng/storage2/subjects/'
stc1 = mne.read_source_estimate(root_path + 'cbs_A101/sss_fif/cbs_A101_ba_cabr_morph-vl.stc')
times = stc1.times
src = mne.read_source_spaces(subjects_dir + 'fsaverage/bem/fsaverage-vol-5-src.fif')
fname_aseg = subjects_dir + 'fsaverage/mri/aparc+aseg.mgz'
label_names = np.asarray(mne.get_volume_labels_from_aseg(fname_aseg))
## FFR relevant ROIs
lh_ROI_label = [12, 72,76,74] # [subcortical] brainstem,[AC] STG, transversetemporal, [controls] frontal pole
rh_ROI_label = [12, 108,112,110] # [subcortical] brainstem,[AC] STG, transversetemporal, [controls] frontal pole

did_pca = '_pcffr' # without or with pca "_pcffr"
filename_ffr_ba = 'group_ba' + did_pca
filename_ffr_mba = 'group_mba' + did_pca
filename_ffr_pa = 'group_pa' + did_pca

baby_or_adult = 'cbsb_meg_analysis' # baby or adult
input_data = 'wholebrain' # ROI or wholebrain or sensor or pcffr

if input_data == 'sensor':
    ffr_ba = np.load(root_path + baby_or_adult + '/MEG/FFR/' + filename_ffr_ba + '_sensor.npy',allow_pickle=True)
    ffr_mba = np.load(root_path + baby_or_adult +'/MEG/FFR/' + filename_ffr_mba + '_sensor.npy',allow_pickle=True)
    ffr_pa = np.load(root_path + baby_or_adult +'/MEG/FFR/' + filename_ffr_pa + '_sensor.npy',allow_pickle=True)
    FFR_accuracy = np.load(root_path + baby_or_adult + '/decoding/FFR_decoding_accuracy_sensor.npy')
    PCFFR_accuracy = np.load(root_path + baby_or_adult + '/decoding/PCFFR_decoding_accuracy_sensor.npy')
elif input_data == 'ROI':
    ffr_ba = np.load(root_path + baby_or_adult +'/MEG/FFR/' + filename_ffr_ba + '_morph_roi.npy',allow_pickle=True)
    ffr_mba = np.load(root_path + baby_or_adult +'/MEG/FFR/' + filename_ffr_mba + '_morph_roi.npy',allow_pickle=True)
    ffr_pa = np.load(root_path + baby_or_adult +'/MEG/FFR/' + filename_ffr_pa + '_morph_roi.npy',allow_pickle=True)
    FFR_accuracy = np.load(root_path + baby_or_adult + '/decoding/FFR_decoding_accuracy_roi.npy')
    PCFFR_accuracy = np.load(root_path + baby_or_adult + '/decoding/PCFFR_decoding_accuracy_roi.npy')
elif input_data == 'wholebrain':
    ffr_ba = np.load(root_path + baby_or_adult +'/MEG/FFR/' + filename_ffr_ba + '_morph.npy',allow_pickle=True)
    ffr_mba = np.load(root_path + baby_or_adult +'/MEG/FFR/' + filename_ffr_mba + '_morph.npy',allow_pickle=True)
    ffr_pa = np.load(root_path + baby_or_adult +'/MEG/FFR/' + filename_ffr_pa + '_morph.npy',allow_pickle=True)
    FFR_accuracy = np.load(root_path + baby_or_adult + '/decoding/FFR_decoding_accuracy_v.npy')
    PCFFR_accuracy = np.load(root_path + baby_or_adult + '/decoding/PCFFR_decoding_accuracy_v.npy')
    
#%%####################################### decoding spatial result
all_score = PCFFR_accuracy
acc_ind = np.where(np.array(all_score) > 0.5)

## visualize sensor
evoked = mne.read_evokeds(root_path + 'cbs_A123/sss_fif/cbs_A123_01_otp_raw_sss_proj_f_evoked_substd_cabr.fif')[0]
ch_name = np.array(evoked.ch_names)
evoked.info['bads'] = ch_name[acc_ind[0]].tolist() # hack the evoked.info['bads'] to visualize the high decoding accuracy sensor
evoked.plot_sensors(ch_type='all',kind ='3d')

## visualize ROI
label_names[acc_ind] # ctx-rh-bankssts reached 0.46363636 decoding accuracy for adults, ctx-rh-middletemporal reached 0.48214286 for infants
np.sort(all_score) 
np.argsort(all_score)

## visualize vertice
stc1.data = np.array([all_score,all_score]).transpose()
stc1.plot(src, clim=dict(kind="percent",pos_lims=[90,95,99]), subject='fsaverage', subjects_dir=subjects_dir)

#%%####################################### decoding temporal result: manually look up the hot spot above
## plot certain sensor time series 
s_ind = acc_ind[0]
plt.figure()
plot_err(np.squeeze(ffr_ba[:,s_ind,:]),'c',times)
plot_err(np.squeeze(ffr_mba[:,s_ind,:]),'b',times)
plot_err(np.squeeze(ffr_pa[:,s_ind,:]),'b',times)
plt.legend(['ba','','mba','','pa',''])
plt.xlabel('Time (s)')
plt.title('sensor' + str(s_ind))
plt.xlim([-0.02, 0.2])

## plot certain ROI time series 
ROI_ind = 12
plt.figure()
plt.subplot(311)
plot_err(np.squeeze(ffr_ba[:,ROI_ind,:]),'k',times)
plt.legend(['ba'])
plt.xlim([-0.02, 0.2])
plt.xlabel('Time (s)')
plt.title('vertno ' + label_names[ROI_ind])
plt.subplot(312)
plot_err(np.squeeze(ffr_mba[:,ROI_ind,:]),'k',times)
plt.legend(['mba'])
plt.xlim([-0.02, 0.2])
plt.xlabel('Time (s)')
plt.subplot(313)
plot_err(np.squeeze(ffr_pa[:,ROI_ind,:]),'k',times)
plt.legend(['pa'])
plt.xlim([-0.02, 0.2])
plt.xlabel('Time (s)')

## plot certain vertice time series
nv = 16017
v_ind = np.where(src[0]['vertno'] == nv) # conv_adult 25056, cont_adult 20041, conv_infant 23669, cont_infant 19842 
plt.figure()
plt.subplot(311)
plot_err(np.squeeze(ffr_ba[:,v_ind,:]),'k',times)
plt.legend(['ba'])
plt.xlim([-0.02, 0.2])
plt.xlabel('Time (s)')
plt.title('vertno ' + str(nv))
plt.subplot(312)
plot_err(np.squeeze(ffr_mba[:,v_ind,:]),'k',times)
plt.legend(['mba'])
plt.xlim([-0.02, 0.2])
plt.xlabel('Time (s)')
plt.subplot(313)
plot_err(np.squeeze(ffr_pa[:,v_ind,:]),'k',times)
plt.legend(['pa'])
plt.xlim([-0.02, 0.2])
plt.xlabel('Time (s)')

nv = 17236
v_ind = np.where(src[0]['vertno'] == nv) # conv_adult 25056, cont_adult 20041, conv_infant 23669, cont_infant 19842 
plt.figure()
plt.subplot(211)
plt.plot(times,EEG_pa_FFR.mean(0))
plt.xlim([-0.02, 0.2])
plt.title('pa EEG/MEG')
plt.subplot(212)
plt.plot(times,np.squeeze(data[:,v_ind,:].mean(0)),'k')
plt.xlim([-0.02, 0.2])
plt.legend(['v' + str(nv)])

