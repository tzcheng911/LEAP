#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 09:18:22 2023

@author: ashdrew
"""

import mne

subj = 'atp_a115'

data_file = '/mnt/storage/ATP/'+subj+'/sss_fif/'+subj+'_01_raw_sss.fif'
erm_file = '/mnt/storage/ATP/' + subj + '/sss_fif/' + subj + '_erm_raw_sss.fif'
raw_erm = mne.io.read_raw_fif(erm_file, preload=True)
raw = mne.io.read_raw_fif(data_file, preload=True)


#ECG & EOG
ecg_evoked = mne.preprocessing.create_ecg_epochs(raw,ch_name='ECG001').average()
ecg_evoked.plot_joint()
    
ecg_projs, ecg_events = mne.preprocessing.compute_proj_ecg(raw, ch_name='ECG001', n_grad=1, n_mag=1, reject=None)
    
fig = mne.viz.plot_projs_joint(ecg_projs, ecg_evoked)
fig.suptitle('ECG projectors')

eog_evoked = mne.preprocessing.create_eog_epochs(raw,ch_name=['EOG002','EOG003']).average() #
eog_evoked.plot_joint()
    
eog_projs, eog_events = mne.preprocessing.compute_proj_eog(raw, ch_name=['EOG002','EOG003'], n_grad=1, n_mag=1, reject=None)
    
fig = mne.viz.plot_projs_joint(eog_projs, eog_evoked)
fig.suptitle('EOG projectors')
    
raw.add_proj(ecg_projs)
raw.add_proj(eog_projs)
raw_erm.add_proj(ecg_projs)
raw_erm.add_proj(eog_projs)


#filter
raw.filter(l_freq=0,h_freq=50,method='iir',iir_params=dict(order=4,ftype='butter'))
raw_erm.filter(l_freq=0,h_freq=50,method='iir',iir_params=dict(order=4,ftype='butter'))

raw.save('/mnt/storage/ATP/'+ subj + '/sss_fif/' + subj + '_01_allclean_fil50_raw_sss.fif', overwrite=True)
raw_erm.save('/mnt/storage/ATP/'+ subj + '/sss_fif/' + subj + '_erm_allclean_fil50_raw_sss.fif', overwrite=True)

#raw.info['projs']

#Noise Covariance
noise_cov = mne.compute_raw_covariance(raw_erm, tmin=0, tmax=None)
noise_cov.plot(raw_erm.info, proj=True)
mne.write_cov('/mnt/storage/ATP/' + subj + '/sss_fif/' + subj+ '_erm_50_sss-cov.fif', noise_cov, overwrite=True)