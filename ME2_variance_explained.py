#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 15:57:22 2026

@author: tzcheng
"""

import mne

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