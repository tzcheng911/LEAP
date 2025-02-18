#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 13:46:51 2025

Investigate the ring pattern in cbs_A114

@author: tzcheng
"""

import mne
import os

root_path='/media/tzcheng/storage2/CBS/'
subjects_dir = '/media/tzcheng/storage2/subjects/'
os.chdir(root_path)

## Parameters
fmin = 50
fmax = 150
sfreq = 5000
lb = 90
hb = 100
n_trial = 'all'
runs = ['_01']
cond = ['dev1']

subjects = [] 
for file in os.listdir():
    if file.startswith('cbs_A114'):
        subjects.append(file)
s = subjects[0]
file_in = root_path + s + '/sss_fif/' + s
evokeds = mne.read_evokeds(file_in + runs[0] + '_otp_raw_sss_proj_f80450_evoked_' + cond[0] + '_ffr_' + str(n_trial) +'.fif')[0]
fwd = mne.read_forward_solution(file_in + '-fwd.fif')
cov = mne.read_cov(file_in + runs[0] + '_erm_otp_raw_sss_proj_f80450_ffr-cov.fif')
epoch = mne.read_epochs(file_in + runs[0] + '_otp_raw_sss_proj_f80450_ffr_e_' + str(n_trial) + '.fif')
inverse_operator = mne.minimum_norm.make_inverse_operator(epoch.info, fwd, cov,loose=1,depth=0.8)
evokeds_inv_stc = mne.minimum_norm.apply_inverse((evokeds), inverse_operator, pick_ori = None)
        
evokeds_inv_stc.plot(src = inverse_operator['src'])
   
fname_src_fsaverage = subjects_dir + 'fsaverage/bem/fsaverage-vol-5-src.fif'
src_fs = mne.read_source_spaces(fname_src_fsaverage)
morph = mne.compute_source_morph(
    inverse_operator["src"],
    subject_from=s,
    subjects_dir=subjects_dir,
    niter_affine=[10, 10, 5],
    niter_sdr=[10, 10, 5],  # just for speed
    src_to=src_fs,
    verbose=True)
evokeds_inv_stc_fsaverage = morph.apply(evokeds_inv_stc)

## Visualization
trans=mne.read_trans(file_in + s + '-trans.fif')
mne.viz.plot_alignment(
    epoch.info,
    trans=trans,
    subject="cbs_A114",
    src=inverse_operator['src'],
    subjects_dir=subjects_dir,
    dig=True,
    surfaces=["head-dense", "white"],
    coord_frame="meg",
)