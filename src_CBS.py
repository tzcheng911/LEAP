#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 15:01:42 2023
Used for looping subjects to save their fwd and stc + morphed data
BEM, src and trans files should be saved from the coreg process done manually
@author: tzcheng
"""

## Import library  
import mne
import matplotlib
import numpy as np
import os
import nibabel as nib
from mne.datasets import sample, fetch_fsaverage

def do_foward(s):
    root_path='/media/tzcheng/storage/CBS/'
    subjects_dir = '/media/tzcheng/storage2/subjects/'

    file_in = root_path + '/' + s + '/sss_fif/' 
    raw_file = mne.io.read_raw_fif(file_in + '/' + s + '_01_otp_raw_sss.fif')
    trans=mne.read_trans(file_in + '-trans.fif')
    src=mne.read_source_spaces(subjects_dir + '/' + s + '/bem/' + s + '-vol5-src.fif')
    bem=mne.read_bem_solution(subjects_dir + '/' + s + '/bem/' + s + '-5120-5120-5120-bem-sol.fif')
    fwd=mne.make_forward_solution(raw_file.info,trans,src,bem,meg=True,eeg=False)
    mne.write_forward_solution(file_in + '-fwd.fif',fwd,overwrite=True)

    return fwd, src


def do_inverse(s,morph):
    root_path='/media/tzcheng/storage/CBS/'
    subject = s
    subjects_dir = '/media/tzcheng/storage2/subjects/'

    file_in = root_path + '/' + s + '/sss_fif/' + s
    fwd = mne.read_forward_solution(file_in + '-fwd.fif')
    trans = mne.read_trans(file_in +'-trans.fif')
    cov = mne.read_cov(file_in + run + '_erm_otp_raw_sss_proj_fil50-cov.fif')
    epoch = mne.read_epochs(file_in + run + '_otp_raw_sss_proj_fil50_mmr_e.fif')
   
    evoked_s = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_fil50_evoked_substd_mmr.fif')[0]
    evoked_d1 = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_fil50_evoked_dev1_mmr.fif')[0]        
    evoked_d2 = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_fil50_evoked_dev2_mmr.fif')[0]
        
    inverse_operator = mne.minimum_norm.make_inverse_operator(epoch.info, fwd, cov,loose=1,depth=0.8)
    standard = mne.minimum_norm.apply_inverse((evoked_s), inverse_operator)
    dev1 = mne.minimum_norm.apply_inverse((evoked_d1), inverse_operator)
    dev2 = mne.minimum_norm.apply_inverse((evoked_d2), inverse_operator)
    mmr1 = dev1 - standard
    mmr2 = dev2 - standard

    src = inverse_operator['src']
        
    standard.save(file_in + '_substd_mmr', overwrite=True)
    dev1.save(file_in + '_dev1_mmr', overwrite=True)
    dev2.save(file_in + '_dev2_mmr', overwrite=True)
    mmr1.save(file_in + '_mmr1', overwrite=True)
    mmr2.save(file_in + '_mmr2', overwrite=True)
    src.save(file_in + '_src', overwrite=True)
    
    if morph == True:
        print('Morph individual src space to common cortical space.')
        fetch_fsaverage(subjects_dir)  # ensure fsaverage src exists
        fname_src_fsaverage = subjects_dir + "/fsaverage/bem/fsaverage-vol-5-src.fif"
        src_fs = mne.read_source_spaces(fname_src_fsaverage)
        morph = mne.compute_source_morph(
        inverse_operator["src"],
        subject_from=s,
        subjects_dir=subjects_dir,
        niter_affine=[10, 10, 5],
        niter_sdr=[10, 10, 5],  # just for speed
        src_to=src_fs,
        verbose=True)

        mmr1_fsaverage = morph.apply(mmr1)
        mmr2_fsaverage = morph.apply(mmr2)

        mmr1_fsaverage.save(file_in + '_mmr1_morph', overwrite=True)
        mmr2_fsaverage.save(file_in + '_mmr2_morph', overwrite=True)
    else: 
        print('No morphing has been performed. The individual results may not be good to average.')

    return mmr1, mmr2, mmr1_fsaverage, mmr2_fsaverage, src, inverse_operator

########################################
root_path='/media/tzcheng/storage/CBS/'
os.chdir(root_path)

morph = True

runs = ['_01','_02']
subj = [] 
for file in os.listdir():
    if file.startswith('cbs_A'):
        subj.append(file)

for s in subj:
    for run in runs:
        print(s)
        do_foward(s)
        do_inverse(s,morph)