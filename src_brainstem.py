#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 21:44:27 2025
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
from tqdm import tqdm

def do_foward(s):
    root_path='/media/tzcheng/storage/Brainstem/'
    subjects_dir = '/media/tzcheng/storage2/subjects/'

    file_in = root_path + s + '/sss_fif/' 
    raw_file = mne.io.read_raw_fif(file_in  + s + '_p10_01_otp_raw_sss.fif') # get the info for object with information about the sensors and methods of measurement
    trans=mne.read_trans(file_in + s + '-trans.fif')
    src=mne.read_source_spaces(subjects_dir + s + '_zoe/bem/' + s + '_zoe-vol-5-src.fif')
    bem=mne.read_bem_solution(subjects_dir +  s + '_zoe/bem/' + s + '_zoe-5120-5120-5120-bem-sol.fif')
    fwd=mne.make_forward_solution(raw_file.info,trans,src,bem,meg=True,eeg=False)
    mne.write_forward_solution(file_in + s +'-fwd.fif',fwd,overwrite=True)
    return fwd, src

def do_inverse_cABR(s,run, condition,morph):
    root_path='/media/tzcheng/storage/Brainstem/'
    subjects_dir = '/media/tzcheng/storage2/subjects/'

    file_in = root_path + s + '/sss_fif/' + s
    fwd = mne.read_forward_solution(file_in + '-fwd.fif')
    cov = mne.read_cov(file_in + run + '_erm_otp_raw_sss_proj_f80200_ffr-cov.fif')
    epoch = mne.read_epochs(file_in + condition + run + '_otp_raw_sss_proj_f80200_ffr_e.fif')
    evoked = mne.read_evokeds(file_in + condition + run + '_otp_raw_sss_proj_f80200_evoked_ffr.fif')[0]
    
    inverse_operator = mne.minimum_norm.make_inverse_operator(epoch.info, fwd, cov,loose=1,depth=0.8)

    ori = None # cabr should use None
    stc = mne.minimum_norm.apply_inverse((evoked), inverse_operator, pick_ori = ori)
    
    if morph == True:
        print('Morph ' + s +  ' src space to common cortical space.')
        fname_src_fsaverage = subjects_dir + 'fsaverage/bem/fsaverage-vol-5-src.fif'
        src_fs = mne.read_source_spaces(fname_src_fsaverage)
        morph = mne.compute_source_morph(
            inverse_operator["src"],
            subject_from=s+'_zoe',
            subjects_dir=subjects_dir,
            niter_affine=[10, 10, 5],
            niter_sdr=[10, 10, 5],  # just for speed
            src_to=src_fs,
            verbose=True)
        stc_fsaverage = morph.apply(stc)
        stc_fsaverage.save(file_in + condition + run + '_morph', overwrite=True)

    else: 
        print('No morphing has been performed. The individual results may not be good to average.')
        stc.save(file_in + '_ba_ffr_all', overwrite=True)

#%%########################################
root_path='/media/tzcheng/storage/Brainstem/'
os.chdir(root_path)

morph = True
ori = 'vector' # 'vector', None. 'sensor_sub' # 'sensor_sub' is doing dev-std subtraction on the sensor level then source localization

conditions = ['_p10','_n40']
runs = ['_01','_02'] 

subj = [] 
for file in os.listdir():
    if file.startswith('brainstem_'):
        subj.append(file)
for s in tqdm(subj):
    print(s)
    for condition in conditions:
        for run in runs:
            do_foward(s)
            do_inverse_cABR(s,run, condition, morph)