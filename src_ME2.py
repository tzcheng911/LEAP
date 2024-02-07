#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 17:22:56 2024
LCMV beamformer for src in ME2

@author: tzcheng
"""

## Import library  
import mne
import matplotlib
import numpy as np
import os
import nibabel as nib
from mne.beamformer import apply_lcmv, make_lcmv
from tqdm import tqdm

def do_foward(s):
    root_path='/media/tzcheng/storage/ME2_MEG/Zoe_analyses/7mo/'
    subjects_dir = '/media/tzcheng/storage2/subjects/'

    file_in = root_path + s + '/sss_fif/' 
    raw_file = mne.io.read_raw_fif(file_in  + s + '_01_otp_raw_sss.fif')
    trans=mne.read_trans(file_in + s + '_trans.fif')
    src=mne.read_source_spaces(subjects_dir + s + '/bem/' + s + '-vol-5-src.fif')
    bem=mne.read_bem_solution(subjects_dir +  s + '/bem/' + s + '-5120-5120-5120-bem-sol.fif')
    fwd=mne.make_forward_solution(raw_file.info,trans,src,bem,meg=True,eeg=False)
    mne.write_forward_solution(file_in + s +'-fwd.fif',fwd,overwrite=True)

    return fwd, src

def do_inverse(s,morph,ori,run):
    root_path='/media/tzcheng/storage/ME2_MEG/Zoe_analyses/7mo/'
    subjects_dir = '/media/tzcheng/storage2/subjects/'

    file_in = root_path + s + '/sss_fif/' + s
    fwd = mne.read_forward_solution(file_in + '-fwd.fif')

    epoch = mne.read_epochs(file_in + run + '_otp_raw_sss_proj_fil50_epoch.fif')
    noise_cov = mne.read_cov(file_in + run + '_erm_otp_raw_sss_proj_fil50_mmr-cov.fif')
    data_cov = mne.compute_covariance(epoch, tmin=0, tmax=None, method="empirical")
    evoked = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_fil50_evoked.fif')[0]
    
    ## can experiment on pick_ori
    filters = make_lcmv(
    evoked.info,
    fwd,
    data_cov,
    reg=0.05,
    noise_cov=noise_cov,
    pick_ori="max-power",
    weight_norm="unit-noise-gain",
    rank=None,
)
    
    stc = apply_lcmv(evoked, filters)
    src = fwd['src']
   
    if morph == True:
        print('Morph' + s +  'src space to common cortical space.')
        fname_src_fsaverage = subjects_dir + 'fsaverage/bem/fsaverage-vol-5-src.fif'
        src_fs = mne.read_source_spaces(fname_src_fsaverage)
        morph = mne.compute_source_morph(
            fwd["src"],
            subject_from=s,
            subjects_dir=subjects_dir,
            niter_affine=[10, 10, 5],
            niter_sdr=[10, 10, 5],  # just for speed
            src_to=src_fs,
            verbose=True)
        stc_fsaverage = morph.apply(stc)
        stc_fsaverage.save(file_in + 'stc_morph', overwrite=True)

    else: 
        print('No morphing has been performed. The individual results may not be good to average.')
        stc.save(file_in + '_stc', overwrite=True)

########################################
root_path='/media/tzcheng/storage/ME2_MEG/Zoe_analyses/7mo/'
os.chdir(root_path)

morph = True
ori = None # 'vector', None

runs = ['_01','_02','_03','_04']
subj = [] 
for file in os.listdir():
    if file.startswith('me2_'):
        subj.append(file)

for s in tqdm(subj):
    do_foward(s)
    for run in runs:
        print(s)
        do_inverse(s,run,morph)