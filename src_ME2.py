#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 17:22:56 2024
LCMV beamformer for src in ME2
me2_320_11m and me2_324_11m have very small epoch files 

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
    # root_path='/media/tzcheng/storage/ME2_MEG/Zoe_analyses/11mo/'
    root_path = '/media/tzcheng/storage/BabyRhythm/'
    subjects_dir = '/media/tzcheng/storage2/subjects/'
    file_in = root_path + s + '/sss_fif/' 
    raw_file = mne.io.read_raw_fif(file_in  + s + '_01_otp_raw_sss.fif')
    trans=mne.read_trans(file_in + s + '-trans.fif')
    src=mne.read_source_spaces(subjects_dir + s + '/bem/' + s + '-vol-5-src.fif')
    bem=mne.read_bem_solution(subjects_dir +  s + '/bem/' + s + '-5120-5120-5120-bem-sol.fif')
    fwd=mne.make_forward_solution(raw_file.info,trans,src,bem,meg=True,eeg=False)
    mne.write_forward_solution(file_in + s +'-fwd.fif',fwd,overwrite=True)

    return fwd, src

def do_inverse(s,morph,run):
    # root_path='/media/tzcheng/storage/ME2_MEG/Zoe_analyses/11mo/'
    root_path = '/media/tzcheng/storage/BabyRhythm/'
    subjects_dir = '/media/tzcheng/storage2/subjects/'
    file_in = root_path + s + '/sss_fif/' + s
    fwd = mne.read_forward_solution(file_in + '-fwd.fif')
    epoch = mne.read_epochs(file_in + run + '_otp_raw_sss_proj_fil50_epoch.fif')
    noise_cov = mne.read_cov(file_in + run + '_erm_otp_raw_sss_proj_fil50-cov.fif')
    data_cov = mne.compute_covariance(epoch, tmin=0, tmax=None)
    # evoked = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_fil50_mag6pT_evoked.fif')[0]
    evoked_random_duple = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_fil50_mag6pT_evoked_randduple.fif')[0]
    evoked_random_triple = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_fil50_mag6pT_evoked_randtriple.fif')[0]
    
    ## can experiment on pick_ori
#     filters = make_lcmv(
#     evoked.info,
#     fwd,
#     data_cov,
#     reg=0.05,
#     noise_cov=noise_cov,
#     pick_ori="max-power",
#     weight_norm="unit-noise-gain",
#     rank='info',
# )
    
#     stc_lcmv = apply_lcmv(evoked, filters)
#     src = fwd['src']
   
    inverse_operator = mne.minimum_norm.make_inverse_operator(epoch.info, fwd, noise_cov,loose=1,depth=0.8)
    # stc_mne = mne.minimum_norm.apply_inverse((evoked), inverse_operator, pick_ori = None)
    stc_mne_random_duple = mne.minimum_norm.apply_inverse((evoked_random_duple), inverse_operator, pick_ori = None)
    stc_mne_random_triple = mne.minimum_norm.apply_inverse((evoked_random_triple), inverse_operator, pick_ori = None)

    if morph == True:
        print('Morph ' + s +  ' src space to common cortical space.')
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
        # stc_lcmv_fsaverage = morph.apply(stc_lcmv)
        # stc_lcmv_fsaverage.save(file_in + run + '_stc_lcmv_morph_mag6pT', overwrite=True)
        # stc_mne_fsaverage = morph.apply(stc_mne)
        # stc_mne_fsaverage.save(file_in + run + '_stc_mne_morph_mag6pT', overwrite=True)
        stc_mne_random_duple_fsaverage = morph.apply(stc_mne_random_duple)
        stc_mne_random_duple_fsaverage.save(file_in + run + '_stc_mne_morph_mag6pT_randduple', overwrite=True)
        stc_mne_random_triple_fsaverage = morph.apply(stc_mne_random_triple)
        stc_mne_random_triple_fsaverage.save(file_in + run + '_stc_mne_morph_mag6pT_randtriple', overwrite=True)
    else: 
        print('No morphing has been performed. The individual results may not be good to average.')
        # stc_lcmv.save(file_in + '_stc_lcmv', overwrite=True)
        # stc_mne.save(file_in + '_stc_mne', overwrite=True)

#%%#######################################   
## manually coregister to get the trans, bem and src.
# subjects_dir = '/media/tzcheng/storage2/subjects'
# mne.gui.coregistration(subject='fsaverage', subjects_dir=subjects_dir)

# root_path='/media/tzcheng/storage/ME2_MEG/Zoe_analyses/11mo/' # change to 11mo and /media/tzcheng/storage/BabyRhythm/
root_path = '/media/tzcheng/storage/BabyRhythm/'
os.chdir(root_path)

morph = True

runs = ['_02']
subj = [] 
for file in os.listdir():
    if file.startswith('br_'):
    # if file.startswith('me2_'):
        subj.append(file)

for s in tqdm(subj):
    # do_foward(s)
    for run in runs:
        print(s)
        do_inverse(s,morph,run)