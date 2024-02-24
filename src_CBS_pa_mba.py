#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 11:54:23 2023
New method to calculate source MMR in vector or magnitude method
MMR1 = first mba -last mba
MMR2 = first pa - last pa 

@author: tzcheng
"""

## Import library  
import mne
import matplotlib
import numpy as np
import os
import nibabel as nib
from tqdm import tqdm

def do_inverse(s,morph,ori):
    run = '_01'
    root_path='/media/tzcheng/storage2/CBS/'
    subject = s
    subjects_dir = '/media/tzcheng/storage2/subjects/'

    file_in = root_path + s + '/sss_fif/' + s
    fwd = mne.read_forward_solution(file_in + '-fwd.fif')
    trans = mne.read_trans(file_in +'-trans.fif')
    cov = mne.read_cov(file_in + run + '_erm_otp_raw_sss_proj_fil50_mmr-cov.fif')
    epoch = mne.read_epochs(file_in + run + '_otp_raw_sss_proj_fil50_mmr_e.fif')
    evoked_s1 = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_fil50_evoked_substd1_reverse_mmr.fif')[0] # last mba
    evoked_d1 = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_fil50_evoked_dev1_mmr.fif')[0] # first mba
    evoked_s2 = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_fil50_evoked_substd2_reverse_mmr.fif')[0] # last pa
    evoked_d2 = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_fil50_evoked_dev2_mmr.fif')[0] # first pa
    
    inverse_operator = mne.minimum_norm.make_inverse_operator(epoch.info, fwd, cov,loose=1,depth=0.8)

    standard1 = mne.minimum_norm.apply_inverse((evoked_s1), inverse_operator, pick_ori = ori)
    standard2 = mne.minimum_norm.apply_inverse((evoked_s2), inverse_operator, pick_ori = ori)
    dev1 = mne.minimum_norm.apply_inverse((evoked_d1), inverse_operator, pick_ori = ori)
    dev2 = mne.minimum_norm.apply_inverse((evoked_d2), inverse_operator, pick_ori = ori)
    mmr1 = dev1 - standard1
    mmr2 = dev2 - standard2
    # src = inverse_operator['src']
    # src.save(file_in + '_src', overwrite=True)

    if ori == 'vector': # only the mmr (dev - std) needs this part
        mmr1 = mmr1.magnitude()
        mmr2 = mmr2.magnitude()
    
    if morph == True:
        print('Morph' + s +  'src space to common cortical space.')
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
        # standard1_fsaverage = morph.apply(standard1)
        # standard2_fsaverage = morph.apply(standard2)
        # dev1_fsaverage = morph.apply(dev1)
        # dev2_fsaverage = morph.apply(dev2)
        mmr1_fsaverage = morph.apply(mmr1)
        mmr2_fsaverage = morph.apply(mmr2)
        
        # standard_fsaverage.save(file_in + '_std_' + str(ori) +'_morph', overwrite=True)
        # dev1_fsaverage.save(file_in + '_dev1_' + str(ori) +'_morph', overwrite=True)
        # dev2_fsaverage.save(file_in + '_dev2_' + str(ori) +'_morph', overwrite=True)
        mmr1_fsaverage.save(file_in + '_mmr1_mba_' + str(ori) +'_morph', overwrite=True)
        mmr2_fsaverage.save(file_in + '_mmr2_pa_' + str(ori) +'_morph', overwrite=True)
    else: 
        print('No morphing has been performed. The individual results may not be good to average.')
        # standard.save(file_in + '_std_' + str(ori), overwrite=True)
        # dev1.save(file_in + '_dev1_' + str(ori), overwrite=True)
        # dev2.save(file_in + '_dev2_' + str(ori), overwrite=True)
        mmr1.save(file_in + '_mmr1_mba_' + str(ori), overwrite=True)
        mmr2.save(file_in + '_mmr2_pa_' + str(ori), overwrite=True)

########################################
root_path='/media/tzcheng/storage2/CBS/'
os.chdir(root_path)

morph = True
ori = 'vector' # 'vector', None

runs = ['_01','_02']
subj = [] 
for file in os.listdir():
    if file.startswith('cbs_b'):
        subj.append(file)

for s in tqdm(subj):
    # for run in runs:
        print(s)
        # do_foward(s)
        do_inverse(s,morph,ori)