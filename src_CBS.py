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
from tqdm import tqdm

def do_foward(s):
    root_path='/media/tzcheng/storage2/CBS/'
    subjects_dir = '/media/tzcheng/storage2/subjects/'

    file_in = root_path + s + '/sss_fif/' 
    raw_file = mne.io.read_raw_fif(file_in  + s + '_01_otp_raw_sss.fif')
    trans=mne.read_trans(file_in + s + '-trans.fif')
    src=mne.read_source_spaces(subjects_dir + s + '/bem/' + s + '-vol-5-src.fif')
    bem=mne.read_bem_solution(subjects_dir +  s + '/bem/' + s + '-5120-5120-5120-bem-sol.fif')
    fwd=mne.make_forward_solution(raw_file.info,trans,src,bem,meg=True,eeg=False)
    mne.write_forward_solution(file_in + s +'-fwd.fif',fwd,overwrite=True)

    return fwd, src

def do_inverse_MMR(s,run, morph,ori):
    root_path='/media/tzcheng/storage2/CBS/'
    subjects_dir = '/media/tzcheng/storage2/subjects/'

    file_in = root_path + s + '/sss_fif/' + s
    fwd = mne.read_forward_solution(file_in + '-fwd.fif')
    cov = mne.read_cov(file_in + run + '_erm_otp_raw_sss_proj_fil50_mmr-cov.fif')
    epoch = mne.read_epochs(file_in + run + '_otp_raw_sss_proj_fil50_mmr_e.fif')
    evoked_s = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_fil50_evoked_substd_mmr.fif')[0]
    evoked_d1 = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_fil50_evoked_dev1_mmr.fif')[0]        
    evoked_d2 = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_fil50_evoked_dev2_mmr.fif')[0]
    
    inverse_operator = mne.minimum_norm.make_inverse_operator(epoch.info, fwd, cov,loose=1,depth=0.8)
    # src.save(file_in + '_src', overwrite=True)
    if ori == None:
        standard = mne.minimum_norm.apply_inverse((evoked_s), inverse_operator, pick_ori = ori)
        dev1 = mne.minimum_norm.apply_inverse((evoked_d1), inverse_operator, pick_ori = ori)
        dev2 = mne.minimum_norm.apply_inverse((evoked_d2), inverse_operator, pick_ori = ori)
        mmr1 = dev1 - standard
        mmr2 = dev2 - standard
    
    elif ori == 'vector': # only the mmr (dev - std) needs this part
        standard = mne.minimum_norm.apply_inverse((evoked_s), inverse_operator, pick_ori = ori)
        dev1 = mne.minimum_norm.apply_inverse((evoked_d1), inverse_operator, pick_ori = ori)
        dev2 = mne.minimum_norm.apply_inverse((evoked_d2), inverse_operator, pick_ori = ori)
        mmr1 = dev1 - standard
        mmr2 = dev2 - standard
        mmr1 = mmr1.magnitude()
        mmr2 = mmr2.magnitude()
    
    elif ori == 'sensor_sub':
        tmp_evoked_mmr1 = evoked_d1.data - evoked_s.data
        evoked_mmr1 = evoked_s.copy()
        evoked_mmr1.data = tmp_evoked_mmr1
        tmp_evoked_mmr2 = evoked_d2.data - evoked_s.data
        evoked_mmr2 = evoked_s.copy()
        evoked_mmr2.data = tmp_evoked_mmr2
        mmr1 = mne.minimum_norm.apply_inverse((evoked_mmr1), inverse_operator)
        mmr2 = mne.minimum_norm.apply_inverse((evoked_mmr2), inverse_operator)
    
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
        # standard_fsaverage = morph.apply(standard)
        # dev1_fsaverage = morph.apply(dev1)
        # dev2_fsaverage = morph.apply(dev2)
        mmr1_fsaverage = morph.apply(mmr1)
        mmr2_fsaverage = morph.apply(mmr2)
        
        # standard_fsaverage.save(file_in + '_std_' + str(ori) +'_morph', overwrite=True)
        # dev1_fsaverage.save(file_in + '_dev1_' + str(ori) +'_morph', overwrite=True)
        # dev2_fsaverage.save(file_in + '_dev2_' + str(ori) +'_morph', overwrite=True)
        mmr1_fsaverage.save(file_in + '_mmr1_' + str(ori) +'_morph', overwrite=True)
        mmr2_fsaverage.save(file_in + '_mmr2_' + str(ori) +'_morph', overwrite=True)
    else: 
        print('No morphing has been performed. The individual results may not be good to average.')
        standard.save(file_in + '_std_' + str(ori), overwrite=True)
        dev1.save(file_in + '_dev1_' + str(ori), overwrite=True)
        dev2.save(file_in + '_dev2_' + str(ori), overwrite=True)
        # mmr1.save(file_in + '_mmr1_' + str(ori), overwrite=True)
        # mmr2.save(file_in + '_mmr2_' + str(ori), overwrite=True)

def do_inverse_cABR(s,run, morph):
    root_path='/media/tzcheng/storage2/CBS/'
    subjects_dir = '/media/tzcheng/storage2/subjects/'

    file_in = root_path + s + '/sss_fif/' + s
    fwd = mne.read_forward_solution(file_in + '-fwd.fif')
    cov = mne.read_cov(file_in + run + '_erm_otp_raw_sss_proj_f_ffr-cov.fif')
    epoch = mne.read_epochs(file_in + run + '_otp_raw_sss_proj_f_cABR_e.fif')
    evoked_s = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_f_evoked_substd_cabr.fif')[0]
    evoked_d1 = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_f_evoked_dev1_cabr.fif')[0]        
    evoked_d2 = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_f_evoked_dev2_cabr.fif')[0]
    
    inverse_operator = mne.minimum_norm.make_inverse_operator(epoch.info, fwd, cov,loose=1,depth=0.8)

    ori = None # cabr should use None
    standard = mne.minimum_norm.apply_inverse((evoked_s), inverse_operator, pick_ori = ori)
    dev1 = mne.minimum_norm.apply_inverse((evoked_d1), inverse_operator, pick_ori = ori)
    dev2 = mne.minimum_norm.apply_inverse((evoked_d2), inverse_operator, pick_ori = ori)
    
    if morph == True:
        print('Morph ' + s +  ' src space to common cortical space.')
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
        standard_fsaverage = morph.apply(standard)
        dev1_fsaverage = morph.apply(dev1)
        dev2_fsaverage = morph.apply(dev2)
        standard_fsaverage.save(file_in + '_ba_cabr_morph', overwrite=True)
        dev1_fsaverage.save(file_in + '_mba_cabr_morph', overwrite=True)
        dev2_fsaverage.save(file_in + '_pa_cabr_morph', overwrite=True)

    else: 
        print('No morphing has been performed. The individual results may not be good to average.')
        standard.save(file_in + '_ba_cabr' + str(ori), overwrite=True)
        dev1.save(file_in + '_mba_cabr' + str(ori), overwrite=True)
        dev2.save(file_in + '_pa_cabr' + str(ori), overwrite=True)

########################################
root_path='/media/tzcheng/storage2/CBS/'
os.chdir(root_path)

morph = True
ori = 'vector' # 'vector', None. 'sensor_sub' # 'sensor_sub' is doing dev-std subtraction on the sensor level then source localization

runs = ['_01','_02']
run = runs[0]
subj = [] 
for file in os.listdir():
    if file.startswith('cbs_b'):
        subj.append(file)
for s in tqdm(subj):
    # for run in runs:
        print(s)
        # do_foward(s)
        do_inverse_MMR(s,run, morph,ori)
        # do_inverse_cABR(s,run, morph)