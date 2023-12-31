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
    root_path='/media/tzcheng/storage/CBS/'
    subjects_dir = '/media/tzcheng/storage2/subjects/'

    file_in = root_path + s + '/sss_fif/' 
    raw_file = mne.io.read_raw_fif(file_in  + s + '_01_otp_raw_sss.fif')
    trans=mne.read_trans(file_in + s + '-trans.fif')
    src=mne.read_source_spaces(subjects_dir + s + '/bem/' + s + '-vol-5-src.fif')
    bem=mne.read_bem_solution(subjects_dir +  s + '/bem/' + s + '-5120-5120-5120-bem-sol.fif')
    fwd=mne.make_forward_solution(raw_file.info,trans,src,bem,meg=True,eeg=False)
    mne.write_forward_solution(file_in + s +'-fwd.fif',fwd,overwrite=True)

    return fwd, src


def do_inverse(s,morph,ori,direction):
    run = '_01'
    root_path='/media/tzcheng/storage/CBS/'
    subject = s
    subjects_dir = '/media/tzcheng/storage2/subjects/'

    file_in = root_path + '/' + s + '/sss_fif/' + s
    fwd = mne.read_forward_solution(file_in + '-fwd.fif')
    trans = mne.read_trans(file_in +'-trans.fif')
    cov = mne.read_cov(file_in + run + '_erm_otp_raw_sss_proj_fil50_mmr-cov.fif')
    
    if direction == 'pa_to_ba':
        epoch = mne.read_epochs(file_in + run + '_otp_raw_sss_proj_fil50_mmr_reverse_e.fif')
        evoked_s1 = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_fil50_evoked_substd1_reverse_mmr.fif')[0]
        evoked_s2 = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_fil50_evoked_substd2_reverse_mmr.fif')[0]
        evoked_d = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_fil50_evoked_dev_reverse_mmr.fif')[0]
        
        inverse_operator = mne.minimum_norm.make_inverse_operator(epoch.info, fwd, cov,loose=1,depth=0.8)
        standard1 = mne.minimum_norm.apply_inverse((evoked_s1), inverse_operator, pick_ori = ori)
        standard2 = mne.minimum_norm.apply_inverse((evoked_s2), inverse_operator, pick_ori = ori)
        dev = mne.minimum_norm.apply_inverse((evoked_d), inverse_operator, pick_ori = ori)
        mmr1 = dev - standard1
        mmr2 = dev - standard2
        src = inverse_operator['src']

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
            dev_fsaverage = morph.apply(dev)
            standard1_fsaverage = morph.apply(standard1)
            standard2_fsaverage = morph.apply(standard2)
            mmr1_fsaverage = morph.apply(mmr1)
            mmr2_fsaverage = morph.apply(mmr2)
            
            dev_fsaverage.save(file_in + '_dev_reverse_' + str(ori) +'_morph', overwrite=True)
            standard1_fsaverage.save(file_in + '_std1_reverse_' + str(ori) +'_morph', overwrite=True)
            standard2_fsaverage.save(file_in + '_std2_reverse_' + str(ori) +'_morph', overwrite=True)
            mmr1_fsaverage.save(file_in + '_mmr1_reverse_' + str(ori) +'_morph', overwrite=True)
            mmr2_fsaverage.save(file_in + '_mmr2_reverse_' + str(ori) +'_morph', overwrite=True)
        else: 
            print('No morphing has been performed. The individual results may not be good to average.')
            standard1.save(file_in + '_std1_reverse_' + str(ori), overwrite=True)
            standard2.save(file_in + '_std2_reverse_' + str(ori), overwrite=True)
            dev.save(file_in + '_dev_reverse_' + str(ori), overwrite=True)
            mmr1.save(file_in + '_mmr1_reverse_' + str(ori), overwrite=True)
            mmr2.save(file_in + '_mmr2_reverse_' + str(ori), overwrite=True)
    
    elif direction == 'first_last_ba':
        epoch = mne.read_epochs(file_in + run + '_otp_raw_sss_proj_fil50_mmr_ba_e.fif')
        evoked_s = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_fil50_evoked_substd_ba_mmr.fif')[0]
        evoked_d1 = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_fil50_evoked_dev1_ba_mmr.fif')[0]
        evoked_d2 = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_fil50_evoked_dev2_ba_mmr.fif')[0]
        
        inverse_operator = mne.minimum_norm.make_inverse_operator(epoch.info, fwd, cov,loose=1,depth=0.8)
        standard = mne.minimum_norm.apply_inverse((evoked_s), inverse_operator, pick_ori = ori)
        dev1 = mne.minimum_norm.apply_inverse((evoked_d1), inverse_operator, pick_ori = ori)
        dev2 = mne.minimum_norm.apply_inverse((evoked_d2), inverse_operator, pick_ori = ori)
        mmr1 = dev1 - standard
        mmr2 = dev2 - standard
        src = inverse_operator['src']

        if ori == 'vector':
            mmr1 = mmr1.magnitude()
            mmr2 = mmr2.magnitude()
            standard = standard.magnitude()
            dev1 = dev1.magnitude()
            dev2 = dev2.magnitude()
        
        if morph == True:
            print('Morph ' + s +  'src space to common cortical space.')
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
            mmr1_fsaverage = morph.apply(mmr1)
            mmr2_fsaverage = morph.apply(mmr2)
            
            dev1_fsaverage.save(file_in + '_dev1_ba_' + str(ori) +'_morph', overwrite=True)
            dev2_fsaverage.save(file_in + '_dev2_ba_' + str(ori) +'_morph', overwrite=True)
            standard_fsaverage.save(file_in + '_std_ba_' + str(ori) +'_morph', overwrite=True)
            mmr1_fsaverage.save(file_in + '_mmr1_ba_' + str(ori) +'_morph', overwrite=True)
            mmr2_fsaverage.save(file_in + '_mmr2_ba_' + str(ori) +'_morph', overwrite=True)
        else: 
            print('No morphing has been performed. The individual results may not be good to average.')
            standard.save(file_in + '_std_ba_' + str(ori), overwrite=True)
            dev1.save(file_in + '_dev1_ba_' + str(ori), overwrite=True)
            dev2.save(file_in + '_dev2_ba_' + str(ori), overwrite=True)
            mmr1.save(file_in + '_mmr1_ba_' + str(ori), overwrite=True)
            mmr2.save(file_in + '_mmr2_ba_' + str(ori), overwrite=True)

########################################
root_path='/media/tzcheng/storage/CBS/'
os.chdir(root_path)

morph = True
ori = None # 'vector', None
direction = 'first_last_ba' # traditional direction 'ba_to_pa': ba to pa and ba to mba
# reverse direction 'pa_to_ba' : is pa to ba and mba to ba; 
# only comparing /ba/ 'first_last_ba': only comparing /ba/ before and after habituation 

runs = ['_01','_02']
subj = [] 
for file in os.listdir():
    if file.startswith('cbs_A'):
        subj.append(file)

for s in tqdm(subj):
    # for run in runs:
        print(s)
        # do_foward(s)
        do_inverse(s,morph,ori,direction)