#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 13:56:04 2026

@author: tzcheng
"""

#%% Import library  
import mne
import os

#%% set the path
root_path='/media/tzcheng/storage2/CBS/'
subjects_dir = '/media/tzcheng/storage2/subjects/'
os.chdir(root_path)


subj = [] 
for file in os.listdir():
    if file.startswith('cbs_A'):
        subj.append(file)

#%% do the jobs
for s in subj:
    print(s)
    file_in = root_path + s + '/sss_fif/' 
    raw_file = mne.io.read_raw_fif(file_in  + s + '_01_otp_raw_sss.fif')
    trans=mne.read_trans(file_in + s + '-trans.fif')
    src=mne.read_source_spaces(subjects_dir + s + '/bem/' + s + '-vol-5-src.fif')
    bem=mne.read_bem_solution(subjects_dir +  s + '/bem/' + s + '-5120-5120-5120-bem-sol.fif')
    
    fig = mne.viz.plot_alignment(
        info=raw_file.info,
        trans=trans,         # Transformation file
        subject=s,
        subjects_dir=subjects_dir,
        surfaces='brain', # Show dense head surface
        src=src,
        bem=bem
    )
    mne.viz.set_3d_view(fig, azimuth=180, elevation=90)