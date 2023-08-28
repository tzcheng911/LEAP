# -*- coding: utf-8 -*-
"""
Created on Tue Aug 8 13:50:12 2023
For the subjects who DONT have individual MRI, use freesurfer fsaverage instead

@author: tzcheng
"""

###### Import library 
import mne
import mnefun
import matplotlib
from mne.preprocessing import maxwell_filter
import numpy as np
from scipy import linalg
from mne.io.constants import FIFF
import os

########################################
root_path='/media/tzcheng/storage/vmmr/'
os.chdir(root_path)

## parameters 
runs = ['_1','_2','_3','_4']

subj = [] 
for file in os.listdir():
    if file.startswith('vMMR_902'):
        subj.append(file)

run = runs[0]
s = subj[0]

subjects_dir = '/media/tzcheng/storage2/subjects/'
subject = s

fname = root_path + s + '/sss_fif/' + s + run
fname_erm =  root_path + s + '/sss_fif/' + s + run + '_erm'
raw_fname = fname + '_raw_sss.fif'
clean_sss_raw = mne.io.read_raw_fif(raw_fname,allow_maxshield=True,preload=True)
cov = mne.read_cov(fname_erm + '_raw_sss_proj_fil50-cov')
trans = mne.read_trans(root_path + s + '/sss_fif/' + s + '-trans.fif')
bem = mne.read_bem_solution(subjects_dir + subject + '/bem/' + subject + '-5120-5120-5120-bem-sol.fif')
src = mne.read_source_spaces(subjects_dir + subject + '/bem/' + subject + '-ico-5-src.fif')

info = mne.io.read_info(raw_fname)


#%%
## visualization for BEM
plot_bem_kwargs = dict(
    subject=subject,
    subjects_dir=subjects_dir,
    brain_surfaces="white",
    orientation="coronal",
    slices=[50, 100, 150, 200],
)
mne.viz.plot_bem(**plot_bem_kwargs)

## visualization for src
mne.viz.plot_bem(src=src, **plot_bem_kwargs)

## visualization for the src, brain, head and sensor 
mne.viz.plot_alignment(
    clean_sss_raw.info,
    trans=trans,
    subject=s,
    src=src,
    subjects_dir=subjects_dir,
    dig=True,
    surfaces=["head-dense", "white"],
    coord_frame="meg",
)

## do forward solution
fwd = mne.make_forward_solution(
    raw_fname,
    trans=trans,
    src=src,
    bem=bem,
    meg=True,
    eeg=False,
    mindist=5.0,
    n_jobs=None,
    verbose=True,
)
print(fwd)

leadfield = fwd["sol"]["data"]
mne.write_forward_solution( root_path + s + '/sss_fif/' + s + '_fwd_test.fif', fwd,overwrite=True)

