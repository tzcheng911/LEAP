# -*- coding: utf-8 -*-
"""
Created on Tue Aug 8 13:50:12 2023
For the subjects who have individual MRI

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
root_path='/media/tzcheng/storage/CBS/'
os.chdir(root_path)

## parameters 
runs = ['_01','_02']

subj = [] 
for file in os.listdir():
    if file.startswith('cbs_A'):
        subj.append(file)

run = runs[0]
s = subj[0]

fname = root_path + s + '/sss_fif/' + s + run
fname_erm =  root_path + s + '/sss_fif/' + s + '_erm'
raw_fname = fname + '_otp_raw_sss.fif'
clean_sss_raw = mne.io.read_raw_fif(raw_fname,allow_maxshield=True,preload=True)

cov = mne.read_cov(fname_erm + '_raw_sss_clean_fil50-cov.fif')
trans = mne.read_trans(root_path + s + '/source/' + s + '_1'  + '_raw-trans.fif')

info = mne.io.read_info(raw_fname)
#%% Forward modeling 

## freesurfer MRI reconstruction
# in commend line
# export FREESURFER_HOME=/home/tzcheng/freesurfer
# source $FREESURFER_HOME/SetUpFreeSurfer.sh
# export SUBJECTS_DIR=/media/tzcheng/storage/vmmr/vMMR_901
# my_subject=sample # or change to whatever your subject called
# my_NIfTI=/media/tzcheng/storage/vmmr/vMMR_901/sub-JA_T1w.nii.gz
# recon-all -i $my_NIfTI -s $my_subject -all
# mne _ --subject=$my_subject --subjects-dir=SUBJECTS_DIR --overwrite
# mne make_scalp_surfaces --subject=$my_subject --overwrite # can use -f to Force creation of the surface even if it has some topological defects.

## visualize freesurfer output
subject = s
subjects_dir = '/media/tzcheng/storage2/subjects'
# Brain = mne.viz.get_brain_class()
# brain = Brain(
#     "sample", hemi="lh", surf="pial", subjects_dir=subjects_dir, size=(800, 600))
# brain.add_annotation("aparc.a2009s", borders=False)

## trans (co-registration) 
# in commend line 
# use mne coreg after all the freesurfer steps
mne.gui.coregistration(subject=subject, subjects_dir=subjects_dir)

#%%
## BEM
# this requires precomputation using watershed
conductivity = (0.3,)  # for single layer
# conductivity = (0.3, 0.006, 0.3)  # for three layers
model = mne.make_bem_model(
    subject=s, ico=4, conductivity=conductivity, subjects_dir=subjects_dir
)
bem = mne.make_bem_solution(model)

plot_bem_kwargs = dict(
    subject=subject,
    subjects_dir=subjects_dir,
    brain_surfaces="white",
    orientation="coronal",
    slices=[50, 100, 150, 200],
)

mne.viz.plot_bem(**plot_bem_kwargs)

## src
# surface-based src
src = mne.setup_source_space(
    subject, spacing="ico5", add_dist="patch", subjects_dir=subjects_dir
)
print(src)
mne.viz.plot_bem(src=src, **plot_bem_kwargs)

# volume-based src
sphere = (0.0, 0.0, 0.04, 0.09)
vol_src = mne.setup_volume_source_space(
    subject,
    subjects_dir=subjects_dir,
    sphere=sphere,
    sphere_units="m",
    add_interpolator=False,
)  # just for speed!
print(vol_src)

mne.viz.plot_bem(src=vol_src, **plot_bem_kwargs)

## visualization for the src, brain, head and sensor 
mne.viz.plot_alignment(
    clean_sss_raw.info,
    trans=trans,
    subject="sample",
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
mne.write_forward_solution(fname + '_fwd.fif', fwd)

#%% Inverse modeling
## Load the evoked responses (1,2,3,4) of interest
fwd = mne.read_forward_solution('/media/tzcheng/storage/vmmr/vMMR_901/source/vMMR_901_1_fwd.fif')
evoked_s = mne.read_evokeds('/media/tzcheng/storage/vmmr/vMMR_902/sss_fif/vMMR_902_4_raw_sss_clean_fil50_evoked_s.fif')[0]
evoked_d = mne.read_evokeds('/media/tzcheng/storage/vmmr/vMMR_902/sss_fif/vMMR_902_4_raw_sss_clean_fil50_evoked_d.fif')[0]

inverse_operator = mne.minimum_norm.make_inverse_operator(clean_sss_raw.info, fwd, cov,loose=0.2,depth=0.8)
standard = mne.minimum_norm.apply_inverse((evoked_s), inverse_operator)
deviant = mne.minimum_norm.apply_inverse((evoked_d), inverse_operator)
mmr = deviant - standard

mmr.plot(subject=subject, subjects_dir=subjects_dir, hemi='both')

def do_forward(subj,raw):
    src = mne.read_source_spaces('/mnt/storage/Subjects/' + subj + '/bem/' + subj + '-ico-5-src.fif')
    trans = mne.read_trans('/mnt/storage/Subjects/' + subj + '/' + subj +'-trans.fif')
    bem = mne.read_bem_solution('/mnt/storage/Subjects/' + subj + '/bem/' + subj + '-5120-5120-5120-bem-sol.fif')

    fwd = mne.make_forward_solution(raw.info, trans, src, bem, eeg = False)

    mne.write_forward_solution('/mnt/storage/ATP/'+subj+'/sss_fif/'+subj+'_fwd.fif', fwd)
    return fwd

