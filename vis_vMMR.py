#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 16:47:33 2023

@author: tzcheng
"""
## Import library  
import mne
import mnefun
import matplotlib.pyplot as plt

## Visualize epochs
subjects_dir = '/media/tzcheng/storage2/subjects/'

s = 'vMMR_902'
run = '_4'

root_path = '/media/tzcheng/storage/vmmr/'
fwd = mne.read_forward_solution(root_path + s + '/sss_fif/'+ s + '_fwd.fif')
cov = mne.read_cov(root_path + s + '/sss_fif/' + s + run + '_erm_raw_sss_proj_fil50-cov')
epoch = mne.read_epochs(root_path + s + '/sss_fif/' + s + run + '_raw_sss_proj_fil50_e.fif')

evoked_s = mne.read_evokeds(root_path + s + '/sss_fif/' + s + run + '_raw_sss_proj_fil50_evoked_s.fif')[0]
evoked_d = mne.read_evokeds(root_path + s + '/sss_fif/' + s + run + '_raw_sss_proj_fil50_evoked_d.fif')[0]

# mne.viz.plot_compare_evokeds(evoked_s, picks=["MEG0721"], combine="mean")
# mne.viz.plot_compare_evokeds(evoked_s, picks="meg", axes="topo") # plot all of them
# epoch.plot_sensors(kind='3d', ch_type='mag', ch_groups='position')
# chs = ["MEG0721","MEG0631","MEG0741","MEG1821"]
# mne.viz.plot_compare_evokeds(evoked_s, picks=chs, combine="mean", show_sensors="upper right")

inverse_operator = mne.minimum_norm.make_inverse_operator(epoch.info, fwd, cov,loose=0.2,depth=0.8)
standard = mne.minimum_norm.apply_inverse((evoked_s), inverse_operator)
deviant = mne.minimum_norm.apply_inverse((evoked_d), inverse_operator)

mmr = deviant - standard

standard.plot(subject='vMMR_902', subjects_dir=subjects_dir, hemi='both')
# deviant.plot(subject='vMMR_902', subjects_dir=subjects_dir, hemi='both')
# mmr.plot(subject='vMMR_902', subjects_dir=subjects_dir, hemi='both')

## visualize signer (vMMR_901) vs. non-signer (vMMR_902) in a few ROIs
src = mne.read_source_spaces('/media/tzcheng/storage2/subjects/' + s + '/bem/' + s +'-ico-5-src.fif') 
label = mne.read_labels_from_annot(subject=s, parc='HCPMMP1_combined',subjects_dir='/media/tzcheng/storage2/subjects/')
label_tc=mne.extract_label_time_course(mmr,label,src,mode='mean',allow_empty=True)

# mask1=get_atlas_roi_mask(mmr,roi='superiortemporal,temporalpole',atlas_subject='fsaverage')

for name in label:
    print(name.name)

Brain = mne.viz.get_brain_class()
brain = Brain(
    s,
    "lh",
    "inflated",
    subjects_dir=subjects_dir,
    cortex="low_contrast",
    background="white",
    size=(800, 600),
)

brain.add_annotation("aparc")

label_name = label[2]
brain.add_label(label_name, borders=False)