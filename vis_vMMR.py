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
import numpy as np

## Visualize epochs
subjects_dir = '/media/tzcheng/storage2/subjects/'

s = 'vMMR_902'
run = '_1'

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

#standard.plot(subject='vMMR_902', subjects_dir=subjects_dir, hemi='both')
# deviant.plot(subject='vMMR_902', subjects_dir=subjects_dir, hemi='both')
## for publishable figures (still working on it, screen shot for now)
# brain.save_movie(time_dilation=20, tmin=0.05, tmax=0.16,
#                  interpolation='linear', framerate=10)
#brain = mmr.plot(subject=s, subjects_dir=subjects_dir, hemi='both', background="w",  clim =dict(kind="value",pos_lims= [0,9,19]))
#screenshot = brain.screenshot()

#%%########################################
## visualize signer (vMMR_901) vs. non-signer (vMMR_902) in a few ROIs
src = mne.setup_source_space(
    'sample', spacing="ico5", add_dist="patch", subjects_dir=subjects_dir)
src = mne.read_source_spaces('/media/tzcheng/storage2/subjects/' + s + '/bem/' + s +'-ico-5-src.fif') 
label = mne.read_labels_from_annot(subject=s, parc='aparc',subjects_dir='/media/tzcheng/storage2/subjects/')
label_tc=mne.extract_label_time_course(mmr,label,src,mode='mean',allow_empty=True)
np.save('vMMR_902_all_run1.npy',label_tc)
# mask1=get_atlas_roi_mask(mmr,roi='superiortemporal,temporalpole',atlas_subject='fsaverage')
for name in label:
    print(name.name)

# ROI based on Qi's request superior temporal, middle temporal, IFG-triangularis (pars triangularis), IFG-opercularis (pars opercularis), inferior parietal, lateraloccipital
Brain = mne.viz.get_brain_class()
brain = Brain(
    s,
    "both",
    "inflated",
    subjects_dir=subjects_dir,
    cortex="low_contrast",
    background="white",
    size=(800, 600),
)

brain.add_annotation("aparc")

# visualize the label
ROI_label = [0,66,36,40]
Qis_ROI_label = [1,67,23,37] # STS, transverse temporal, the occipital ROI, the IFG ROI (pars opercularis)

for nROI in ROI_label:
    brain.add_label(label[nROI], borders=False)

#%%# visualize signer (vMMR_901) vs. non-signer (vMMR_902) in a few hot spot
r4_901 = np.load('/media/tzcheng/storage/vmmr/figures/vMMR_901_all_run1.npy')
r4_902 = np.load('/media/tzcheng/storage/vmmr/figures/vMMR_902_all_run1.npy')
times = np.load('/media/tzcheng/storage/vmmr/figures/vmmr_times.npy')
label = np.load('/media/tzcheng/storage/vmmr/figures/aparc_label.npy',allow_pickle=True)

ROI_label = [0,66,36,40]

for i in np.arange(0,len(ROI_label),1):
    plt.figure()
    plt.plot(times,r4_901[ROI_label[i],])
    plt.plot(times,r4_902[ROI_label[i],])
    plt.title(label[ROI_label[i]].name)
    plt.legend(['signer','non-signer'])
    plt.xlabel('Time (s)')
    plt.ylabel('Activation (AU)')
    plt.xlim([-0.1, 0.5])
    plt.ylim([-10, 25])
    plt.savefig('/home/tzcheng/Desktop/' + 'run1_' + label[ROI_label[i]].name +'.pdf')
    