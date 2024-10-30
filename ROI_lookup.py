#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 18:05:02 2024
Look up the ROI <-> vertices correspondance from fsaverage and aparc atals
@author: tzcheng
"""
import mne
import matplotlib.pyplot as plt 
import numpy as np
from scipy.stats import pearsonr
from scipy import stats
import pandas as pd

label = ["ctx-rh-superiortemporal"]
subject = 'fsaverage'
root_path='/media/tzcheng/storage2/CBS/'
subjects_dir = '/media/tzcheng/storage2/subjects/'
stc1 = mne.read_source_estimate(root_path + 'cbs_A101/sss_fif/cbs_A101_mmr2_morph-vl.stc')
src = mne.read_source_spaces(subjects_dir + subject + '/bem/fsaverage-vol-5-src.fif')

## Get the atlas labels 
fname_aseg = subjects_dir + subject + '/mri/aparc+aseg.mgz'
labels = mne.get_volume_labels_from_aseg(fname_aseg)

#%% create a dummy eye matrix to feed in as stc.data
dummy = np.eye(stc1.shape[0])
stc_dummy = stc1.copy()
stc_dummy.data = dummy
label_tc_dummy = mne.extract_label_time_course(stc_dummy,(fname_aseg,["ctx-rh-superiortemporal"]),src)
idx = np.where(label_tc_dummy[0]>0)

label_v_ind = []
for nlabel in np.arange(0,len(labels),1):
    label_tc_dummy = mne.extract_label_time_course(stc_dummy,(fname_aseg,labels[nlabel]),src)
    idx = np.where(label_tc_dummy[0]>0)
    label_v_ind.append(idx)
np.save('ROI_lookup.npy',np.array(label_v_ind, dtype=object),allow_pickle=True)
#%% Key in the vertex number from the stc.plot to see which ROI it's in, and check whether this location is relevant 
label_v_ind = np.load('/media/tzcheng/storage/scripts_zoe/ROI_lookup.npy', allow_pickle=True)

#%%
nv = 25172
v_ind = np.where(src[0]['vertno'] == nv)
for nlabel in np.arange(0,len(labels),1):
    if v_ind in label_v_ind[nlabel][0]:
        print("nv: " + str(nv), "idx: " + str(nlabel), "label: " + labels[nlabel])