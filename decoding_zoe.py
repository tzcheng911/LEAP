#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 12:31:26 2023

@author: tzcheng
"""
import os 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression

import mne
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne.decoding import cross_val_multiscore, LinearModel, SlidingEstimator, get_coef

#%%####################################### Load data
root_path='/media/tzcheng/storage/CBS/'
os.chdir(root_path)
subjects_dir = '/media/tzcheng/storage2/subjects/'


subj = [] 
for file in os.listdir():
    if file.startswith('cbs_A'):
        subj.append(file)

runs = ['_01','_02']
run = runs[0]
s = subj[0]
subject = s

file_in = root_path + s + '/sss_fif/' + s
fwd = mne.read_forward_solution(file_in + '-fwd.fif')
cov = mne.read_cov(file_in + run + '_erm_otp_raw_sss_proj_fil50-cov.fif')

epochs = mne.read_epochs(file_in + run + '_otp_raw_sss_proj_fil50_mmr_e.fif')
evoked_s = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_fil50_evoked_substd_mmr.fif')[0]
evoked_d1 = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_fil50_evoked_dev1_mmr.fif')[0]
evoked_d2 = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_fil50_evoked_dev2_mmr.fif')[0]

stcs = mne.read_source_estimate(file_in + '_mmr2_morph-vl.stc')
src = mne.read_source_spaces(subjects_dir + '/fsaverage/bem/fsaverage-vol-5-src.fif') # if morph, use fsaverage

#%%####################################### Decoding source activities 
# Example https://mne.tools/stable/auto_examples/decoding/decoding_spatio_temporal_source.html#ex-dec-st-source
# Retrieve source space data into an array
X = np.array([stc.lh_data for stc in stcs])  # only keep left hemisphere
y = epochs.events[:, 2]

# prepare a series of classifier applied at each time sample
clf = make_pipeline(
    StandardScaler(),  # z-score normalization
    SelectKBest(f_classif, k=500),  # select features for speed
    LinearModel(LogisticRegression(C=1, solver="liblinear")),
)
time_decod = SlidingEstimator(clf, scoring="roc_auc")

# Run cross-validated decoding analyses:
scores = cross_val_multiscore(time_decod, X, y, cv=5, n_jobs=None)

# Plot average decoding scores of 5 splits
fig, ax = plt.subplots(1)
ax.plot(epochs.times, scores.mean(0), label="score")
ax.axhline(0.5, color="k", linestyle="--", label="chance")
ax.axvline(0, color="k")
plt.legend()

#%%####################################### Investigate the weights
# The fitting needs not be cross validated because the weights are based on
# the training sets
time_decod.fit(X, y)

# Retrieve patterns after inversing the z-score normalization step:
patterns = get_coef(time_decod, "patterns_", inverse_transform=True)

stc = stcs[0]  # for convenience, lookup parameters from first stc
vertices = [stc.lh_vertno, np.array([], int)]  # empty array for right hemi
stc_feat = mne.SourceEstimate(
    np.abs(patterns),
    vertices=vertices,
    tmin=stc.tmin,
    tstep=stc.tstep,
    subject="sample",
)

brain = stc_feat.plot(
    views=["lat"],
    transparent=True,
    initial_time=0.1,
    time_unit="s",
    subjects_dir=subjects_dir,
)
