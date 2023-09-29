#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 12:31:26 2023
Modified from czhao, still working
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

#%%####################################### Load baby data
root_path='/media/tzcheng/storage/CBS/'
subjects_dir = '/media/tzcheng/storage2/subjects/'

fname_aseg = subjects_dir + 'ANTS15-0Months3T/mri/aparc+aseg.mgz'
label_names = np.asarray(mne.get_volume_labels_from_aseg(fname_aseg))

lh_ROI_label = [61,63] # STG and frontal pole
rh_ROI_label = [96,98] # STG and IFG (parsopercularis, parsorbitalis, parstriangularis)

mmr1 = np.load('cbsb_meg_analysis/group_mmr1_vector_roi.npy',allow_pickle=True)
mmr2 = np.load('cbsb_meg_analysis/group_mmr2_vector_roi.npy',allow_pickle=True)

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

#%%
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
import mne
from mne.decoding import cross_val_multiscore, LinearModel, SlidingEstimator, get_coef, Vectorizer, Scaler
#Retrieve source space data into an array
X1= np.load('/mnt/storage/ATP/group/ba_diff.npy')
X2= np.load('/mnt/storage/ATP/group/ga_diff.npy')
X=np.concatenate((X1,X2),axis=0)
y = np.concatenate((np.repeat(0,20),np.repeat(1,20))) #0 is for ba and 1 is for ga
# prepare a series of classifier applied at each time sample
clf = make_pipeline(
    StandardScaler(),  # z-score normalization
    SelectKBest(f_classif, k=500),  # select features for speed
    LinearModel(),
    )
time_decod = SlidingEstimator(clf, scoring="roc_auc")
# Run cross-validated decoding analyses:
scores_observed = cross_val_multiscore(time_decod, X, y, cv=5 , n_jobs=None)
#Plot average decoding scores of 5 splits
times = np.linspace(-100,800,num=901)
fig, ax = plt.subplots(1)
ax.plot(times, scores_observed.mean(0), label="score")
ax.axhline(0.5, color="k", linestyle="--", label="chance")
ax.axvline(0, color="k")
plt.legend()
# The fitting needs not be cross validated because the weights are based on
# the training sets
time_decod.fit(X, y)
# Retrieve patterns after inversing the z-score normalization step:
patterns = get_coef(time_decod, "patterns_", inverse_transform=True)

#%% create a permutation of scores
# prepare a series of classifier applied at each time sample
import copy
import random
n_perm=100
scores_perm=[]
for i in range(n_perm):
    yp = copy.deepcopy(y)
    random.shuffle(yp)
    clf = make_pipeline(
        StandardScaler(),  # z-score normalization
        SelectKBest(f_classif, k=500),  # select features for speed
        LinearModel(LogisticRegression(C=1, solver="liblinear")),
        )
    time_decod = SlidingEstimator(clf, scoring="roc_auc")
    # Run cross-validated decoding analyses:
    scores = cross_val_multiscore(time_decod, X, yp, cv=5, n_jobs=None)
    scores_perm.append(np.mean(scores,axis=0))
scores_perm_array=np.asarray(scores_perm)

#%% train a classification model
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
clf=SVC(kernel='linear',C=1.0)
predicted=cross_val_predict(clf,X,Y,cv=2)
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score
C=confusion_matrix(Y,predicted)
A=accuracy_score(Y,predicted)
P=precision_score(Y,predicted)
R=recall_score(Y,predicted)
#%% build a distribution with shuffled Y
n_perm=1000
R2=[]
for i in range(n_perm):
    np.random.shuffle(Y)
    clf=SVR(kernel='linear',C=1)
    predicted=cross_val_predict(clf,X,Y,cv=2)
    R2.append(r2_score(Y,predicted))
plt.figure(figsize=(10,10))
plt.hist(R2,bins=30,color='r')
plt.vlines(0.305,ymin=0,ymax=100,color='k',linewidth=3)
plt.vlines(np.percentile(R2,97.5),ymin=0,ymax=100,color='r',linewidth=3)
plt.ylabel('Count',fontsize=20)
plt.xlabel('R2 score bins',fontsize=20)
#%% build distribution
n_perm=1000
Adis=[]
for i in range(n_perm):
    np.random.shuffle(Y)
    clf=SVC(kernel='linear',C=1.0)
    predicted=cross_val_predict(clf,X,Y,cv=2)
    a=accuracy_score(Y,predicted)
    Adis.append(a)
plt.figure(figsize=(10,10))
plt.hist(Adis,bins=10,color='b')
plt.vlines(A,ymin=0,ymax=300,color='k',linewidth=3)
plt.vlines(np.percentile(Adis,97.5),ymin=0,ymax=300,color='r',linewidth=3)
plt.ylabel('Count',fontsize=20)
plt.xlabel('R2 score bins',fontsize=20)