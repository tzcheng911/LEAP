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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
import time
import random


import mne
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne.preprocessing import Xdawn
from mne.decoding import (
    SlidingEstimator,
    GeneralizingEstimator,
    Scaler,
    cross_val_multiscore,
    LinearModel,
    get_coef,
    Vectorizer,
    CSP,
)


#%%####################################### decoding for single channel EEG
## Could apply to MMR or FFR, just load different group file
root_path='/media/tzcheng/storage2/CBS/'
# times = np.linspace(-0.02,0.2,1101)
# times = np.linspace(-0.1,0.6,3501)

ts = 500
te = 1750

## 1st run
std = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_std_cabr_eeg_200.npy')
dev1 = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_dev1_cabr_eeg_200.npy')
dev2 = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_dev2_cabr_eeg_200.npy')
std = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_std_eeg.npy')
dev1 = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_dev1_eeg.npy')
dev2 = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_dev2_eeg.npy')

## 2nd run
std = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_02_std_cabr_eeg_all.npy')
dev1 = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_02_dev1_cabr_eeg_all.npy')
dev2 = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_02_dev2_cabr_eeg_all.npy')
std = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_02_std_eeg.npy')
dev1 = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_02_dev1_eeg.npy')
dev2 = np.load(root_path + 'cbsA_meeg_analysis/EEG/' + 'group_02_dev2_eeg.npy')

MMR1 = dev1 - std
MMR2 = dev2 - std

#%%
X = np.concatenate((MMR1,MMR2),axis=0)
X = X[:,ts:te]
y = np.concatenate((np.repeat(0,len(dev1)),np.repeat(1,len(dev2))))

# X = np.concatenate((std,dev1,dev2),axis=0)
# y = np.concatenate((np.repeat(0,len(std)),np.repeat(1,len(dev1)),np.repeat(2,len(dev2))))

## randomization
rand_ind = np.arange(0,len(X))
random.shuffle(rand_ind)
X = X[rand_ind,:]
y = y[rand_ind]

clf = make_pipeline(
    StandardScaler(),  # z-score normalization
    LogisticRegression(solver="liblinear")  # liblinear is faster than lbfgs
)
scores = cross_val_multiscore(clf, X, y, cv=5, n_jobs=None) # takes about 10 mins to run
score = np.mean(scores, axis=0)
print("Accuracy: %0.1f%%" % (100 * score,))

#%%# if preserve the subject MMR1 and MMR2 relationship but randomize the order within each group
rand_ind = np.arange(0,len(MMR1))
random.shuffle(rand_ind)
X = np.concatenate((MMR1[rand_ind,:],MMR2[rand_ind,:]),axis=0)
X = X[:,ts:te]
y = np.concatenate((np.repeat(0,len(dev1)),np.repeat(1,len(dev2))))

clf = make_pipeline(
    StandardScaler(),  # z-score normalization
    LogisticRegression(solver="liblinear")  # liblinear is faster than lbfgs
)
scores = cross_val_multiscore(clf, X, y, cv=5, n_jobs=None) # takes about 10 mins to run
score = np.mean(scores, axis=0)
print("Accuracy: %0.1f%%" % (100 * score,))

#%%#######################################Run permutation
import copy
import random
n_perm=500
scores_perm=[]
for i in range(n_perm):
    yp = copy.deepcopy(y)
    random.shuffle(yp)
    clf = make_pipeline(
        StandardScaler(),  # z-score normalization
        LogisticRegression(solver="liblinear")  # liblinear is faster than lbfgs
        )
    # Run cross-validated decoding analyses:
    scores = cross_val_multiscore(clf, X, yp, cv=5, n_jobs=None)
    scores_perm.append(np.mean(scores,axis=0))
    print("Iteration " + str(i))
scores_perm_array=np.asarray(scores_perm)

plt.figure()
plt.hist(scores_perm_array,bins=30,color='k')
plt.vlines(score,ymin=0,ymax=12,color='r',linewidth=2)
plt.vlines(np.percentile(scores_perm_array,97.5),ymin=0,ymax=12,color='grey',linewidth=2)
plt.ylabel('Count',fontsize=20)
plt.xlabel('Accuracy',fontsize=20)
plt.title('FFR accuracy compared to 97.5 percentile of n = 500 null distribution')

plt.figure()
plt.subplot(311)
plt.plot(times,std.mean(axis=0))
plt.title('ntrial = 200, Accuracy = 0.82')
plt.xlim([-0.01,0.2])
plt.legend(['ba'])

plt.subplot(312)
plt.plot(times,dev1.mean(axis=0))
plt.xlim([-0.01,0.2])
plt.ylabel('Amplitude')
plt.legend(['mba'])

plt.subplot(313)
plt.plot(times,dev2.mean(axis=0))
plt.xlim([-0.01,0.2])
plt.xlabel('Time (s)')
plt.legend(['pa'])


#%%####################################### Subject-by-subject decoding for each condition 
tic = time.time()
root_path='/media/tzcheng/storage2/CBS/'
subjects_dir = '/media/tzcheng/storage2/subjects/'
os.chdir(root_path)

stc1 = mne.read_source_estimate(root_path + 'cbs_b101/sss_fif/cbs_b101_mmr2_vector_morph-vl.stc')
times = stc1.times

## parameters
ts = 250 # -0.05s
te = 2750 # 0.45s
ROI_wholebrain = 'wholebrain' # ROI or wholebrain or sensor
k_feature = 'all' # ROI: 'all' features; whole brain: 500 features

#%%####################################### Load adults
filename = 'vector'
filename_mmr1 = 'group_mmr1_vector_morph_mmr-cov'
filename_mmr2 = 'group_mmr2_vector_morph_mmr-cov'

fname_aseg = subjects_dir + 'fsaverage/mri/aparc+aseg.mgz'
label_names = np.asarray(mne.get_volume_labels_from_aseg(fname_aseg))
lh_ROI_label = [72,60,61,62] # Simport random
rh_ROI_label = [108,96,97,98] # STG and IFG (parsopercularis, parsorbitalis, parstriangularis)

if ROI_wholebrain == 'ROI':
    mmr1 = np.load(root_path + 'cbsA_meeg_analysis/MEG/vector_method/' + filename_mmr1 + '_roi.npy',allow_pickle=True)
    mmr2 = np.load(root_path + 'cbsA_meeg_analysis/MEG/vector_method/' + filename_mmr2 + '_roi.npy',allow_pickle=True)
elif ROI_wholebrain == 'wholebrain':
    mmr1 = np.load(root_path + 'cbsA_meeg_analysis/MEG/vector_method/' + filename_mmr1 + '.npy',allow_pickle=True)
    mmr2 = np.load(root_path + 'cbsA_meeg_analysis/MEG/vector_method/' + filename_mmr2 + '.npy',allow_pickle=True)
else:
    print("Need to decide whether to use ROI or whole brain as feature.")
X = np.concatenate((mmr1,mmr2),axis=0)
X = X[:,:,ts:te] 
y = np.concatenate((np.repeat(0,len(mmr1)),np.repeat(1,len(mmr1)))) #0 is for mmr1 and 1 is for mmr2

# prepare a series of classifier applied at each time sample
clf = make_pipeline(
    StandardScaler(),  # z-score normalization
    SelectKBest(f_classif, k=k_feature),  # select features for speed
    LinearModel(),
    )
time_decod = SlidingEstimator(clf, scoring="roc_auc")

# Run cross-validated decoding analyses
scores_observed = cross_val_multiscore(time_decod, X, y, cv=18, n_jobs=None) # leave one out
score = np.mean(scores_observed, axis=0)

#Plot average decoding scores of 5 splits
TOI = np.linspace(0,450,num=2250)
fig, ax = plt.subplots(1)
ax.plot(TOI, scores_observed.mean(0), label="score")
ax.axhline(0.5, color="k", linestyle="--", label="chance")
ax.axvline(0, color="k")
plt.legend()

# The fitting needs not be cross validated because the weights are based on
# the training sets
time_decod.fit(X, y) # not changed after shuffling the initial
# Retrieve patterns after inversing the z-score normalization step:
patterns = get_coef(time_decod, "patterns_", inverse_transform=True)

toc = time.time()
print('It takes ' + str((toc - tic)/60) + 'min to run decoding')

# np.save(root_path + 'cbsA_meeg_analysis/decoding/roc_auc_kall_' + filename + '.npy',scores_observed)
# np.save(root_path + 'cbsA_meeg_analysis/decoding/patterns_kall_' + filename + '.npy',patterns)

#%%####################################### Load babies
# fname_aseg = subjects_dir + 'ANTS15-0Months3T/mri/aparc+aseg.mgz'
# label_names = np.asarray(mne.get_volume_labels_from_aseg(fname_aseg))
# lh_ROI_label = [61,63] # STG and frontal pole
# rh_ROI_label = [96,98] # STG and IFG (parsopercularis, parsorbitalis, parstriangularis)

# if ROI_wholebrain == 'ROI':
#     mmr1 = np.load(root_path + 'cbsb_meeg_analysis/'  + filename_mmr1 + '_roi.npy',allow_pickle=True)
#     mmr2 = np.load(root_path + 'cbsb_meeg_analysis/' + filename_mmr2 + '_roi.npy',allow_pickle=True)
# elif ROI_wholebrain == 'wholebrain':
#     mmr1 = np.load(root_path + 'cbsb_meg_analysis/'  + filename_mmr1 + '.npy',allow_pickle=True)
#     mmr2 = np.load(root_path + 'cbsb_meg_analysis/'  + filename_mmr2 + '.npy',allow_pickle=True)
# else:
#     print("Need to decide whether to use ROI or whole brain as feature.")

# X=np.concatenate((mmr1,mmr2),axis=0)
# y = np.concatenate((np.repeat(0,len(mmr1)),np.repeat(1,len(mmr2)))) #0 is for ba and 1 is for ga

# # prepare a series of classifier applied at each time sample
# clf = make_pipeline(
#     StandardScaler(),  # z-score normalization
#     SelectKBest(f_classif, k=k_feature),  # select features for speed
#     LinearModel(),
#     )
# time_decod = SlidingEstimator(clf, scoring="roc_auc")

# # Run cross-validated decoding analyses
# scores_observed = cross_val_multiscore(time_decod, X, y, cv=5 , n_jobs=None)
# score = np.mean(scores_observed, axis=0)
# #Plot average decoding scores of 5 splits
# fig, ax = plt.subplots(1)
# ax.plot(stc1.times, scores_observed.mean(0), label="score")
# ax.axhline(0.5, color="k", linestyle="--", label="chance")
# ax.axvline(0, color="k")
# plt.xlim([-0.1, 0.6])
# plt.legend()

#%% create a permutation of scores
# prepare a series of classifier applied at each time sample
tic = time.time()
import copy
import random
n_perm=100
scores_perm=[]
for i in range(n_perm):
    print('Iteration' + str(i))
    yp = copy.deepcopy(y)
    random.shuffle(yp)
    clf = make_pipeline(
        StandardScaler(),  # z-score normalization
        SelectKBest(f_classif, k=k_feature),  # select features for speed
        LinearModel(),
        )
    time_decod = SlidingEstimator(clf, scoring="roc_auc")
    # Run cross-validated decoding analyses:
    scores = cross_val_multiscore(time_decod, X, yp, cv=5, n_jobs=None)
    scores_perm.append(np.mean(scores,axis=0))
scores_perm_array=np.asarray(scores_perm)
np.save(root_path + 'cbsA_meeg_analysis/' + filename + '_scores_perm_array.npy',scores_perm_array)

toc = time.time()
print('It takes ' + str((toc - tic)/60) + 'min to run 100 iterations of kall 150 ms - 200 ms decoding')

plt.figure()
plt.hist(scores_perm_array,bins=30,color='k')
plt.vlines(score,ymin=0,ymax=12,color='r',linewidth=2)
plt.vlines(np.percentile(scores_perm_array,97.5),ymin=0,ymax=12,color='grey',linewidth=2)
plt.ylabel('Count',fontsize=20)
plt.xlabel('Accuracy',fontsize=20)
plt.title('Accuracy compared to 97.5 percentile of n = 100 null distribution')

#%%####################################### Investigate the weights
# The fitting needs not be cross validated because the weights are based on
# the training sets
time_decod.fit(X, y)

# Retrieve patterns after inversing the z-score normalization step:
patterns = get_coef(time_decod, "patterns_", inverse_transform=True)

stc1_crop = stc1.copy().crop(tmin=stc1.times[ts],tmax=stc1.times[te],include_tmax=False)
stc1_crop.data = patterns
src = mne.read_source_spaces(subjects_dir + 'fsaverage/bem/fsaverage-vol-5-src.fif')

# Plot patterns across sources
stc1_crop.plot(src,clim=dict(kind="value",pos_lims=[0,15,30]), subject='fsaverage', subjects_dir=subjects_dir)

stc1_crop.plot(src, subject='fsaverage', subjects_dir=subjects_dir)

# Plot patterns across time
TOI = np.linspace(0,450,num=2250)
fig, ax = plt.subplots(1)
ax.plot(TOI, patterns.transpose())
ax.axhline(0.5, color="k", linestyle="--", label="chance")
ax.axvline(0, color="k")
plt.legend()

# Plot acc across time
TOI = np.linspace(0,450,num=2250)
fig, ax = plt.subplots(1)
ax.plot(TOI, scores_observed.mean(0), label="score")
ax.axhline(0.5, color="k", linestyle="--", label="chance")
ax.axvline(0, color="k")
plt.legend()

#%%####################################### Investigate the feature number
scores_observed_k50 = np.load('roc_auc_vector_morph_k50_100_450_mba_pa.npy')
patterns_k50 = np.load('patterns_vector_morph_k50_100_450_mba_pa.npy')

scores_observed_k500 = np.load('roc_auc_vector_morph_k500_100_450_mba_pa.npy')
patterns_k500 = np.load('patterns_vector_morph_k500_100_450_mba_pa.npy')

scores_observed_kall = np.load('roc_auc_vector_morph_kall_100_450_mba_pa.npy')
patterns_kall = np.load('patterns_vector_morph_kall_100_450_mba_pa.npy')

fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3), ncols=3)
fig1 = ax1.imshow(patterns_k50,extent=[0,2250,0,14629],aspect='auto')
fig2 = ax2.imshow(patterns_k500,extent=[0,2250,0,14629],aspect='auto')
fig3 = ax3.imshow(patterns_kall,extent=[0,2250,0,14629],aspect='auto')

fig.colorbar(fig1, ax=ax1)
fig.colorbar(fig2, ax=ax2)
fig.colorbar(fig3, ax=ax3)
fig1.set_clim(-60,60)
fig2.set_clim(-60,60)
fig3.set_clim(-60,60)
ax1.set_title('k = 50')
ax2.set_title('k = 500')
ax3.set_title('k = all')

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

#%%####################################### Trial-by-trial decoding for each individual
subjects_dir = '/media/tzcheng/storage2/subjects/'
root_path='/media/tzcheng/storage/CBS/'
s = 'cbs_b112'
file_in = root_path + '/' + s + '/sss_fif/' + s

epochs = mne.read_epochs(file_in +'_01_otp_raw_sss_proj_fil50_mmr_e.fif')
epochs.crop(0,0.3)
epochs_d = epochs["Deviant1","Deviant2"]
X = epochs_d.get_data()  # MEG signals features: n_epochs, n_meg_channels, n_times make sure the first dimension is epoch
y = epochs_d.events[:, 2]  # target: standard, deviant1 and 2

## spatial-temporal features
# all sensors all time points (0 - 300 ms)
clf = make_pipeline(
    Scaler(epochs.info),
    Vectorizer(),
    LogisticRegression(solver="liblinear")  # liblinear is faster than lbfgs
)
scores = cross_val_multiscore(clf, X, y, cv=5, n_jobs=None) # takes about 10 mins to run
score = np.mean(scores, axis=0)
print("Spatio-temporal: %0.1f%%" % (100 * score,))

clf = make_pipeline(
    Scaler(epochs.info),
    Vectorizer(),
    SVC(kernel='rbf',gamma='auto')  
)
scores = cross_val_multiscore(clf, X, y, cv=5, n_jobs=None) # takes about 10 mins to run

# Mean scores across cross-validation splits
score = np.mean(scores, axis=0)
print("Spatio-temporal: %0.1f%%" % (100 * score,))

## SlidingEstimator
clf = make_pipeline(
    StandardScaler(), LinearModel(LogisticRegression(solver="liblinear"))
)
time_decod = SlidingEstimator(clf, n_jobs=None, scoring="roc_auc", verbose=True)
scores = cross_val_multiscore(time_decod, X, y, cv=5, n_jobs=None)
time_decod.fit(X, y)

coef = get_coef(time_decod, "patterns_", inverse_transform=True)
evoked_time_gen = mne.EvokedArray(coef, epochs.info, tmin=epochs.times[0])
joint_kwargs = dict(ts_args=dict(time_unit="s"), topomap_args=dict(time_unit="s"))
evoked_time_gen.plot_joint(
    times=np.arange(0.100, 0.300, 0.05), title="patterns", **joint_kwargs
)

# Plot decoding accuracy for each time instance
fig, ax = plt.subplots()
ax.plot(epochs.times, np.mean(scores, axis=0), label="score")
ax.axhline(0.5, color="k", linestyle="--", label="chance")
ax.set_xlabel("Times")
ax.set_ylabel("AUC")  # Area Under the Curve
ax.legend()
ax.axvline(0.0, color="k", linestyle="-")
ax.set_title("Sensor space decoding")

# Projecting sensor-space patterns to source space
cov = mne.read_cov(file_in + '_01_erm_otp_raw_sss_proj_fil50-cov.fif')
fwd = mne.read_forward_solution(file_in + '-fwd.fif')
inv = mne.minimum_norm.make_inverse_operator(evoked_time_gen.info, fwd, cov, loose=1,depth=0.8)
src = inv['src']
stc = mne.minimum_norm.apply_inverse(evoked_time_gen, inv, 1.0 / 9.0, "dSPM")
brain = stc.plot(src=src, subjects_dir=subjects_dir
)

## X-Dawn (doesn't work yet)
# n_filter = 3
# # Create classification pipeline
# clf = make_pipeline(
#     Xdawn(n_components=n_filter),
#     Vectorizer(),
#     MinMaxScaler(),
#     LogisticRegression(solver="liblinear")
# )

# # Get the labels
# labels = epochs.events[:, -1]

# # Cross validator
# cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# # Do cross-validation
# preds = np.empty(len(labels))
# for train, test in cv.split(epochs, labels):
#     clf.fit(epochs[train], labels[train])
#     preds[test] = clf.predict(epochs[test])

# # Classification report
# target_names = ["Standard", "Deviant1", "Deviant2"]
# report = classification_report(labels, preds, target_names=target_names)
# print(report)

# # Normalized confusion matrixt last
# cm = confusion_matrix(labels, preds)
# cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

# # Plot confusion matrix
# fig, ax = plt.subplots(1, layout="constrained")
# im = ax.imshow(cm_normalized, interpolation="nearest", cmap=plt.cm.Blues)
# ax.set(title="Normalized Confusion matrix")
# fig.colorbar(im)
# tick_marks = np.arange(len(target_names))
# plt.xticks(tick_marks, target_names, rotation=45)
# plt.yticks(tick_marks, target_names)
# ax.set(ylabel="True label", xlabel="Predicted label")

