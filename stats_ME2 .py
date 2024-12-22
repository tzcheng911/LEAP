#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 09:54:23 2024

Run statistical analysis on the output from analysis_ME2.py 
Input: .npy files in the "analyzed data" i.e. SSEP, ERSP, decoding, connectivity folders
 
@author: tzcheng
"""
#%%####################################### Import library  
import statsmodels.api as sm
import pandas as pd
import numpy as np
import time
import random
import copy
import scipy.stats as stats
from scipy import stats,signal
from scipy.io import wavfile
import mne
from mne import spatial_src_adjacency
from mne.stats import spatio_temporal_cluster_1samp_test, summarize_clusters_stc
from mne.time_frequency import tfr_morlet, tfr_multitaper, tfr_stockwell, AverageTFRArray
from mne_connectivity import spectral_connectivity_epochs, spectral_connectivity_time,read_connectivity
from mne_connectivity.viz import plot_connectivity_circle
from mne.viz import circular_layout
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
import sklearn 
from sklearn.decomposition import PCA, FastICA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
import matplotlib.pyplot as plt 

#%%####################################### Define functions
def stats_SSEP(psds1,psds2,freqs,nonparametric):
    X = psds1-psds2
    if nonparametric: 
        T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(X, seed = 0,verbose='ERROR')
        good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
        for i in np.arange(0,len(good_cluster_inds),1):
            print("The " + str(i+1) + "st significant cluster")
            print(clusters[good_cluster_inds[i]])
            print('Significant freqs: ' + str(freqs[clusters[good_cluster_inds[i]]]))
    else:   
        t,p = stats.ttest_1samp(X[:,[6,7]].mean(axis=1),0) # duple vs. random in beat 3.3 Hz, meter 1.67 Hz
        # print('Testing freqs: ' + str(freqs[[6,7]]))
        # print('t statistics: ' + str(t))
        print('p-value: ' + str(p))
        t,p = stats.ttest_1samp(X[:,[12,13]].mean(axis=1),0) # duple vs. random in beat 3.3 Hz, meter 1.67 Hz
        # print('Testing freqs: ' + str(freqs[[12,13]]))
        # print('t statistics: ' + str(t))
        print('p-value: ' + str(p))
        t,p = stats.ttest_1samp(X[:,[30,31]].mean(axis=1),0) # duple vs. random in beat 3.3 Hz, meter 1.67 Hz
        # print('Testing freqs: ' + str(freqs[[30,31]]))
        # print('t statistics: ' + str(t))
        print('p-value: ' + str(p))
# def stats_Conn(power1,power2,times,freqs,nonparametric):

# def stats_age_effect():
    
   
     

        
#%%####################################### Set path
root_path = '/media/tzcheng/storage/ME2_MEG/Zoe_analyses/me2_meg_analysis/'
subjects_dir = '/media/tzcheng/storage2/subjects/'

#%%####################################### Load the audio files
fs, audio = wavfile.read(root_path + 'Stimuli/Duple300.wav') # Random, Duple300, Triple300
# plot_audio(audio,fmin=0.5,fmax=5,fs=fs)

#%% Parameters
age = ['7mo','11mo','br'] 
folders = ['SSEP/','ERSP/','decoding/','connectivity/'] # random, duple, triple
analysis = ['psds','bc_percent_power','decoding_acc_perm100','conn_plv','conn_coh','conn_pli']
which_data_type = ['_sensor_','_roi_','_roi_redo_','_morph_'] ## currently not able to run ERSP and conn on the wholebrain data

#%%####################################### Analysis on the sensor level 
data_type = which_data_type[0]
n_analysis = analysis[0]
n_folder = folders[0]
lm_np_duple = []
lm_np_triple = []

for n_age in age:
    print("Doing age " + n_age)
    random = np.load(root_path + n_folder + n_age + '_group_02_rs_mag6pT' + data_type + n_analysis +'.npz') 
    duple = np.load(root_path + n_folder + n_age + '_group_03_rs_mag6pT' + data_type + n_analysis + '.npz') 
    triple = np.load(root_path + n_folder + n_age + '_group_04_rs_mag6pT' + data_type + n_analysis + '.npz') 
    
    analysis_type = random.files[0]
    freqs = random[random.files[1]]
    random = random[random.files[0]]
    duple = duple[duple.files[0]]
    triple = triple[triple.files[0]]
    psds_random = random.mean(axis = 1)
    psds_duple = duple.mean(axis = 1)
    psds_triple = triple.mean(axis = 1)
    print("-------------------Doing duple-------------------")
    stats_SSEP(psds_duple,psds_random,freqs,nonparametric=False)
    print("-------------------Doing triple-------------------")
    stats_SSEP(psds_triple,psds_random,freqs,nonparametric=False)
    X_duple = psds_duple-psds_random
    X_triple = psds_triple-psds_random
    SSEP_duple = np.vstack((X_duple[:,[6,7]].mean(axis=1),X_duple[:,[12,13]].mean(axis=1),X_duple[:,[30,31]].mean(axis=1))).transpose()
    SSEP_triple = np.vstack((X_triple[:,[6,7]].mean(axis=1),X_triple[:,[12,13]].mean(axis=1),X_triple[:,[30,31]].mean(axis=1))).transpose()
    lm_np_duple.append(SSEP_duple)
    lm_np_triple.append(SSEP_triple)

age_group = np.concatenate((np.repeat('7mo',len(lm_np_duple[0])),np.repeat('11mo',len(lm_np_duple[1])),np.repeat('adult',len(lm_np_duple[2]))))
lm_df = pd.DataFrame({'age':age_group,'duple_1.67Hz':np.concatenate(lm_np_duple)[:,1],
              'duple_3.3Hz':np.concatenate(lm_np_duple)[:,2],'triple_1.1Hz':np.concatenate(lm_np_duple)[:,0],
              'triple_3.3Hz':np.concatenate(lm_np_duple)[:,2]})
age_dummies = pd.get_dummies(lm_df['age'], prefix='age', drop_first=True)
# lm_df = pd.concat([lm_df,age_dummies],axis=1)
X = sm.add_constant(age_dummies)
model = sm.OLS(lm_df['duple_1.67Hz'].astype(float),X.astype(float)).fit()

#%%####################################### Analysis on the source level: ROI 
data_type = which_data_type[2]
n_analysis = analysis[2]
n_folder = folders[2]
vmin = -1
vmax = 1
nlines = 10
FOI = 'beta'


fname_aseg = subjects_dir + 'fsaverage/mri/aparc+aseg.mgz'
if data_type == '_roi_':
    label_names = np.asarray(mne.get_volume_labels_from_aseg(fname_aseg))
    nROI = [72,108,66,102,64,100,59,95,7,8,26,27,60,61,62,96,97,98,50,86,71,107] 
elif data_type == '_roi_redo_':
    # label_names = np.asarray(["AuditoryL", "AuditoryR", "MotorL", "MotorR", "SensoryL", "SensoryR", "BGL", "BGR", "IFGL", "IFGR"])
    label_names = np.asarray(["Auditory", "Motor", "Sensory", "BG", "IFG"])
    nROI = np.arange(0,len(label_names),1)

# Auditory (STG 72,108, HG 76,112), Motor (precentral 66 102), Sensorimotor (postcentral 64 100), and between them is paracentral 59, 95
# Basal ganglia group (7,8,9,16,26,27,28,31): out of all include caudate (7 26) and putamen (8 27) only based on Cannon & Patel 2020 TICS, putamen is most relevant 
# Frontal IFG (60,61,62,96,97,98)
# Posterior Parietal: inferior parietal (50 86),  superior parietal (71 107)
# roi_redo pools ROIs to be 6 new_ROIs = {"Auditory": [72,108], "Motor": [66,102], "Sensory": [64,100], "BG": [7,8,26,27], "IFG": [60,61,62,96,97,98],  "Posterior": [50,86,71,107]}

for n_age in age:
    print("Doing age " + n_age)
    if n_folder == 'connectivity/':
        random = read_connectivity(root_path + n_folder + n_age + '_group_02_stc_rs_mne_mag6pT' + data_type + n_analysis) 
        duple = read_connectivity(root_path + n_folder + n_age + '_group_03_stc_rs_mne_mag6pT' + data_type + n_analysis) 
        triple = read_connectivity(root_path + n_folder + n_age + '_group_04_stc_rs_mne_mag6pT' + data_type + n_analysis) 
        freqs = random.freqs
        random_conn = random.get_data(output='dense')
        duple_conn = duple.get_data(output='dense')
        triple_conn = triple.get_data(output='dense')
 
        # print("-------------------Doing duple-------------------")
        # stats_CONN(duple_conn,random_conn,freqs,nonparametric=True)
        # print("-------------------Doing triple-------------------")
        # stats_CONN(triple_conn,random_conn,freqs,nonparametric=True)
    else:
        for n in nROI: 
            print("---------------------------------------------------Doing ROI: " + label_names[n])
            if n_folder == 'SSEP/':
                random0 = np.load(root_path + n_folder + n_age + '_group_02_stc_rs_mne_mag6pT' + data_type + n_analysis +'.npz') 
                duple0 = np.load(root_path + n_folder + n_age + '_group_03_stc_rs_mne_mag6pT' + data_type + n_analysis + '.npz') 
                triple0 = np.load(root_path + n_folder + n_age + '_group_04_stc_rs_mne_mag6pT' + data_type + n_analysis + '.npz') 
                random = random0[random0.files[0]]
                duple = duple0[duple0.files[0]]
                triple = triple0[triple0.files[0]]
                freqs = random0[random0.files[1]]          
                print("-------------------Doing duple-------------------")
                stats_SSEP(duple[:,n,:],random[:,n,:],freqs,nonparametric=True)
                print("-------------------Doing triple-------------------")
                stats_SSEP(triple[:,n,:],random[:,n,:],freqs,nonparametric=True)
          
            elif n_folder == 'decoding/':
                decoding = np.load(root_path + n_folder + n_age + '_' + n_analysis + '_roi_redo.npz') 
                all_score = decoding['all_score']
                scores_perm_array = decoding['scores_perm_array']
                ind = decoding['ind']

                
# np.save(root_path + 'figures/ERSP_sig_ROI_duple.npy',np.asarray(ERSP_sig_ROI_duple))
# np.save(root_path + 'figures/ERSP_sig_ROI_triple.npy',np.asarray(ERSP_sig_ROI_triple))
        
#%%####################################### Analysis on the source level: wholebrain 
n_age = age[2]
stc1 = mne.read_source_estimate('/media/tzcheng/storage/BabyRhythm/br_03/sss_fif/br_03_01_stc_mne_morph_mag6pT-vl.stc')
src = mne.read_source_spaces(subjects_dir + 'fsaverage/bem/fsaverage-vol-5-src.fif')

decoding = np.load(root_path + n_folder + n_age + '_' + n_analysis + '_morph.npz') 
all_score = decoding['all_score']
scores_perm_array = decoding['scores_perm_array']
ind = decoding['ind']
stc1.data=np.array([all_score,all_score]).transpose()
stc1.plot(src=src,clim=dict(kind="percent",lims=[95,97.5,99.975]))