#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 09:54:23 2024

Visualize the output from analysis_ME2.py 
Input: .npy files in the "analyzed data" i.e. SSEP, ERSP, decoding, connectivity folders
 
@author: tzcheng
"""
#%%####################################### Import library  
import numpy as np
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

import matplotlib.pyplot as plt 

#%%####################################### Define functions
def ff(input_arr,target): # find the idx of the closest freqeuncy in freqs
    delta = 1000000000
    idx = -1
    for i, val in enumerate(input_arr):
        if abs(input_arr[i]-target) < delta:
            idx = i
            delta = abs(input_arr[i]-target)
    return idx

def plot_err(group_stc,color,t):
    group_avg=np.mean(group_stc,axis=0)
    err=np.std(group_stc,axis=0)/np.sqrt(group_stc.shape[0])
    up=group_avg+err
    lw=group_avg-err
    plt.plot(t,group_avg,color=color)
    plt.fill_between(t,up,lw,color=color,alpha=0.5)

def plot_audio(audio,fmin,fmax,fs):
    plt.figure()
    plt.plot(np.linspace(0,audio.size/fs,audio.size),audio)

    psds, freqs = mne.time_frequency.psd_array_welch(
    audio,fs, 
    n_fft=len(audio),
    n_overlap=0,
    n_per_seg=None,
    fmin=fmin,
    fmax=fmax,)
    plt.figure()
    plt.plot(freqs,psds)
    plt.xlim([fmin,fmax])
    
def plot_SSEP(psds,freqs,title):
    plt.figure()
    plot_err(psds,'k',freqs)
    plt.xlim([freqs[0],freqs[-1]])
    plt.title(title)

def plot_CONN(conn,freqs,nlines,vmin,vmax,FOI,label_names,title):
    if FOI == "Theta": # 4-8 Hz
        X = conn[:,:,:,ff(freqs,4):ff(freqs,8)].mean(axis=3)
    elif FOI == "Alpha": # 8-12 Hz
        X = conn[:,:,:,ff(freqs,8):ff(freqs,12)].mean(axis=3)
    elif FOI == "Beta":  # 15-30 Hz
        X = conn[:,:,:,ff(freqs,15):ff(freqs,30)].mean(axis=3)
    else:  # broadband
        X = conn.mean(axis=3)
        
    # if FOI == "Theta": # 4-8 Hz
    #     X = conn[:,:,:,18:42].mean(axis=3)
    # elif FOI == "Alpha": # 8-12 Hz
    #     X = conn[:,:,:,42:65].mean(axis=3)
    # elif FOI == "Beta":  # 15-30 Hz
    #     X = conn[:,:,:,82:171].mean(axis=3)
    # else:  # broadband
    #     X = conn.mean(axis=3)
    # circular plot
    ROI_names = label_names
    labels = mne.read_labels_from_annot("sample", parc="aparc", subjects_dir=subjects_dir)
    label_colors = [label.color for label in labels]
    
    node_order = list()
    node_order.extend(ROI_names)  # reverse the order
    node_angles = circular_layout(
        ROI_names, node_order, start_pos=90, group_boundaries=[0, len(ROI_names) / 2])
    
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="black", subplot_kw=dict(polar=True))
    plot_connectivity_circle(
    X.mean(axis=0), # change to the freqs of interest
    ROI_names,
    n_lines=nlines, # plot the top n lines
    vmin=vmin, 
    vmax=vmax,
    node_angles=node_angles,
    node_colors=label_colors,
    title= title + " connectivity " + FOI,
    ax=ax)
    fig.tight_layout()
        
#%%####################################### Set path
root_path = '/media/tzcheng/storage/ME2_MEG/Zoe_analyses/me2_meg_analysis/'
subjects_dir = '/media/tzcheng/storage2/subjects/'

#%%####################################### Load the audio files
fs, audio = wavfile.read(root_path + 'Stimuli/Duple300.wav') # Random, Duple300, Triple300
plot_audio(audio,fmin=0.5,fmax=5,fs=fs)

#%% Parameters
age = ['7mo','11mo','br'] 
folders = ['SSEP/','ERSP/','decoding/','connectivity/'] # random, duple, triple
analysis = ['psds','bc_percent_power','decoding_acc_perm100','conn_plv','conn_coh','conn_pli']
which_data_type = ['_sensor_','_roi_','_roi_redo_','_morph_'] ## currently not able to run ERSP and conn on the wholebrain data

#%%####################################### Visualize the sensor level 
n_folder = folders[0]
n_analysis = analysis[0]
data_type = which_data_type[2]

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
    
    if analysis_type == 'psds':
        psds_random = random.mean(axis = 1)
        psds_duple = duple.mean(axis = 1)
        psds_triple = triple.mean(axis = 1)
        plot_SSEP(psds_random,freqs)
        plot_SSEP(psds_duple,freqs)
        plot_SSEP(psds_triple,freqs)
    else:
        print("Only ran SSEP analysis on the sensor level")

#%%####################################### Visualize on the source level: ROI 
n_folder = folders[3]
n_analysis = analysis[5]
data_type = which_data_type[2]

vmin = 0
vmax = 1
nlines = 10
FOI = 'Theta'

fname_aseg = subjects_dir + 'fsaverage/mri/aparc+aseg.mgz'
if data_type == '_roi_':
    label_names = np.asarray(mne.get_volume_labels_from_aseg(fname_aseg))
    nROI = [72,108,66,102,64,100,59,95,7,8,26,27,60,61,62,96,97,98,50,86,71,107] 
elif data_type == '_roi_redo_':
    label_names = np.asarray(["AuditoryL", "AuditoryR", "MotorL", "MotorR", "SensoryL", "SensoryR", "BGL", "BGR", "IFGL", "IFGR"])
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
        plot_CONN(random_conn,freqs,nlines,vmin,vmax, FOI,label_names,n_age + '_random_' + n_analysis)
        plot_CONN(duple_conn,freqs,nlines,vmin,vmax, FOI,label_names,n_age + '_duple_' + n_analysis)
        plot_CONN(triple_conn,freqs,nlines,vmin,vmax, FOI,label_names,n_age + '_triple_' + n_analysis)
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
                plot_SSEP(random[:,n,:],freqs,label_names[nROI[n]] + '_' + n_age + '_random')
                plot_SSEP(duple[:,n,:],freqs,label_names[nROI[n]] + '_' + n_age + '_duple')
                plot_SSEP(triple[:,n,:],freqs,label_names[nROI[n]] + '_' + n_age + '_triple')
            elif n_folder == 'decoding/':
                decoding = np.load(root_path + n_folder + n_age + '_' + n_analysis + '_morph.npz') 
                all_score = decoding['all_score']
                scores_perm_array = decoding['scores_perm_array']
                ind = decoding['ind']

#%%####################################### Visualize the source level: wholebrain 
n_age = age[0]
stc1 = mne.read_source_estimate('/media/tzcheng/storage/BabyRhythm/br_03/sss_fif/br_03_01_stc_mne_morph_mag6pT-vl.stc')
src = mne.read_source_spaces(subjects_dir + 'fsaverage/bem/fsaverage-vol-5-src.fif')

decoding = np.load(root_path + n_folder + n_age + '_wholebrain_decodingACC.npy') 
# all_score = decoding['all_score']
# scores_perm_array = decoding['scores_perm_array']
# ind = decoding['ind']
all_score = decoding
stc1.data=np.array([all_score,all_score]).transpose()
stc1.plot(src=src,clim=dict(kind="percent",lims=[95,97.5,99.975]))

all_score_all = np.zeros((len(subj),14629))
for ns,s in enumerate(subj):
    all_score_all[ns,:] = np.load(root_path + 'decoding/by_subjects/' + s + '_wholebrain_decodingACC_DT.npy')
    