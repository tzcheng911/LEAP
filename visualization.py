#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Weds Feb 18 17:35:23 2026

Visualize the output from analysis.py 
Input: .npy files in the /data e.g. SSEP, connectivity
 
@author: tzcheng
"""
#%%####################################### Import library  
import numpy as np
from scipy.io import wavfile
import mne
from mne.stats import summarize_clusters_stc
from mne_connectivity import read_connectivity
from mne_connectivity.viz import plot_connectivity_circle
from mne.viz import circular_layout
import pickle
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

def plot_multiFreq_random_vs_rhythmic(
    psds_random,
    psds_rhythmic_list,
    FOIs,
    freq_labels,
    rhythmic_labels,
    colors_rhythmic,
    colors_random="tab:grey",
    ylabel="PSD",
    jitter=0.04,
    bar_width=0.32,
    figsize=(7, 4),
    save_path=None
):

    n_freq = len(FOIs)
    x_base = np.arange(n_freq)

    plt.figure(figsize=figsize)

    for i, (FOI, psds_rhythm) in enumerate(zip(FOIs, psds_rhythmic_list)):

        # ---- average over FOI ----
        R = psds_random[:, FOI].mean(axis=-1)
        Y = psds_rhythm[:, FOI].mean(axis=-1)

        means = [R.mean(), Y.mean()]
        ses = [
            R.std(ddof=1) / np.sqrt(len(R)),
            Y.std(ddof=1) / np.sqrt(len(Y))
        ]

        xR = x_base[i] - bar_width / 2
        xY = x_base[i] + bar_width / 2

        # ---- bars ----
        plt.bar(xR, means[0], width=bar_width,
                color=colors_random, alpha=0.6, zorder=1)

        plt.bar(xY, means[1], width=bar_width,
                color=colors_rhythmic[i], alpha=0.6, zorder=1)

        # ---- datapoints ----
        plt.scatter(
            np.random.normal(xR, jitter, len(R)),
            R, color="black", zorder=2
        )
        plt.scatter(
            np.random.normal(xY, jitter, len(Y)),
            Y, color="black", zorder=2
        )

        # ---- error bars ----
        plt.errorbar(
            [xR, xY],
            means,
            yerr=ses,
            fmt="none",
            ecolor="lightgrey",
            elinewidth=3,
            capsize=6,
            capthick=3,
            zorder=3
        )

    # ---- axes ----
    fontsize = 14
    plt.xticks(x_base, freq_labels, fontsize=fontsize)
    plt.yticks(fontsize=12)
    plt.ylabel(ylabel, fontsize=fontsize)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches="tight", transparent=True)
    plt.show()

    
def plot_audio(audio,fmin,fmax,fs):
    plt.figure()
    plt.plot(np.linspace(0,audio.size/fs,audio.size),audio)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

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
    plt.ylim([-0.01,2.5])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD')
    
def plot_CONN(conn,freqs,nlines,vmin,vmax,FOI,label_names,title):
    if FOI == "Theta": # 4-8 Hz
        X = conn[:,:,:,ff(freqs,4):ff(freqs,8)].mean(axis=3)
    elif FOI == "Alpha": # 8-12 Hz
        X = conn[:,:,:,ff(freqs,8):ff(freqs,12)].mean(axis=3)
    elif FOI == "Beta":  # 15-30 Hz
        X = conn[:,:,:,ff(freqs,15):ff(freqs,30)].mean(axis=3)
    else:  # broadband
        X = conn.mean(axis=3)
        

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
root_path = '/home/tzcheng/Desktop/ME2_upload_to_github/'
subjects_dir = '/home/tzcheng/Desktop/ME2_upload_to_github/subjects/'

#%%####################################### set up the template brain
stc1 = mne.read_source_estimate(subjects_dir + 'me2_101_7m_03_stc_mne_morph_mag6pT-vl.stc')
src = mne.read_source_spaces(subjects_dir + 'fsaverage-vol-5-src.fif')
label_v_ind = np.load(subjects_dir + 'ROI_lookup.npy', allow_pickle=True)
fname_aseg = subjects_dir + 'aparc+aseg.mgz'
labels_all = mne.get_volume_labels_from_aseg(fname_aseg)

#%% Parameters
conditions = ['_02','_03','_04'] # random, duple, triple
rhythms = ['random','duple','triple']
folders = ['SSEP/','connectivity/'] 
analysis = ['psds','conn_plv']
which_data_type = ['_sensor_','_roi_redo4_','_morph_'] 
label_names = np.asarray(["Auditory", "SensoryMotor", "BG"])

#%%####################################### Figure 1: Visualize the audio files
fs, audio = wavfile.read(root_path + 'stimuli/randomt15rr.wav') # Random, Duple300, Triple300
plot_audio(audio,fmin=0.5,fmax=5,fs=fs)
fs, audio = wavfile.read(root_path + 'stimuli/random15rr.wav') # Random, Duple300, Triple300
plot_audio(audio,fmin=0.5,fmax=5,fs=fs)
fs, audio = wavfile.read(root_path + 'stimuli/Duple300rr.wav') # Random, Duple300, Triple300
plot_audio(audio,fmin=0.5,fmax=5,fs=fs)
fs, audio = wavfile.read(root_path + 'stimuli/Triple300rr.wav') # Random, Duple300, Triple300
plot_audio(audio,fmin=0.5,fmax=5,fs=fs)

#%%####################################### Figure 2: Visualize the sensor level 
n_folder = folders[0]
n_analysis = analysis[0]
data_type = which_data_type[0]

random = np.load(root_path + 'data/SSEP/7mo_group_02_rs_mag6pT_sensor_psds.npz') 
duple = np.load(root_path + 'data/SSEP/7mo_group_03_rs_mag6pT_sensor_psds.npz') 
triple = np.load(root_path + 'data/SSEP/7mo_group_04_rs_mag6pT_sensor_psds.npz') 
freqs = random[random.files[1]]
random = random[random.files[0]]
duple = duple[duple.files[0]]
triple = triple[triple.files[0]]
        
psds_random = random.mean(axis = 1)
psds_duple = duple.mean(axis = 1)
psds_triple = triple.mean(axis = 1)

## Line plot for steady-state evoked potential
plt.figure()
plot_err(psds_random,'k',freqs)
plot_err(psds_duple,'#ff7f0e',freqs)
plot_err(psds_triple,'#1f77b4',freqs)
plt.xlim([freqs[0],freqs[-1]])
plt.ylim([0,4.25e-25])
plt.legend(["Radnom","","Duple","","Triple",""])

## Bar plot for the significant freqs based on non-parametric test (analysis.py)
# Random: always grey
color_random = "tab:grey"

## Beat frequency 3.33 Hz for duple (orange) and triple (blue)
FOIs = [
        [30, 31, 32],  # duple beat
        [30, 31],  # triple beat
    ]
colors_rhythmic = ["#ff7f0e","#1f77b4"]
plot_multiFreq_random_vs_rhythmic(
    psds_random,
    psds_rhythmic_list=[psds_duple,psds_triple],
    FOIs=FOIs,
    freq_labels=["3.33 Hz","3.33 Hz"],
    rhythmic_labels=["Duple","Triple"],
    colors_random=color_random,
    colors_rhythmic=colors_rhythmic,
    ylabel="PSD at beat rate")

## Meter frequency 1.1 Hz for triple meter (blue), 1.67 Hz for duple meter(orange), 2.22 Hz for triple 1st harmonic (blue)
FOIs = [
        [6, 7],    # triple meter
        [12, 13],  # duple meter
        [18, 19],  # triple 1st harmonic
    ]
colors_rhythmic = ["#1f77b4","#ff7f0e","#1f77b4"]
plot_multiFreq_random_vs_rhythmic(
    psds_random,
    psds_rhythmic_list=[psds_triple,psds_duple,psds_triple],
    FOIs=FOIs,
    freq_labels=["1.11 Hz","1.67 Hz","2.22 Hz"],
    rhythmic_labels=["Duple","Triple"],
    colors_random=color_random,
    colors_rhythmic=colors_rhythmic,
    ylabel="PSD at meter rate")

#%%####################################### Figure 3: Visualize on the source level -- ROI 
nROI = np.arange(0,len(label_names),1)

for n in nROI: 
    print("---------------------------------------------------Doing ROI: " + label_names[n])
    random0 = np.load(root_path + 'data/SSEP/7mo_group_02_stc_rs_mne_mag6pT_roi_redo4_psds.npz') 
    duple0 = np.load(root_path + 'data/SSEP/7mo_group_03_stc_rs_mne_mag6pT_roi_redo4_psds.npz') 
    triple0 = np.load(root_path + 'data/SSEP/7mo_group_04_stc_rs_mne_mag6pT_roi_redo4_psds.npz') 
    random = random0[random0.files[0]]
    duple = duple0[duple0.files[0]]
    triple = triple0[triple0.files[0]]
    freqs = random0[random0.files[1]]          
                
    ## Bar plot for the significant freqs based on non-parametric test (analysis.py)
    # Random: always grey
    color_random = "tab:grey"
    
    ## Beat frequency 3.33 Hz for duple (orange) and triple (blue)
    colors_rhythmic = ["#ff7f0e","#1f77b4"]
    ## beats
    FOIs = [
        [30, 31],  # duple beat
        [30, 31],  # triple beat
    ]

    plot_multiFreq_random_vs_rhythmic(
        psds_random,
        psds_rhythmic_list=[duple[:,n,:],triple[:,n,:]],
        FOIs=FOIs,
        freq_labels=["3.33 Hz","3.33 Hz"],
        rhythmic_labels=["Duple","Triple"],
        colors_random=color_random,
        colors_rhythmic=colors_rhythmic,
        ylabel="PSD at beat rate"
    )   
        
    ## Meter frequency 1.1 Hz for triple meter (blue), 1.67 Hz for duple meter(orange), 2.22 Hz for triple 1st harmonic (blue)
    FOIs = [
            [6, 7],    # triple meter
            [12, 13],  # duple meter
            [18, 19],  # triple 1st harmonic
        ]
    colors_rhythmic = ["#1f77b4","#ff7f0e","#1f77b4"] 
    plot_multiFreq_random_vs_rhythmic(
        psds_random,
        psds_rhythmic_list=[triple[:,n,:],duple[:,n,:],triple[:,n,:]],
        FOIs=FOIs,
        freq_labels=["1.11 Hz","1.67 Hz","2.22 Hz"],
        rhythmic_labels=["Duple","Triple"],
        colors_random=color_random,
        colors_rhythmic=colors_rhythmic,
        ylabel="PSD at meter rate")

#%%####################################### Figure 3: Visualize the source level -- wholebrain 
p_threshold = 0.05 # set a cluster forming threshold based on a p-value for the cluster based permutation test

for n_meter in rhythms[1:]:
    
    with open(root_path + 'data/SSEP/7mo_SSEP_wholebrain_cluster_test_' + n_meter + '.pkl', 'rb') as f:
        clu = pickle.load(f)
    good_cluster_inds = np.where(clu[2] < p_threshold)[0]
    good_clusters = [clu[1][idx] for idx in good_cluster_inds]
    
    ## visualize this cluster
    stc_all_cluster_vis = summarize_clusters_stc(
        clu, p_thresh = p_threshold, vertices=src, subject="fsaverage"
    )
    stc_all_cluster_vis.plot(src=src,clim=dict(kind="percent",lims=[99.7,99.75,99.975])) 
    # The first time point in this SourceEstimate object is the summation of all the clusters
    # Subsequent time points contain each individual cluster. The magnitude of the activity corresponds to the duration spanned (the freq in this case) by the cluster

#%%####################################### Figure 4: Visualize the ROI conn results 
## ROI order should be (ROI1, ROI2) = (1,0) or (2,1) based on how conn structure is stored
ymin = 0.39
ymax = 1

ROI1 = 1 # 1: SensoryMotor or 2: BG
ROI2 = 0 # 0: Auditory or 1: SensoryMotor
label_names = np.asarray(["Auditory", "SensoryMotor", "BG"])

random = read_connectivity(root_path + 'connectivity/7mo_group_02_stc_rs_mne_mag6pT_roi_redo4_conn_plv') 
duple = read_connectivity(root_path + 'connectivity/7mo_group_03_stc_rs_mne_mag6pT_roi_redo4_conn_plv') 
triple = read_connectivity(root_path + 'connectivity/7mo_group_04_stc_rs_mne_mag6pT_roi_redo4_conn_plv') 
freqs = np.array(random.freqs)
random_conn = random.get_data(output='dense')
duple_conn = duple.get_data(output='dense')
triple_conn = triple.get_data(output='dense')

plt.figure()
plot_err(random_conn[:,ROI1,ROI2,:],'k',freqs)
plot_err(duple_conn[:,ROI1,ROI2,:],'#ff7f0e',freqs)
plot_err(triple_conn[:,ROI1,ROI2,:],'#1f77b4',freqs)

plt.xlim([4,35])
plt.ylim([ymin,ymax])
fname = "PLV between " + label_names[ROI1] + " and " + label_names[ROI2]
plt.title(fname)

#%%####################################### Figure 5: The correlation results are visualized in analysis.py