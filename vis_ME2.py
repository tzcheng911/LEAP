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
    """
    One plot: Random vs frequency-specific rhythmic conditions.

    Parameters
    ----------
    psds_random : ndarray (n_subjects, n_freqs)
    psds_rhythmic_list : list of ndarrays
        [psds_duple, psds_triple, psds_beat, ...]
    FOIs : list of lists
        Frequency indices per rhythmic condition
    freq_labels : list
        X-axis labels (e.g., meter rates)
    rhythmic_labels : list
        Labels for rhythmic conditions (for legend)
    """

    n_freq = len(FOIs)
    x_base = np.arange(n_freq)

    plt.figure(figsize=figsize)

    for i, (FOI, psds_rhythm) in enumerate(zip(FOIs, psds_rhythmic_list)):

        # ---- average over FOI ----
        R = psds_random[:, FOI].mean(axis=1)
        Y = psds_rhythm[:, FOI].mean(axis=1)

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
    plt.ylim([0,4.25e-25])
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

#%%####################################### set up the template brain
stc1 = mne.read_source_estimate('/media/tzcheng/storage/BabyRhythm/br_03/sss_fif/br_03_01_stc_mne_morph_mag6pT-vl.stc')
src = mne.read_source_spaces(subjects_dir + 'fsaverage/bem/fsaverage-vol-5-src.fif')
label_v_ind = np.load('/media/tzcheng/storage/scripts_zoe/ROI_lookup.npy', allow_pickle=True)
fname_aseg = subjects_dir + 'fsaverage/mri/aparc+aseg.mgz'
label_names = mne.get_volume_labels_from_aseg('/media/tzcheng/storage2/subjects/fsaverage/mri/aparc+aseg.mgz')

#%%####################################### Load the audio files
fs, audio = wavfile.read(root_path + 'Stimuli/Random/random15rr.wav') # Random, Duple300, Triple300
plot_audio(audio,fmin=0.5,fmax=5,fs=fs)

#%% Parameters
ages = ['7mo'] 
folders = ['SSEP/','decoding/','connectivity/'] # random, duple, triple
analysis = ['psds','decoding_acc_perm100','conn_plv','conn_coh','conn_pli']
which_data_type = ['_sensor_','_roi_','_roi_redo4_','_morph_'] ## currently not able to run ERSP and conn on the wholebrain data

#%%####################################### Visualize the sensor level 
n_folder = folders[0]
n_analysis = analysis[0]
data_type = which_data_type[0]

for n_age in ages:
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
    plot_SSEP(psds_random,freqs,'MEG_random_psds')
    plot_SSEP(psds_duple,freqs,'MEG_duple_psds')
    plot_SSEP(psds_triple,freqs,'MEG_triple_psds')
    
    ## plot bar plot for the significant freqs based on non-parametric test (stats_ME2.py)
    # Doing age 7mo
    # -------------------Doing duple-------------------
    # The 1st significant cluster
    # (array([12, 13]),)
    # Significant freqs: [1.63636364 1.72727273]
    # The 2st significant cluster
    # (array([30, 31, 32]),)
    # Significant freqs: [3.27272727 3.36363636 3.45454545]
    # -------------------Doing triple-------------------
    # The 1st significant cluster
    # (array([6, 7]),)
    # Significant freqs: [1.09090909 1.18181818]
    # The 2st significant cluster
    # (array([18, 19]),)
    # Significant freqs: [2.18181818 2.27272727]
    # The 3st significant cluster
    # (array([30, 31]),)
    # Significant freqs: [3.27272727 3.36363636]

    ## take the significant freqs to plug in FOI for each condition from above 
    ## duple meter [12, 13] and beat [30, 31, 32]
    
    #### plotting the results
    # Random: always grey
    color_random = "tab:grey"
    
    # Rhythmic: Duple = orange, Triple = blue
    # Map your rhythmic data order:
    colors_rhythmic = [
        "#1f77b4",
        "#ff7f0e",     # Duple #ff7f0e
        "#1f77b4"      # Triple #1f77b4
    ]

    ## beats
    # FOIs = [
    #     [30, 31, 32],  # duple beat
    #     [30, 31],  # triple beat
    # ]


    ## meters
    FOIs = [
        [6, 7],    # triple meter
        [12, 13],  # duple meter
        [18, 19],  # triple 1st harmonic
    ]
    
    plot_multiFreq_random_vs_rhythmic(
        psds_random,
        psds_rhythmic_list=[psds_triple,psds_duple,psds_triple],
        FOIs=FOIs,
        freq_labels=["1.11 Hz","1.67 Hz","2.22 Hz"],
        rhythmic_labels=["Duple","Triple"],
        colors_random=color_random,
        colors_rhythmic=colors_rhythmic,
        ylabel="PSD at meter rate",
        save_path="/home/tzcheng/Desktop/sensor_PSD_barplot_all_meter.pdf"
    )



#%%####################################### Visualize on the source level: ROI 
n_folder = folders[2]
n_analysis = analysis[2]
data_type = which_data_type[2]

vmin = 0.5
vmax = 1
nlines = 2
FOI = 'alpha'

fname_aseg = subjects_dir + 'fsaverage/mri/aparc+aseg.mgz'
if data_type == '_roi_':
    label_names = np.asarray(mne.get_volume_labels_from_aseg(fname_aseg))
    nROI = [72,108,66,102,64,100,59,95,7,8,26,27,60,61,62,96,97,98,50,86,71,107] 
elif data_type == '_roi_redo_':
    label_names = np.asarray(["AuditoryL", "AuditoryR", "MotorL", "MotorR", "SensoryL", "SensoryR", "BGL", "BGR", "IFGL", "IFGR"])
elif data_type == '_roi_redo5_':
    label_names = np.asarray(["Auditory", "Motor", "Sensory", "BG", "IFG"])
elif data_type == '_roi_redo4_':
    label_names = np.asarray(["Auditory", "SensoryMotor", "BG", "IFG"])

# Auditory (STG 72,108, HG 76,112), Motor (precentral 66 102), Sensorimotor (postcentral 64 100), and between them is paracentral 59, 95
# Basal ganglia group (7,8,9,16,26,27,28,31): out of all include caudate (7 26) and putamen (8 27) only based on Cannon & Patel 2020 TICS, putamen is most relevant 
# Frontal IFG (60,61,62,96,97,98)
# Posterior Parietal: inferior parietal (50 86),  superior parietal (71 107)
# roi_redo pools ROIs to be 6 new_ROIs = {"Auditory": [72,108], "Motor": [66,102], "Sensory": [64,100], "BG": [7,8,26,27], "IFG": [60,61,62,96,97,98],  "Posterior": [50,86,71,107]}

plt.figure()    
color1 = ['m','r']
color2 = ['c','b']
ROI1 = 66
ROI2 = 53
for nn_age,n_age in enumerate(ages):
    print("Doing age " + n_age)
    if n_folder == 'connectivity/':
        random = read_connectivity(root_path + n_folder + n_age + '_group_02_stc_rs_mne_mag6pT' + data_type + n_analysis) 
        duple = read_connectivity(root_path + n_folder + n_age + '_group_03_stc_rs_mne_mag6pT' + data_type + n_analysis) 
        triple = read_connectivity(root_path + n_folder + n_age + '_group_04_stc_rs_mne_mag6pT' + data_type + n_analysis) 
        freqs = np.array(random.freqs)
        random_conn = random.get_data(output='dense')
        duple_conn = duple.get_data(output='dense')
        triple_conn = triple.get_data(output='dense')
        
        plot_err(duple_conn[:,ROI1,ROI2,:]-random_conn[:,ROI1,ROI2,:],color1[nn_age],freqs)
        plot_err(triple_conn[:,ROI1,ROI2,:]-random_conn[:,ROI1,ROI2,:],color2[nn_age],freqs)
        
        plt.figure()
        plt.title(n_age)
        plot_err(duple_conn[:,ROI1,ROI2,:],'r',freqs)
        plot_err(triple_conn[:,ROI1,ROI2,:],'b',freqs)
        plot_err(random_conn[:,ROI1,ROI2,:],'k',freqs)
        plt.xlim([4,35])
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
                # plot_SSEP(random[:,n,:],freqs,label_names[nROI[n]] + '_' + n_age + '_random')
                # plot_SSEP(duple[:,n,:],freqs,label_names[nROI[n]] + '_' + n_age + '_duple')
                # plot_SSEP(triple[:,n,:],freqs,label_names[nROI[n]] + '_' + n_age + '_triple')
                
                # Doing ROI SSEP: Auditory
                # -------------------Doing duple-------------------
                # The 1st significant cluster
                # (array([12, 13]),)
                # Significant freqs: [1.63636364 1.72727273]
                # The 2st significant cluster
                # (array([29, 30, 31]),)
                # Significant freqs: [3.18181818 3.27272727 3.36363636]
                # The 3st significant cluster
                # (array([48, 49]),)
                # Significant freqs: [4.90909091 5.        ]
                # -------------------Doing triple-------------------
                # The 1st significant cluster
                # (array([30, 31]),)
                # Significant freqs: [3.27272727 3.36363636]
                
                # Doing ROI SSEP: SensoryMotor
                # -------------------Doing duple-------------------
                # The 1st significant cluster
                # (array([29, 30, 31]),)
                # Significant freqs: [3.18181818 3.27272727 3.36363636]
                # -------------------Doing triple-------------------
                # The 1st significant cluster
                # (array([29, 30, 31]),)
                # Significant freqs: [3.18181818 3.27272727 3.36363636]
                
                # Doing ROI SSEP: BG
                # -------------------Doing duple-------------------
                # The 1st significant cluster
                # (array([30, 31]),)
                # Significant freqs: [3.27272727 3.36363636]
                # -------------------Doing triple-------------------
                
                ## take the significant freqs to plug in FOI for each condition from above 
                ## duple meter [12, 13] and beat [30, 31, 32]
               
                color_random = "tab:grey"
                
                # Rhythmic: Duple = orange, Triple = blue
                # Map your rhythmic data order:
                colors_rhythmic = [
                    "#ff7f0e",     # Duple #ff7f0e
                    "#1f77b4"      # Triple #1f77b4
                ]
            
                ## beats
                FOIs = [
                    [30, 31],  # duple beat
                    [30, 31],  # triple beat
                ]
            
            
                ## meters
                # FOIs = [
                #     [6, 7],    # triple meter
                #     [12, 13],  # duple meter
                #     [18, 19],  # triple 1st harmonic
                # ]
                
                nROI = 1 # 0,1,2
                
                plot_multiFreq_random_vs_rhythmic(
                    psds_random,
                    psds_rhythmic_list=[duple[:,nROI,:],triple[:,nROI,:]],
                    FOIs=FOIs,
                    freq_labels=["3.33 Hz","3.33 Hz"],
                    rhythmic_labels=["Duple","Triple"],
                    colors_random=color_random,
                    colors_rhythmic=colors_rhythmic,
                    ylabel="PSD at beat rate",
                    save_path="/home/tzcheng/Desktop/ROI_SM_PSD_barplot_all_beat.pdf"
                )              
                
            elif n_folder == 'decoding/':
                decoding_duple = np.load(root_path + n_folder + n_age + data_type +'decodingACC_duple.npy') 
                print(decoding_duple[n])
                decoding_triple = np.load(root_path + n_folder + n_age + data_type +'decodingACC_triple.npy') 
                print(decoding_triple[n])

#%%####################################### Visualize the source level: wholebrain 
data_type = which_data_type[-1]
n_analysis = analysis[0]
n_folder = folders[0]
n_meter = 'duple' # 'duple' or 'triple'
p_threshold = 0.05 # set a cluster forming threshold based on a p-value for the cluster based permutation test

for n_age in ages:
    if n_folder == 'SSEP/':
        with open(root_path + n_folder + n_age + '_SSEP_wholebrain_cluster_test_' + n_meter + '.pkl', 'rb') as f:
            clu = pickle.load(f)
        good_cluster_inds = np.where(clu[2] < p_threshold)[0]
        good_clusters = [clu[1][idx] for idx in good_cluster_inds]

        for c in good_clusters:
            print(c[0])
            
            ## print the min and max and mean MNI coordinate for this cluster
            coord = [] 
            for i in np.arange(0,len(c[-1]),1):
                coord.append(np.round(src[0]['rr'][src[0]['vertno'][c[-1][i]]]*1000))
            np_coord = np.array(coord)
            print("min MNI coord:" + str(np.min(np_coord,axis=0)))
            print("max MNI coord:" + str(np.max(np_coord,axis=0)))
            print("mean MNI coord (center of mass):" + str(np.mean(np_coord,axis=0)))
            
            ## get all the ROIs in this cluster (no repeat)
            # ROIs = []
            # for i in np.arange(0,len(c[-1]),1):
            #     for nlabel in np.arange(0,len(label_names),1):
            #         if c[-1][i] in label_v_ind[nlabel][0] and label_names[nlabel] not in ROIs:
            #             ROIs.append(label_names[nlabel])
            # print(ROIs)
            
            ## get all the ROIs in this cluster (with repeat)
            ROIs = []
            count_ROIs = []
            for i in np.arange(0,len(c[-1]),1):
                for nlabel in np.arange(0,len(label_names),1):
                    if c[-1][i] in label_v_ind[nlabel][0]:
                        ROIs.append(label_names[nlabel])
            ROIs.sort()
            for label in label_names:
                count_ROIs.append(ROIs.count(label))
                print(str(ROIs.count(label)) + " " + label)
            
        ## visualize this cluster
        stc_all_cluster_vis = summarize_clusters_stc(
            clu, p_thresh = p_threshold, vertices=src, subject="fsaverage"
        )
        stc_all_cluster_vis.plot(src=src,clim=dict(kind="percent",lims=[99.7,99.75,99.975])) ## The first time point in this SourceEstimate object is the summation of all the clusters. Subsequent time points contain each individual cluster. The magnitude of the activity corresponds to the duration spanned (the freq in my case) by the cluster
        # stc_all_cluster_vis.plot(src=src) ## The first time point in this SourceEstimate object is the summation of all the clusters. Subsequent time points contain each individual cluster. The magnitude of the activity corresponds to the duration spanned (the freq in my case) by the cluster
        # stc_all_cluster_vis.plot_3d(src=src)
        
    elif n_folder == 'decoding/':
        decoding_acc = np.load(root_path + n_folder + n_age + data_type + 'decodingACC_' + n_meter +'.npy') 
        stc1.data = np.array([decoding_acc,decoding_acc]).transpose()
        stc1.plot(src=src,clim=dict(kind="percent",lims=[95,97.5,99.975]))
        stc1.plot_3d(src=src)

#%%####################################### The correlation results are visualized in stats_ME2.py