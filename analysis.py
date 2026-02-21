#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 09:54:23 2024

Run statistical analysis on the SSEP (sensor, ROI, whole brian), conn (ROI), and correlation
Input: .npy files in the "analyzed data" i.e. SSEP, ERSP, decoding, connectivity folders
Output: some figures, statistic results (stats, p-value)

@author: tzcheng
"""
#%%####################################### Import library  
import os
import pickle
import pandas as pd
import numpy as np
import scipy.stats as stats
import mne
from mne_connectivity import read_connectivity
from mne.viz import circular_layout
import matplotlib.pyplot as plt 

#%%####################################### Define functions
def ff(input_arr,target): 
    ## find the idx of the closest freqeuncy in freqs
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
    
def stats_SSEP(X,freqs,nonparametric):
    ## Compute parametric t-test and non-parametric 1D cluster test (across freqs) on X (psd1 - psd2)
    if nonparametric: 
        T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(X, seed = 0,verbose='ERROR')
        good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
        for i in np.arange(0,len(good_cluster_inds),1):
            print("The " + str(i+1) + "st significant cluster")
            print('Significant freqs: ' + str(freqs[clusters[good_cluster_inds[i]]]))
    else:   
        t,p = stats.ttest_1samp(X[:,ff(freqs,1.11)],0) # meter 1.11 Hz
        print('Testing freqs: ' + str(ff(freqs,1.11)))
        print('t statistics: ' + str(t))
        print('p-value: ' + str(p))
        t,p = stats.ttest_1samp(X[:,ff(freqs,1.67)],0) # meter 1.67 Hz
        print('Testing freqs: ' + str(ff(freqs,1.67)))
        print('t statistics: ' + str(t))
        print('p-value: ' + str(p))
        t,p = stats.ttest_1samp(X[:,ff(freqs,3.33)],0) # beat 3.3 Hz
        print('Testing freqs: ' + str(ff(freqs,3.33)))
        print('t statistics: ' + str(t))
        print('p-value: ' + str(p))

def wholebrain_spatio_temporal_cluster_test(X,n_meter,n_folder,p_threshold):
    ## Compute non-parametric 2D cluster test (across freqs and vertex) on X (psd1 - psd2) and save the cluster results
    print("Computing adjacency.")
    src = mne.read_source_spaces(subjects_dir + 'fsaverage/bem/fsaverage-vol-5-src.fif')
    adjacency = mne.spatial_src_adjacency(src)
    
    ## set the cluster settings 
    df = np.shape(X)[0] - 1  # degrees of freedom for the test
    t_threshold = stats.distributions.t.ppf(1 - p_threshold / 2, df=df)
    X = np.transpose(X,(0,2,1)) # subj, time, space            
    T_obs, clusters, cluster_p_values, H0 = clu = mne.stats.spatio_temporal_cluster_1samp_test(
        X,
        seed=0,
        adjacency=adjacency,
        n_jobs=None,
        threshold=t_threshold,
        buffer_size=None,
        verbose=True,
    )
    filename = root_path + n_folder + '7mo_SSEP_wholebrain_cluster_test_' + n_meter +'.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(clu, f) # clu: clustering results of T_obs, clusters, cluster_p_values, H0

def stats_CONN(conn1,conn2,freqs,FOI,ROI1,ROI2,fmin,fmax,ymin,ymax):
    XX = conn1-conn2
    ## compare whole freq spectrum between conditions and ages 
    # non-parametric
    threshold_tfce = dict(start=0, step=0.05)
    T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(XX[:,ROI1,ROI2,:], threshold = threshold_tfce,  seed = 0,verbose='ERROR') # test which frequency in Sensorimotor-Auditory is significant
    good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
    print(cluster_p_values)
    for i in np.arange(0,len(good_cluster_inds),1):
        print("The " + str(i+1) + "st significant cluster")
        print(clusters[good_cluster_inds[i]])
        print('Significant freqs: ' + str(freqs[clusters[good_cluster_inds[i]][0]]))
    # parametric
    t,p = stats.ttest_1samp(XX[:,ROI1,ROI2,ff(freqs,fmin):ff(freqs,fmax)].mean(-1),0) 
    print('Significant freqs for uncorrected ttest (' + str(fmin) + '-' + str(fmax) + ' Hz): ' + 't-stats = ' + str(t))
    print('Significant freqs for uncorrected ttest (' + str(fmin) + '-' + str(fmax) + ' Hz): ' + 'p-value = ' + str(p))

    t,p = stats.ttest_1samp(XX[:,ROI1,ROI2,:],0) 
    good_cluster_inds = np.where(p < 0.05)[0]
    print('Significant freqs for uncorrected ttest: ' + str(freqs[good_cluster_inds]))
    
    plt.figure()
    plot_err(conn1[:,ROI1,ROI2,:],'m',freqs)
    plot_err(conn2[:,ROI1,ROI2,:],'r',freqs)
    plt.xlim([4,35])
    plt.ylim([ymin,ymax])
    plt.vlines(x = freqs[good_cluster_inds],color='y',ymin = ymin, ymax=ymax)
    
    ## compare for a priori freq spectrum between conditions and ages 
    if FOI == "Delta": # 1-4 Hz
        X = XX[:,:,:,ff(freqs,1):ff(freqs,4)].mean(axis=3)
    elif FOI == "Theta": # 4-8 Hz
        X = XX[:,:,:,ff(freqs,4):ff(freqs,8)].mean(axis=3)
    elif FOI == "Alpha": # 8-12 Hz
        X = XX[:,:,:,ff(freqs,8):ff(freqs,12)].mean(axis=3)
    elif FOI == "Beta":  # 15-30 Hz
        X = XX[:,:,:,ff(freqs,15):ff(freqs,30)].mean(axis=3)
    else:  # broadband
        X = XX.mean(axis=3)
            
    t,p = stats.ttest_1samp(X,0)

def extract_CDI(CDIAge,CDIscore):
    subj_7mo = []
    subj_path=['/media/tzcheng/storage/ME2_MEG/Zoe_analyses/7mo/'] 
    for file in os.listdir(subj_path[0]):
        if file.endswith('7m'):
            print('append ' + file)
            subj_7mo.append(file[:-3]) 

    ## Select subjects CDI score for those who have neural data
    CDI_WS0 = pd.read_excel(root_path + 'ME2_WG_WS_zoe.xlsx',sheet_name=2)
    CDI_WS_7mo = CDI_WS0[CDI_WS0['ParticipantId'].isin(subj_7mo)] # select the subjects who has neural data 
    
    ## De-select subjects who has neural data but does not have CDI data: 7mo ('me2_203', 'me2_120', 'me2_117')
    subj_noCDI_ind = [2,8,25]
    CDI = CDI_WS_7mo[CDI_WS_7mo['CDIAge'] == CDIAge][CDIscore]
   
    return CDI, subj_noCDI_ind
    
def extract_MEG(MEGAge,data_type,n_analysis,n_condition,subj_noCDI_ind,conn_FOI,ROI1,ROI2,SSEP_FOI,F1,F2):
    if n_analysis == 'conn_plv':
        conn0 = read_connectivity(root_path + 'connectivity/' + MEGAge + '_group' + n_condition + '_stc_rs_mne_mag6pT' + data_type + n_analysis) 
        conn = conn0.get_data(output='dense')
        if MEGAge == '7mo':
            conn = np.delete(conn,subj_noCDI_ind,axis=0) # delete the 3 subjects 'me2_203', 'me2_120', 'me2_117' who don't have CDI
        conn_delta = conn[:,ROI1,ROI2,ff(conn0.freqs,1):ff(conn0.freqs,4)].mean(-1)
        conn_theta = conn[:,ROI1,ROI2,ff(conn0.freqs,4):ff(conn0.freqs,8)].mean(-1)
        conn_alpha = conn[:,ROI1,ROI2,ff(conn0.freqs,8):ff(conn0.freqs,12)].mean(-1)
        conn_beta = conn[:,ROI1,ROI2,ff(conn0.freqs,15):ff(conn0.freqs,30)].mean(-1)
        conn_alpha_beta = conn[:,ROI1,ROI2,ff(conn0.freqs,F1):ff(conn0.freqs,F2)].mean(-1)
        if conn_FOI == 'delta':
            return conn_delta
        elif conn_FOI == 'theta':
            return conn_theta
        elif conn_FOI == 'alpha':
            return conn_alpha      
        elif conn_FOI == 'beta':
            return conn_beta     
        elif conn_FOI == 'alpha_beta':
            print('From freqs ' + str(F1) + ' Hz to ' + str(F2) + ' Hz')
            return conn_alpha_beta     
    elif n_analysis == 'psds':
        SSEP0 = np.load(root_path + 'SSEP/' + MEGAge + '_group' + n_condition + '_stc_rs_mne_mag6pT' + data_type + n_analysis + '.npz') 
        SSEP = SSEP0[SSEP0.files[0]]
        if MEGAge == '7mo':
            SSEP = np.delete(SSEP,subj_noCDI_ind,axis=0) # delete the 3 subjects 'me2_203', 'me2_120', 'me2_117' who don't have CDI
        SSEP_triple = SSEP[:,:,ff(SSEP0[SSEP0.files[1]],1.11)]
        SSEP_duple = SSEP[:,:,ff(SSEP0[SSEP0.files[1]],1.67)]
        SSEP_triple_1harm = SSEP[:,:,ff(SSEP0[SSEP0.files[1]],2.22)]
        SSEP_beat = SSEP[:,:,ff(SSEP0[SSEP0.files[1]],3.33)]
        if SSEP_FOI == '1.11Hz':
            return SSEP_triple
        elif SSEP_FOI == '1.67Hz':
            return SSEP_duple
        elif SSEP_FOI == '2.22Hz':
            return SSEP_triple_1harm
        elif SSEP_FOI == '3.33Hz':
            return SSEP_beat   
        else:
            print("Freqs not exist.")

def wholebrain_corr_cluster_test(MEG, CDI, src, filename, n_permutations=500, p_value_threshold=0.05):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    # Prep the data
    np.random.seed(42)
    n_subjects, n_sensors = np.shape(MEG)[0], np.shape(MEG)[1]
    neural = MEG
    behavior = CDI.to_numpy()

    print("Computing adjacency.")
    adjacency = mne.spatial_src_adjacency(src)
    adjacency_sparse = csr_matrix(adjacency)

    # Compute observed correlations
    r_vals = np.array([pearsonr(neural[:, i], behavior)[0] for i in range(n_sensors)])
    t_vals = r_vals * np.sqrt((n_subjects - 2) / (1 - r_vals**2))

    # Threshold for cluster formation (e.g., |t| > critical t for p<0.05, df=n-2)
    t_crit = t.ppf(1 - p_value_threshold / 2, df=n_subjects - 2)  # two-tailed
    supra_thresh = np.abs(t_vals) > t_crit

    # Define clusters 
    sub_adj = adjacency_sparse[supra_thresh][:, supra_thresh]
    n_clusters, labels = connected_components(sub_adj, directed=False)
    cluster_labels = np.zeros(n_sensors, dtype=int)
    cluster_labels[supra_thresh] = labels + 1  # +1 to start cluster IDs at 1

    # Compute cluster statistics (sum of t-values within each cluster)
    cluster_stats = np.array([np.sum(np.abs(t_vals)[cluster_labels == c + 1]) for c in range(n_clusters)])

    # Build null distribution with permutation
    max_cluster_stats = np.zeros(n_permutations)
    for p in range(n_permutations):
        print("Iteration " + str(p))
        y_perm = np.random.permutation(behavior)
        r_perm = np.array([pearsonr(neural[:, i], y_perm)[0] for i in range(n_sensors)])
        t_perm = r_perm * np.sqrt((n_subjects - 2) / (1 - r_perm**2))
        supra_perm = np.abs(t_perm) > t_crit
        sub_adj_perm = adjacency_sparse[supra_perm][:, supra_perm]
        perm_n_clusters, perm_labels = connected_components(sub_adj_perm, directed=False)
        cluster_labels_perm = np.zeros(n_sensors, dtype=int)
        cluster_labels_perm[supra_perm] = perm_labels + 1  # +1 to start cluster IDs at 1
        if perm_n_clusters > 0:
            perm_stats = np.array([np.sum(np.abs(t_perm)[cluster_labels_perm == c + 1]) for c in range(perm_n_clusters)])
            max_cluster_stats[p] = np.max(perm_stats)
        else:
            max_cluster_stats[p] = 0

    # Compare observed cluster stats to null (family-wise correction)
    p_values = np.array([np.mean(max_cluster_stats >= s) for s in cluster_stats])

    # Print results
    results = []
    for i, (s, p) in enumerate(zip(cluster_stats, p_values)):
        results.append(f"Cluster {i+1}: sum(|t|)={s:.2f}, p={p:.4f}")
        print(f"Cluster {i+1}: sum(|t|)={s:.2f}, p={p:.4f}")
    
    return results, cluster_stats, p_values, cluster_labels, max_cluster_stats

#%%####################################### Set path
root_path = '/media/tzcheng/storage/ME2_MEG/Zoe_analyses/me2_meg_analysis/'
subjects_dir = '/media/tzcheng/storage2/subjects/'

#%%####################################### set up the template brain
stc1 = mne.read_source_estimate('/media/tzcheng/storage/BabyRhythm/br_03/sss_fif/br_03_01_stc_mne_morph_mag6pT-vl.stc')
src = mne.read_source_spaces(subjects_dir + 'fsaverage/bem/fsaverage-vol-5-src.fif')
label_v_ind = np.load('/media/tzcheng/storage/scripts_zoe/ROI_lookup.npy', allow_pickle=True)
fname_aseg = subjects_dir + 'fsaverage/mri/aparc+aseg.mgz'
label_names = mne.get_volume_labels_from_aseg('/media/tzcheng/storage2/subjects/fsaverage/mri/aparc+aseg.mgz')

#%% Parameters
conditions = ['_02','_03','_04'] # random, duple, triple
folders = ['SSEP/','connectivity/'] 
analysis = ['psds','conn_plv']
which_data_type = ['_sensor_','_roi_redo4_','_morph_'] 

#%%####################################### Analysis on the sensor SSEP
random = np.load(root_path + 'SSEP/7mo_group_02_rs_mag6pT_sensor_psds.npz') 
duple = np.load(root_path + 'SSEP/7mo_group_03_rs_mag6pT_sensor_psds.npz') 
triple = np.load(root_path + 'SSEP/7mo_group_04_rs_mag6pT_sensor_psds.npz') 
freqs = random[random.files[1]]
psds_random = random[random.files[0]].mean(axis = 1)
psds_duple = duple[duple.files[0]].mean(axis = 1)
psds_triple = triple[triple.files[0]].mean(axis = 1)
print("-------------------Doing duple-------------------")
stats_SSEP(psds_duple-psds_random,freqs,True)
print("-------------------Doing triple-------------------")
stats_SSEP(psds_triple-psds_random,freqs,True)

#%%####################################### Analysis on the ROI SSEP
root_path = '/home/tzcheng/Desktop/ME2_upload_to_github/data/'
n_folder = 'SSEP/'
n_analysis = 'psds'
data_type = '_roi_redo4_'

fname_aseg = subjects_dir + 'fsaverage/mri/aparc+aseg.mgz'
label_names = np.asarray(["Auditory", "SensoryMotor", "BG"])
nROI = np.arange(0,len(label_names),1)

for n in nROI: 
    print("Doing ROI SSEP: " + label_names[n])
    random0 = np.load(root_path + 'SSEP/7mo_group_02_stc_rs_mne_mag6pT_roi_redo3_psds.npz') 
    duple0 = np.load(root_path + 'SSEP/7mo_group_03_stc_rs_mne_mag6pT_roi_redo3_psds.npz') 
    triple0 = np.load(root_path + 'SSEP/7mo_group_04_stc_rs_mne_mag6pT_roi_redo3_psds.npz') 
    random = random0[random0.files[0]]
    duple = duple0[duple0.files[0]]
    triple = triple0[triple0.files[0]]
    freqs = random0[random0.files[1]]
    print("-------------------Doing duple-------------------")
    stats_SSEP(duple[:,n,:]-random[:,n,:],freqs,True)
    print("-------------------Doing triple-------------------")
    stats_SSEP(triple[:,n,:]-random[:,n,:],freqs,True)
    
#%%####################################### Analysis on the wholebrain SSEP
p_threshold = 0.05 # set a cluster forming threshold based on a p-value for the cluster based permutation test
random0 = np.load(root_path + 'SSEP/7mo_group_02_stc_rs_mne_mag6pT_morph_psds.npz') 
duple0 = np.load(root_path + 'SSEP/7mo_group_03_stc_rs_mne_mag6pT_morph_psds.npz') 
triple0 = np.load(root_path + 'SSEP/7mo_group_04_stc_rs_mne_mag6pT_morph_psds.npz') 
random = random0[random0.files[0]]
duple = duple0[duple0.files[0]]
triple = triple0[triple0.files[0]]
freqs = random0[random0.files[1]] 
wholebrain_spatio_temporal_cluster_test(duple-random,'duple',n_folder,p_threshold)
wholebrain_spatio_temporal_cluster_test(triple-random,'triple',n_folder,p_threshold)

#%%####################################### Analysis on the ROI conn
n_folder = 'connectivity/'
n_analysis = 'conn_plv'
data_type = '_roi_redo4_'

ROI1 = 1
ROI2 = 0
fmin = 5
fmax = 10
FOI = 'Beta' # Delta, Theta, Alpha, Beta 

random = read_connectivity(root_path + 'connectivity/7mo_group_02_stc_rs_mne_mag6pT_roi_redo4_conn_plv') 
duple = read_connectivity(root_path + 'connectivity/7mo_group_03_stc_rs_mne_mag6pT_roi_redo4_conn_plv') 
triple = read_connectivity(root_path + 'connectivity/7mo_group_04_stc_rs_mne_mag6pT_roi_redo4_conn_plv') 
freqs = np.array(random.freqs)
random_conn = random.get_data(output='dense')
duple_conn = duple.get_data(output='dense')
triple_conn = triple.get_data(output='dense')
print("-------------------Doing duple-------------------")
stats_CONN(duple_conn,random_conn,freqs,FOI,ROI1,ROI2,fmin,fmax, 0.39,1)
print("-------------------Doing triple-------------------")
stats_CONN(triple_conn,random_conn,freqs,FOI,ROI1,ROI2,fmin,fmax, 0.39,1)

#%%####################################### Correlation analysis initial setting
#%% run the loop
meter = conditions[2]
data_type = '_roi_redo4_'

## parameter for the wholebrain SSEP test
if meter == 'duple':
    peak_freqs = ['1.67Hz','3.33Hz']
elif meter == 'triple':
    peak_freqs = ['1.11Hz','2.22Hz','3.33Hz']

## parameters for the ROI conn test
label_names = np.asarray(["Auditory", "SensoryMotor", "BG", "IFG"])
ROI1 = 2 # correspond to the label_names
ROI2 = 1 # correspond to the label_names
F1 = 4
F2 = 35
FOI = 'alpha_beta'
CDI,subj_noCDI_ind = extract_CDI('7mo',27,'VOCAB')

#%% Correlation analysis between ROI conn and CDI 
for peak_freq in peak_freqs:
    MEG = extract_MEG('7mo',data_type,'psds',meter,subj_noCDI_ind,'theta',ROI1,ROI2,peak_freq,F1,F2)    
    filename = '7mo' + meter + '_' + peak_freq + '_permutation_fix'
    print(filename)
    results, cluster_stats, p_values, cluster_labels, max_cluster_stats = wholebrain_corr_cluster_test(MEG, CDI, src, filename, n_permutations=500, p_value_threshold=0.05)

#%%####################################### Model relationship between SSEP and CDI   
MEG = extract_MEG('7mo',data_type,'psds',meter,subj_noCDI_ind,FOI,ROI1,ROI2,peak_freq,F1,F2)
MEG_rand = extract_MEG('7mo',data_type,'psds','_02',subj_noCDI_ind,FOI,ROI1,ROI2,peak_freq,F1,F2)
MEG_diff = MEG - MEG_rand

#%% Correlation analysis between ROI SSEP and CDI 
for n,ROI in enumerate(label_names):
    print(ROI)
    print(pearsonr(MEG[:,n], CDI))

#%% Correlation analysis between wholebrain SSEP and CDI 
#### Without correction for multiple comparison
r_all = []
p_all = []
for n in np.arange(0,len(MEG[0])):
    # print('Doing vertex ' + str(n))
    tmp_r,tmp_p = pearsonr(MEG_diff[:,n],CDI)
    r_all.append(tmp_r)
    p_all.append(tmp_p)

            
stc1.data = np.array([r_all,r_all]).transpose()
stc1.subject = 'fsaverage'
stc1.plot(src=src,clim=dict(kind="percent",lims=[95,97.5,99.975]))
stc1.plot_3d(src=src)

#### Cluster-based permutation test correction for multiple comparison 
filename = '7mo' + meter + '_' + peak_freq + '_diff_permutation'
results, cluster_stats, p_values, cluster_labels, max_cluster_stats = wholebrain_corr_cluster_test(MEG, CDI, src, filename, n_permutations=500, p_value_threshold=0.05)

#%%####################################### Model relationship between ROI conn and CDI   
## sensorimotor, IFG-motor, and IFG-auditory showed the significance for 11 mo
MEG = extract_MEG('7mo',data_type,'conn_plv',meter,subj_noCDI_ind,FOI,ROI1,ROI2,peak_freq,F1,F2) 
plt.figure()
plt.scatter(MEG,CDI)
plt.ylim([50,710])
plt.xlim([0.2,0.95])
plt.title('Conn between ' + label_names[ROI1] + ' ' + label_names[ROI2])
print('Conn between ' + label_names[ROI1] + ' ' + label_names[ROI2])
print(pearsonr(MEG, CDI))