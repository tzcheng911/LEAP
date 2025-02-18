#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 09:54:23 2024

Run statistical analysis on the output SSEP (sensor, ROI, whole brian), conn (ROI) from analysis_ME2.py
Apply stats_SSEP on sensor and ROI SSEP
Apply spatio_temporal_cluster_test on whole brian SSEP
Apply stats_CONN on ROI
Apply correlation between (1) AM conn and CDI (2) SSEP and CDI (whole brain results)
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
from scipy.stats import pearsonr, spearmanr
import mne
from mne_connectivity.viz import plot_connectivity_circle
from mne_connectivity import spectral_connectivity_epochs, spectral_connectivity_time,read_connectivity
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
            print(clusters[good_cluster_inds[i]])
            print('Significant freqs: ' + str(freqs[clusters[good_cluster_inds[i]]]))
    else:   
        t,p = stats.ttest_1samp(X[:,ff(freqs,1.11)].mean(axis=1),0) # meter 1.11 Hz
        print('Testing freqs: ' + str(ff(freqs,1.11)))
        print('t statistics: ' + str(t))
        print('p-value: ' + str(p))
        t,p = stats.ttest_1samp(X[:,ff(freqs,1.67)].mean(axis=1),0) # meter 1.67 Hz
        print('Testing freqs: ' + str(ff(freqs,1.67)))
        print('t statistics: ' + str(t))
        print('p-value: ' + str(p))
        t,p = stats.ttest_1samp(X[:,ff(freqs,3.33)].mean(axis=1),0) # beat 3.3 Hz
        print('Testing freqs: ' + str(ff(freqs,3.33)))
        print('t statistics: ' + str(t))
        print('p-value: ' + str(p))

def wholebrain_spatio_temporal_cluster_test(X,n_meter,n_age,n_folder,p_threshold,freqs):
    ## Compute non-parametric 2D cluster test (across freqs and vertex) on X (psd1 - psd2) and save the cluster results
    print("Computing adjacency.")
    stc1 = mne.read_source_estimate('/media/tzcheng/storage/BabyRhythm/br_03/sss_fif/br_03_01_stc_mne_morph_mag6pT-vl.stc')
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
    filename = root_path + n_folder + n_age + '_SSEP_wholebrain_cluster_test_' + n_meter +'.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(clu, f) # clu: clustering results of T_obs, clusters, cluster_p_values, H0

def stats_CONN(conn1,conn2,freqs,nlines,FOI,label_names,title,ROI1,ROI2,ymin,ymax):
    XX = conn1-conn2
    ## compare whole freq spectrum between conditions and ages 
    # non-parametric
    T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(XX[:,ROI1,ROI2,:], seed = 0,verbose='ERROR') # test which frequency in Sensorimotor-Auditory is significant
    good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
    print(cluster_p_values)
    for i in np.arange(0,len(good_cluster_inds),1):
        print("The " + str(i+1) + "st significant cluster")
        print(clusters[good_cluster_inds[i]])
        print('Significant freqs: ' + str(freqs[clusters[good_cluster_inds[i]][0]]))
    # parametric
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
    labels = mne.read_labels_from_annot("sample", parc="aparc", subjects_dir=subjects_dir)
    label_colors = [label.color for label in labels]
      
    node_order = list()
    node_order.extend(label_names)  # reverse the order
    node_angles = circular_layout(
         label_names, node_order, start_pos=90, group_boundaries=[0, len(label_names) / 2])
    
    # fig, ax = plt.subplots(figsize=(8, 8), facecolor="black", subplot_kw=dict(polar=True))
    # plot_connectivity_circle(
    # 1-p,
    # label_names,
    # n_lines=nlines, # plot the top n lines
    # vmin=0.95, # correspond to p = 0.05
    # vmax=1, # correspond to p = 0
    # node_angles=node_angles,
    # node_colors=label_colors,
    # title= title + " 1-p-value " + FOI,
    # ax=ax)
    # fig.tight_layout() 

def convert_to_csv(data_type,labels,n_analysis,n_folder,ROI1,ROI2):
    lm_np = []
    sub_col = [] 
    age_col = []
    cond_col = []
    ROI_col = []
    ages = ['7mo','11mo','br'] 
    conditions = ['_02','_03','_04']
    subj_path=['/media/tzcheng/storage/ME2_MEG/Zoe_analyses/7mo/' ,
               '/media/tzcheng/storage/ME2_MEG/Zoe_analyses/11mo/',
               '/media/tzcheng/storage/BabyRhythm/'] # NEED TO BE THE ORDER OF where 7mo, 11mo and br data at
    for n_age,age in enumerate(ages):
        print(age)
        for n_cond,cond in enumerate(conditions):
            print(cond)
            if n_analysis == 'psds':
                if data_type == which_data_type[1] or data_type == which_data_type[2]:
                    print('-----------------Extracting ROI data-----------------')
                    for nROI, ROI in enumerate(labels):
                        print(ROI)
                        data0 = np.load(root_path + n_folder + age + '_group' + cond + '_stc_rs_mne_mag6pT' + data_type + n_analysis +'.npz') 
                        data1 = data0[data0.files[0]]
                        freqs = data0[data0.files[1]]
                        print(np.shape(data1))
                        data2 = np.vstack((data1[:,nROI,ff(freqs,1.11)],data1[:,nROI,ff(freqs,1.67)],data1[:,nROI,ff(freqs,3.33)])).transpose()
                        lm_np.append(data2)
                        for file in os.listdir(subj_path[n_age]):
                            if file.endswith('7m') or file.endswith('11m') or file.startswith('br'):
                                print('save '  + file)
                                sub_col.append(file)
                                cond_col.append(cond)
                                age_col.append(age)
                                ROI_col.append(ROI)
                    lm_df = pd.DataFrame({'sub_id': sub_col,'age':age_col,'condition':cond_col, 'ROI':ROI_col,'1.11Hz': np.concatenate(lm_np)[:,0], '1.67Hz': np.concatenate(lm_np)[:,1],'3.3Hz': np.concatenate(lm_np)[:,2]})
                elif data_type == which_data_type[0]:
                    print('-----------------Extracting sensor data-----------------')
                    data0 = np.load(root_path + n_folder + age + '_group' + cond + '_rs_mag6pT' + data_type + n_analysis +'.npz') 
                    data1 = data0[data0.files[0]].mean(axis=1)
                    print(np.shape(data1))
                    data2 = np.vstack((data1[:,[6,7]].mean(axis=1),data1[:,[12,13]].mean(axis=1),data1[:,[30,31]].mean(axis=1))).transpose()
                    lm_np.append(data2)
                    for file in os.listdir(subj_path[n_age]):
                        if file.endswith('7m') or file.endswith('11m') or file.startswith('br'):
                            print('save '  + file)
                            sub_col.append(file)
                            cond_col.append(cond)
                            age_col.append(age)
                    lm_df = pd.DataFrame({'sub_id': sub_col,'age':age_col,'condition':cond_col,'1.11Hz': np.concatenate(lm_np)[:,0], '1.67Hz': np.concatenate(lm_np)[:,1],'3.3Hz': np.concatenate(lm_np)[:,2]}) 
            elif n_analysis == 'conn_plv':
                data0 = read_connectivity(root_path + n_folder + age + '_group' + cond + '_stc_rs_mne_mag6pT' + data_type + n_analysis) 
                freqs = data0.freqs
                data0_conn = data0.get_data(output='dense')
                print(np.shape(data0_conn))
                data = np.vstack((data0_conn[:,ROI1,ROI2,ff(freqs,1):ff(freqs,4)].mean(axis=-1),
                                  data0_conn[:,ROI1,ROI2,ff(freqs,4):ff(freqs,8)].mean(axis=-1),
                                  data0_conn[:,ROI1,ROI2,ff(freqs,8):ff(freqs,12)].mean(axis=-1),
                                  data0_conn[:,ROI1,ROI2,ff(freqs,15):ff(freqs,30)].mean(axis=-1),
                                  data0_conn[:,ROI1,ROI2,:].mean(axis=-1))).transpose() # delta, theta, alpha, beta total 4 cols
                lm_np.append(data)
                for file in os.listdir(subj_path[n_age]):
                    if file.endswith('7m') or file.endswith('11m') or file.startswith('br'):
                        print('save '  + file)
                        sub_col.append(file)
                        cond_col.append(cond)
                        age_col.append(age)
                lm_df = pd.DataFrame({'sub_id': sub_col,'age':age_col,'condition':cond_col,
                          'Delta conn': np.concatenate(lm_np)[:,0], 
                          'Theta conn': np.concatenate(lm_np)[:,1],
                          'Alpha conn': np.concatenate(lm_np)[:,2],
                          'Beta conn': np.concatenate(lm_np)[:,3],
                          'Broadband conn': np.concatenate(lm_np)[:,4]})
            lm_df.to_csv(root_path + n_folder + n_analysis + data_type + labels[ROI1][0] + label_names[ROI2][0] + '.csv')

def extract_CDI(MEGAge,CDIAge,CDIscore):
    ages = ['7mo','11mo']
    subj_7mo = []
    subj_11mo = []
    subj_path=['/media/tzcheng/storage/ME2_MEG/Zoe_analyses/7mo/' ,
            '/media/tzcheng/storage/ME2_MEG/Zoe_analyses/11mo/']
    for n_age,age in enumerate(ages):# only need the me2_7m and me2_11m
        for file in os.listdir(subj_path[n_age]):
            if file.endswith('7m'):
                print('append ' + file)
                subj_7mo.append(file[:-3]) 
            elif file.endswith('11m'):
                print('append ' + file)
                subj_11mo.append(file[:-4])

    ## Select subjects CDI score for those who have neural data
    CDI_WG0 = pd.read_excel(root_path + 'ME2_WG_WS_zoe.xlsx',sheet_name=0)
    CDI_WS0 = pd.read_excel(root_path + 'ME2_WG_WS_zoe.xlsx',sheet_name=2)
    CDI_WS_7mo = CDI_WS0[CDI_WS0['ParticipantId'].isin(subj_7mo)] # select the subjects who has neural data 
    CDI_WS_11mo = CDI_WS0[CDI_WS0['ParticipantId'].isin(subj_11mo)] # select the subjects who has neural data 
    
    ## De-select subjects who has neural data but does not have CDI data: 7mo ('me2_203', 'me2_120', 'me2_117')
    subj_noCDI = list(set(subj_7mo) - set(CDI_WS0['ParticipantId'])) # same result in list(set(subj_all) - set(CDI_WG0['ParticipantId']))   
    subj_noCDI_ind = [2,8,25] # CAUTION hardcoded manual input here, check if this is the data storing order for 7mo in group_ME2.py (confirmed 2025/1/13 Zoe)
    if MEGAge == '7mo':
        CDI = CDI_WS_7mo[CDI_WS_7mo['CDIAge'] == CDIAge][CDIscore]
    elif MEGAge == '11mo':
        CDI = CDI_WS_11mo[CDI_WS_11mo['CDIAge'] == CDIAge][CDIscore]
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
        if SSEP_FOI == '1.11 Hz':
            return SSEP_triple
        elif SSEP_FOI == '1.67 Hz':
            return SSEP_duple
        elif SSEP_FOI == '2.22 Hz':
            return SSEP_triple_1harm
        elif conn_FOI == '3.33 Hz':
            return SSEP_beat   
        else:
            print("Freqs not exist.")

#%%####################################### Set path
root_path = '/media/tzcheng/storage/ME2_MEG/Zoe_analyses/me2_meg_analysis/'
subjects_dir = '/media/tzcheng/storage2/subjects/'

#%% Parameters
ages = ['7mo','11mo','br'] 
conditions = ['_02','_03','_04'] # random, duple, triple
folders = ['SSEP/','ERSP/','decoding/','connectivity/'] 
analysis = ['psds','conn_plv','conn_coh','conn_pli','conn_GC_AM','conn_GC_MA']
which_data_type = ['_sensor_','_roi_','_roi_redo4_','_morph_'] 

#%%####################################### Analysis on the sensor SSEP
for n_age in ages:
    print("Doing age " + n_age)
    random = np.load(root_path + 'SSEP/' + n_age + '_group_02_rs_mag6pT_sensor_psds.npz') 
    duple = np.load(root_path + 'SSEP/' + n_age + '_group_03_rs_mag6pT_sensor_psds.npz') 
    triple = np.load(root_path + 'SSEP/' + n_age + '_group_04_rs_mag6pT_sensor_psds.npz') 
    freqs = random[random.files[1]]
    psds_random = random[random.files[0]].mean(axis = 1)
    psds_duple = duple[duple.files[0]].mean(axis = 1)
    psds_triple = triple[triple.files[0]].mean(axis = 1)
    print("-------------------Doing duple-------------------")
    stats_SSEP(psds_duple-psds_random,freqs,True)
    print("-------------------Doing triple-------------------")
    stats_SSEP(psds_triple-psds_random,freqs,True)
convert_to_csv('_sensor_',0,'psds','SSEP',1,0)

#%%####################################### Analysis on the ROI SSEP
n_folder = folders[0] # 0: SSEP
n_analysis = analysis[0] # 0: psds
data_type = which_data_type[2] # 1:_roi_ or 2:_roi_redo_

fname_aseg = subjects_dir + 'fsaverage/mri/aparc+aseg.mgz'
if data_type == '_roi_':
    label_names = np.asarray(mne.get_volume_labels_from_aseg(fname_aseg))
    nROI = [72,108,66,102,64,100,59,95,7,8,26,27,60,61,62,96,97,98,50,86,71,107]
elif data_type == '_roi_redo4_':
    # label_names = np.asarray(["AuditoryL", "AuditoryR", "MotorL", "MotorR", "SensoryL", "SensoryR", "BGL", "BGR", "IFGL", "IFGR"])
    # label_names = np.asarray(["Auditory", "Motor", "Sensory", "BG", "IFG"])
    label_names = np.asarray(["Auditory", "SensoryMotor", "BG", "IFG"])
    nROI = np.arange(0,len(label_names),1)

for n_age in ages:
    for n in nROI: 
        print("Doing ROI SSEP: " + label_names[n])
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
convert_to_csv(data_type,label_names,n_analysis,n_folder,1,0)
    
#%%####################################### Analysis on the whole brain SSEP
p_threshold = 0.05 # set a cluster forming threshold based on a p-value for the cluster based permutation test
for n_age in ages:
    print("Doing age " + n_age)
    random0 = np.load(root_path + n_folder + n_age + '_group_02_stc_rs_mne_mag6pT_morph_psds.npz') 
    duple0 = np.load(root_path + n_folder + n_age + '_group_03_stc_rs_mne_mag6pT_morph_psds.npz') 
    triple0 = np.load(root_path + n_folder + n_age + '_group_04_stc_rs_mne_mag6pT_morph_psds.npz') 
    random = random0[random0.files[0]]
    duple = duple0[duple0.files[0]]
    triple = triple0[triple0.files[0]]
    freqs = random0[random0.files[1]] 
    wholebrain_spatio_temporal_cluster_test(duple-random,'duple',n_age,n_folder,p_threshold,freqs)
    wholebrain_spatio_temporal_cluster_test(triple-random,'triple',n_age,n_folder,p_threshold,freqs)

#%%####################################### Analysis on the ROI conn
n_folder = folders[3] # 0: connectivity/
n_analysis = analysis[1] # 1:'conn_plv', 2:'conn_coh', 3:'conn_pli'
data_type = which_data_type[2] # 1:_roi_ or 2:_roi_redo5_

random_conn_all = []
duple_conn_all = []
triple_conn_all = []

nlines = 10
ROI1 = 2
ROI2 = 0
FOI = 'Beta' # Delta, Theta, Alpha, Beta 

for n_age in ages[:-1]:
    print("Doing connectivity " + n_age)
    random = read_connectivity(root_path + n_folder + n_age + '_group_02_stc_rs_mne_mag6pT' + data_type + n_analysis) 
    duple = read_connectivity(root_path + n_folder + n_age + '_group_03_stc_rs_mne_mag6pT' + data_type + n_analysis) 
    triple = read_connectivity(root_path + n_folder + n_age + '_group_04_stc_rs_mne_mag6pT' + data_type + n_analysis) 
    freqs = np.array(random.freqs)
    random_conn = random.get_data(output='dense')
    duple_conn = duple.get_data(output='dense')
    triple_conn = triple.get_data(output='dense')
    random_conn_all.append(random_conn)
    duple_conn_all.append(duple_conn)
    triple_conn_all.append(triple_conn)
    print("-------------------Doing duple-------------------")
    stats_CONN(duple_conn,random_conn,freqs,nlines,FOI,label_names,n_age + ' duple vs. random ' + n_analysis,ROI1,ROI2,0.39,1)
    print("-------------------Doing triple-------------------")
    stats_CONN(triple_conn,random_conn,freqs,nlines,FOI,label_names,n_age + ' triple vs. random ' + n_analysis,ROI1,ROI2,0.39,1)
print("-------------------Doing duple-------------------")
conn1 = duple_conn_all[0]-random_conn_all[0] # 7mo
conn2 = duple_conn_all[1]-random_conn_all[1] # 11mo
stats_CONN(conn1,conn2,freqs,nlines,FOI,label_names,'7 mo vs 11 mo',ROI1,ROI2,-0.15,0.25)
print("-------------------Doing triple-------------------")
conn1 = triple_conn_all[0]-random_conn_all[0] # 7mo
conn2 = triple_conn_all[1]-random_conn_all[1] # 11mo
stats_CONN(conn1,conn2,freqs,nlines,FOI,label_names,'7 mo vs 11 mo',ROI1,ROI2,-0.15,0.25)

convert_to_csv('_roi_redo4_',label_names,'conn_plv','connectivity/',3,0)

#%%####################################### Correlation analysis between neural responses and CDI   
meter = '_03'
age = '7mo'
peak_freq = '2.22 Hz'
data_type = '_roi_redo4_'
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

## correlation between ROI CONN and CDI 
ROI1 = 2
ROI2 = 1
F1 = 5
F2 = 9
FOI = 'alpha_beta'

## correlation between conn and CDI: sensorimotor, IFG-motor, and IFG-auditory showed the significance for 11 mo
CDI,subj_noCDI_ind = extract_CDI(age,27,'VOCAB')
MEG = extract_MEG(age,data_type,'conn_plv',meter,subj_noCDI_ind,FOI,ROI1,ROI2,peak_freq,F1,F2) 
# CDI1,subj_noCDI_ind = extract_CDI('7mo',27,'VOCAB')
# CDI2,subj_noCDI_ind = extract_CDI('11mo',27,'VOCAB')
# MEG1 = extract_MEG('7mo',data_type,'conn_plv',meter,subj_noCDI_ind,FOI,ROI1,ROI2,peak_freq,F1,F2)
# MEG2 = extract_MEG('11mo',data_type,'conn_plv',meter,subj_noCDI_ind,FOI,ROI1,ROI2,peak_freq,F1,F2)
# CDI = pd.concat([CDI1,CDI2])
# MEG = np.concatenate((MEG1,MEG2))

plt.figure()
plt.scatter(MEG,CDI)
print('Conn between ' + label_names[ROI1] + ' ' + label_names[ROI2])
print(pearsonr(MEG, CDI))
# print(spearmanr(MEG, CDI))
#%%
## correlation between ROI SSEP and CDI 
CDI,subj_noCDI_ind = extract_CDI('7mo',27,'VOCAB')
MEG = extract_MEG('7mo',data_type,'psds',meter,subj_noCDI_ind,'theta',ROI1,ROI2,peak_freq)
# CDI = pd.concat([CDI1,CDI2])
# MEG = np.concatenate((MEG1,MEG2))
for n,ROI in enumerate(label_names):
    plt.figure()
    plt.scatter(MEG[:,n],CDI)
    print(pearsonr(MEG[:,n], CDI))
    print(pearsonr(MEG[:,n], CDI))

## correlation between whole brain SSEP and CDI 
data_type = '_morph_'
r_all = []
p_all = []
CDI,subj_noCDI_ind = extract_CDI('7mo',27,'VOCAB')
MEG = extract_MEG('7mo',data_type,'psds',meter,subj_noCDI_ind,'theta',ROI1,ROI2,peak_freq)
# CDI = pd.concat([CDI1,CDI2])
# MEG = np.concatenate((MEG1,MEG2))
for n in np.arange(0,len(MEG[0])):
    print('Doing vertex ' + str(n))
    tmp_r,tmp_p = pearsonr(MEG[:,n],CDI)
    r_all.append(tmp_r)
    p_all.append(tmp_p)

stc1 = mne.read_source_estimate('/media/tzcheng/storage/BabyRhythm/br_03/sss_fif/br_03_01_stc_mne_morph_mag6pT-vl.stc')
src = mne.read_source_spaces(subjects_dir + 'fsaverage/bem/fsaverage-vol-5-src.fif')
stc1.data = np.array([r_all,r_all]).transpose()
stc1.plot(src=src,clim=dict(kind="percent",lims=[95,97.5,99.975]))