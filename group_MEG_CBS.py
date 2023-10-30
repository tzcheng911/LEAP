#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 17:57:34 2023

@author: tzcheng
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats,signal
import os
    
#%%########################################
root_path='/media/tzcheng/storage/CBS/'
subjects_dir = '/media/tzcheng/storage2/subjects/'

os.chdir(root_path)

## parameters 
subj = [] # A104 got some technical issue
for file in os.listdir():
    if file.startswith('cbs_A'):
        subj.append(file)

runs = ['01','02']
run = runs [0]

#%% output the sensor time series in npy files
group_mmr1=[]
group_mmr2=[]

for s in subj:
    print('Extracting ' + s + ' data')
    file_in = root_path + s + '/sss_fif/' + s
    
    # stc_std=mne.read_source_estimate(file_in+'_std_vector_morph-vl.stc')
    dev1=mne.read_evokeds(file_in+'_01_otp_raw_sss_proj_fil50_evoked_dev1_mmr.fif')[0]
    dev2=mne.read_evokeds(file_in+'_01_otp_raw_sss_proj_fil50_evoked_dev2_mmr.fif')[0]
    std=mne.read_evokeds(file_in+'_01_otp_raw_sss_proj_fil50_evoked_substd_mmr.fif')[0]
   
    mmr1 = dev1.data - std.data
    mmr2 = dev2.data - std.data
    
    # group_std.append(stc_std.data)
    group_mmr1.append(mmr1)
    group_mmr2.append(mmr2)
    
group_mmr1=np.asarray(group_mmr1)
group_mmr2=np.asarray(group_mmr2)
np.save(root_path + 'cbsA_meeg_analysis/group_mmr1_sensor.npy',group_mmr1)
np.save(root_path + 'cbsA_meeg_analysis/group_mmr2_sensor.npy',group_mmr2)


#%% output the source time series in npy files
group_mmr1 = []
group_mmr2 = []
group_mmr1_roi = []
group_mmr2_roi = []
group_std = []
group_dev1 = []
group_dev2 = []
group_std_roi = []
group_dev1_roi = []
group_dev2_roi = []

# #extract ROIS for morphing data
src = mne.read_source_spaces('/media/tzcheng/storage2/subjects/fsaverage/bem/fsaverage-vol-5-src.fif') # for morphing data
fname_aseg = subjects_dir + 'fsaverage' + '/mri/aparc+aseg.mgz'

for s in subj:
    print('Extracting ' + s + ' data')
    file_in = root_path + s + '/sss_fif/' + s
    
    # stc_std=mne.read_source_estimate(file_in+'_std_None_morph-vl.stc')
    # stc_dev1=mne.read_source_estimate(file_in+'_dev1_None_morph-vl.stc')
    # stc_dev2=mne.read_source_estimate(file_in+'_dev2_None_morph-vl.stc')
    stc_mmr1=mne.read_source_estimate(file_in+'_mmr1_mba_None_morph-vl.stc')
    stc_mmr2=mne.read_source_estimate(file_in+'_mmr2_pa_None_morph-vl.stc')
    # group_std.append(stc_std.data)
    # group_dev1.append(stc_dev1.data)
    # group_dev2.append(stc_dev2.data)
    group_mmr1.append(stc_mmr1.data)
    group_mmr2.append(stc_mmr2.data)

    # #extract ROIS for non-morphing data
    # src = mne.read_source_spaces(file_in +'_src')
    # fname_aseg = subjects_dir + s + '/mri/aparc+aseg.mgz'
    
    label_names = mne.get_volume_labels_from_aseg(fname_aseg)
    
    # std_roi=mne.extract_label_time_course(stc_std,fname_aseg,src,mode='mean',allow_empty=True)
    # dev1_roi=mne.extract_label_time_course(stc_dev1,fname_aseg,src,mode='mean',allow_empty=True)
    # dev2_roi=mne.extract_label_time_course(stc_dev2,fname_aseg,src,mode='mean',allow_empty=True)

    mmr1_roi=mne.extract_label_time_course(stc_mmr1,fname_aseg,src,mode='mean',allow_empty=True)
    mmr2_roi=mne.extract_label_time_course(stc_mmr2,fname_aseg,src,mode='mean',allow_empty=True)
    
    # group_std_roi.append(std_roi)
    # group_dev1_roi.append(dev1_roi)
    # group_dev2_roi.append(dev2_roi)
    group_mmr1_roi.append(mmr1_roi)
    group_mmr2_roi.append(mmr2_roi)

# group_std = np.asarray(group_std)
# group_dev1 = np.asarray(group_dev1)
# group_dev2 = np.asarray(group_dev2)
# group_std_roi = np.asarray(group_std_roi)
# group_dev1_roi = np.asarray(group_dev1_roi)
# group_dev2_roi = np.asarray(group_dev2_roi)
group_mmr1=np.asarray(group_mmr1)
group_mmr2=np.asarray(group_mmr2)
# group_std_roi = np.asarray(group_std_roi)
# group_mmr1_roi=np.asarray(group_mmr1_roi)
# group_mmr2_roi=np.asarray(group_mmr2_roi)

# np.save(root_path + 'cbsA_meeg_analysis/group_std_None_morph.npy',group_std)
# np.save(root_path + 'cbsA_meeg_analysis/group_dev1_None_morph.npy',group_dev1)
# np.save(root_path + 'cbsA_meeg_analysis/group_dev2_None_morph.npy',group_dev2)
# np.save(root_path + 'cbsA_meeg_analysis/group_std_None_morph_roi.npy',group_std_roi)
# np.save(root_path + 'cbsA_meeg_analysis/group_dev1_None_morph_roi.npy',group_dev1_roi)
# np.save(root_path + 'cbsA_meeg_analysis/group_dev2_None_morph_roi.npy',group_dev2_roi)
# np.save(root_path + 'cbsA_meeg_analysis/group_std_vector_morph.npy',group_std)
# np.save(root_path + 'cbsA_meeg_analysis/group_std_vector_morph_roi.npy',group_std_roi)
np.save(root_path + 'cbsA_meeg_analysis/MEG/group_mmr1_mba_None_morph.npy',group_mmr1)
np.save(root_path + 'cbsA_meeg_analysis/MEG/group_mmr2_pa_None_morph.npy',group_mmr2)
np.save(root_path + 'cbsA_meeg_analysis/MEG/group_mmr1_mba_None_morph_roi.npy',group_mmr1_roi)
np.save(root_path + 'cbsA_meeg_analysis/MEG/group_mmr2_pa_None_morph_roi.npy',group_mmr2_roi)

#%% whole brain stats
X = mmr1-mmr2
Xt=np.transpose(X,[0,2,1])

from mne import spatial_src_adjacency
from mne.stats import spatio_temporal_cluster_1samp_test

src=mne.read_source_spaces('/media/tzcheng/storage2/subjects/fsaverage/bem/fsaverage-vol-5-src.fif')
print('Computing adjecency')
adjecency = spatial_src_adjacency(src)

#    Now let's actually do the clustering. This can take a long time...
#    Here we set the threshold quite high to reduce computation.
# p_threshold = 0.01
# t_threshold=-stats.distributions.t.ppf(p_threshold/2.,35-1)
# print('Clustering.')

T_obs, clusters, cluster_p_values, H0 = clu =\
    spatio_temporal_cluster_1samp_test(Xt, adjacency=adjecency, n_jobs=4,threshold=dict(start=0,step=0.5), buffer_size=None,n_permutations=512)
#    Now select the clusters that are sig. at p < 0.05 (note that this value
#    is multiple-comparisons corrected).
#good_cluster_inds = np.where(cluster_p_values < 0.05)[0]

np.save('/media/tzcheng/storage/CBS/cbsA_meeg_analysis/tfce_p_values',cluster_p_values)
np.save('/media/tzcheng/storage/CBS/cbsA_meeg_analysis/tfce_t',T_obs)
np.save('/media/tzcheng/storage/CBS/cbsA_meeg_analysis/tfce_h0',H0)

#%% visualize clusters
stc=mne.read_source_estimate('/CBS/cbs_A101/sss_fif/cbs_A101_mmr1-vl.stc')
cluster_p_values=-np.log10(cluster_p_values)
stc.data=cluster_p_values.reshape(3501,14629).T

#lims=[0, 0.025, 0.05]
lims=[1.5, 2, 2.5]
kwargs=dict(src=src, subject='fsaverage',subjects_dir=subjects_dir)
#stc_to.data=1-cluster_p_values.reshape(701,14629).T
brain=stc.plot_3d(clim=dict(kind='value',pos_lims=lims),hemi='both',views=['axial'],size=(600,300),view_layout='horizontal',show_traces=0.5,**kwargs)

#%% ROIs
#label info
atlas = 'aparc' # aparc, aparc.a2009s
fname_aseg = '/mnt/subjects/fsaverage/mri/'+atlas+'+aseg.mgz'
label_names = mne.get_volume_labels_from_aseg(fname_aseg)

l_stg=label_names.index('ctx-lh-superiortemporal')
r_stg=label_names.index('ctx-rh-superiortemporal')
l_ifg=[label_names.index('ctx-lh-parsopercularis'),label_names.index('ctx-lh-parsorbitalis'),label_names.index('ctx-lh-parstriangularis')]
r_ifg=[label_names.index('ctx-rh-parsopercularis'),label_names.index('ctx-rh-parsorbitalis'),label_names.index('ctx-rh-parstriangularis')]

mmr1_roi=np.load('/mnt/CBS/meg_mmr_analysis/group_mmr1_roi.npy')
mmr2_roi=np.load('/mnt/CBS/meg_mmr_analysis/group_mmr2_roi.npy')

#%% plot ROI by contrast
plt.figure(figsize=(12,12))
plt.subplot(2,2,1)
plot_err(mmr1_roi[:,l_stg,:],'b')
plot_err(mmr2_roi[:,l_stg,:],'r')
plt.xlabel('Time(ms)',fontsize=18)
plt.ylabel('dSPM Value',fontsize=18)
plt.ylim((-0,10))
#plt.legend(['Nonnative','Native'],prop={'size':20},loc='lower right')
plt.title('LEFT Temporal',fontsize=20)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)


plt.subplot(2,2,3)
plot_err(np.mean(mmr1_roi[:,l_ifg,:],axis=1),'b')
plot_err(np.mean(mmr2_roi[:,l_ifg,:],axis=1),'r')
plt.xlabel('Time(ms)',fontsize=18)
plt.ylabel('dSPM Value',fontsize=18)
plt.ylim((-0,10))
#plt.legend(['Nonnative','Native'],prop={'size':20},loc='lower right')
plt.title('LEFT Frontal',fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.subplot(2,2,2)
plot_err(mmr1_roi[:,r_stg,:],'b')
plot_err(mmr2_roi[:,r_stg,:],'r')
plt.xlabel('Time(ms)',fontsize=18)
plt.ylabel('dSPM Value',fontsize=18)
plt.ylim((-0,10))
#plt.legend(['Nonnative','Native'],prop={'size':20},loc='lower right')
plt.title('RIGHT Temporal',fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.subplot(2,2,4)
plot_err(np.mean(mmr1_roi[:,r_ifg,:],axis=1),'b')
plot_err(np.mean(mmr2_roi[:,r_ifg,:],axis=1),'r')
plt.xlabel('Time(ms)',fontsize=18)
plt.ylabel('dSPM Value',fontsize=18)
plt.ylim((-0,10))
#plt.legend(['Nonnative','Native'],prop={'size':20},loc='lower right')
plt.title('RIGHT Frontal',fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
