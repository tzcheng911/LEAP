#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 17:57:34 2023

@author: tzcheng
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_err(group_stc,color,t):
    group_avg=np.mean(group_stc,axis=0)
   #plt.figure()
    err=np.std(group_stc,axis=0)/np.sqrt(group_stc.shape[0])
    up=group_avg+err
    lw=group_avg-err
    t=np.linspace(-100,600,3501)
    plt.plot(t,group_avg,color=color)
    plt.fill_between(t,up,lw,color=color,alpha=0.5)
    
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

#%%
group_mmr1=[]
group_mmr2=[]
group_mmr1_roi=[]
group_mmr2_roi=[]

for s in subj:
    file_in = root_path + '/' + s + '/sss_fif/' + s
    

    stc_mmr1=mne.read_source_estimate(file_in+'_mmr1_vector_morph-vl.stc')
    stc_mmr2=mne.read_source_estimate(file_in+'_mmr2_vector_morph-vl.stc')
   
    group_mmr1.append(stc_mmr1.data)
    group_mmr2.append(stc_mmr2.data)
    # #extract ROIS
    # src = mne.read_source_spaces(file_in +'_src')
    src = mne.read_source_spaces('/media/tzcheng/storage2/subjects/fsaverage/bem/fsaverage-vol-5-src.fif') # for morphing data
    # labels = subjects_dir + s + '/mri/aparc+aseg.mgz'
    labels = subjects_dir + 'fsaverage' + '/mri/aparc+aseg.mgz'
    mmr1_roi=mne.extract_label_time_course(stc_mmr1,labels,src,mode='mean',allow_empty=True)
    mmr2_roi=mne.extract_label_time_course(stc_mmr2,labels,src,mode='mean',allow_empty=True)
    
    group_mmr1_roi.append(mmr1_roi)
    group_mmr2_roi.append(mmr2_roi)
    
    
group_mmr1=np.asarray(group_mmr1)
group_mmr2=np.asarray(group_mmr2)

group_mmr1_roi=np.asarray(group_mmr1_roi)
group_mmr2_roi=np.asarray(group_mmr2_roi)

np.save(root_path + 'meeg_mmr_analysis/group_mmr1_vector_morph.npy',group_mmr1)
np.save(root_path + 'meeg_mmr_analysis/group_mmr2_vector_morph.npy',group_mmr2)
np.save(root_path + 'meeg_mmr_analysis/group_mmr1_vector_morph_roi.npy',group_mmr1_roi)
np.save(root_path + 'meeg_mmr_analysis/group_mmr2_vector_morph_roi.npy',group_mmr2_roi)

#%%######################################## visualize the averaged 
stc1 = mne.read_source_estimate(root_path + 'cbs_A101/sss_fif/cbs_A101_mmr2_morph-vl.stc')
stc2 = stc1
mmr1 = np.load(root_path + 'meeg_mmr_analysis/group_mmr1_vector_morph.npy')
mmr2 = np.load(root_path + 'meeg_mmr_analysis/group_mmr2_vector_morph.npy')
stc1.data = mmr1.mean(axis=0)
stc2.data = mmr2.mean(axis=0)

subject = 'fsaverage'
src = mne.read_source_spaces(subjects_dir + 'fsaverage/bem/fsaverage-vol-5-src.fif')

# stc.plot(src,clim=dict(kind="value",pos_lims=[0,2,5]),subject=subject, subjects_dir=subjects_dir)
stc1.plot(src,subject=subject, subjects_dir=subjects_dir)
stc2.plot(src,subject=subject, subjects_dir=subjects_dir)

#%%
#averaged acrossed fsaverage
t = np.shape(group_mmr1)[-1]

# whole brain mmr plot
plot_err(np.mean(group_mmr1,1),'b',t)
plot_err(np.mean(group_mmr2,1),'r',t)
plt.xlabel('Time (ms)', fontsize=18)
plt.ylabel('dSPM value', fontsize=18)
plt.title('MMR wholebrain',fontsize=18)
plt.legend(['','nonnative','','native'])

#%%
# get the label name
atlas = 'aparc' # aparc, aparc.a2009s
label_names = mne.get_volume_labels_from_aseg(fname_aseg)

#%% 200-600ms average
t=np.linspace(-100,600,700)
idx = np.where((t>200) & (t<500))
mmr1_average=np.mean(np.mean(np.squeeze(mmr1[:,:,idx]),2),1)
mmr2_average=np.mean(np.mean(np.squeeze(mmr2[:,:,idx]),2),1)

#%% whole brain stats
X = mmr1-mmr2
Xt=np.transpose(X,[0,2,1])

from mne import spatial_src_adjacency
from mne.stats import spatio_temporal_cluster_1samp_test

src=mne.read_source_spaces('/mnt/subjects/fsaverage/bem/fsaverage-vol-5-src.fif')
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

np.save('/mnt/CBS/meg_mmr_analysis/tfce_p_values',cluster_p_values)
np.save('/mnt/CBS/meg_mmr_analysis/tfce_t',T_obs)
np.save('/mnt/CBS/meg_mmr_analysis/tfce_h0',H0)
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
