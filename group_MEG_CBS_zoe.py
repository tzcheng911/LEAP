#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 17:57:34 2023

@author: tzcheng
"""

import mne
import numpy as np
import matplotlib.pyplot as plt

def plot_err(group_stc,color):
    group_avg=np.mean(group_stc,axis=0)
   #plt.figure()
    err=np.std(group_stc,axis=0)/np.sqrt(group_stc.shape[0])
    up=group_avg+err
    lw=group_avg-err
    t=np.linspace(-100,600,700)
    plt.plot(t,group_avg,color=color)
    plt.fill_between(t,up,lw,color=color,alpha=0.5)
    
########################################
root_path='/media/tzcheng/storage/CBS/'
os.chdir(root_path)

## parameters 
runs = ['_01','_02']
st_correlation = 0.98 # 0.98 for adults and 0.9 for infants
int_order = 8 # 8 for adults and 6 for infants
lp = 50 
subj = [] # A104 got some technical issue
for file in os.listdir():
    if file.startswith('cbs_A'):
        subj.append(file)

runs = ['01','02']


run = runs [1]
#%%
group_mmr1=[]
group_mmr2=[]
group_mmr1_roi=[]
group_mmr2_roi=[]

for s in subj:
    stc_mmr1=mne.read_source_estimate('/mnt/CBS/'+str(s)+'/sss_fif/'+str(s)+'_dev1_mmr-vl.stc')
    stc_mmr2=mne.read_source_estimate('/mnt/CBS/'+str(s)+'/sss_fif/'+str(s)+'_dev2_mmr-vl.stc')
   
    group_mmr1.append(stc_mmr1.data)
    group_mmr2.append(stc_mmr2.data)
    # #extract ROIS
    src = mne.read_source_spaces('/mnt/subjects/' + str(s) + '/bem/' + str(s) +'-vol-5-src.fif')
    labels = '/mnt/subjects/' + str(s) + '/mri/aparc+aseg.mgz'
    mmr1_roi=mne.extract_label_time_course(stc_mmr1,labels,src,mode='mean',allow_empty=True)
    mmr2_roi=mne.extract_label_time_course(stc_mmr2,labels,src,mode='mean',allow_empty=True)
    
    group_mmr1_roi.append(mmr1_roi)
    group_mmr2_roi.append(mmr2_roi)

    
group_mmr1=np.asarray(group_mmr1)
group_mmr2=np.asarray(group_mmr2)

group_mmr1_roi=np.asarray(group_mmr1_roi)
group_mmr2_roi=np.asarray(group_mmr2_roi)

np.save('/mnt/CBS/meg_mmr_analysis/group_dev1.npy',group_mmr1)
np.save('/mnt/CBS/meg_mmr_analysis/group_dev2.npy',group_mmr2)
np.save('/mnt/CBS/meg_mmr_analysis/group_dev1_roi.npy',group_mmr1_roi)
np.save('/mnt/CBS/meg_mmr_analysis/group_dev2_roi.npy',group_mmr2_roi)

#%%
#averaged acrossed fsaverage
mmr1=np.load('/mnt/CBS/meg_mmr_analysis/group_dev1.npy')
mmr2=np.load('/mnt/CBS/meg_mmr_analysis/group_dev2.npy')

# whole brain mmr plot
plot_err(np.mean(mmr1,1),'b')
plot_err(np.mean(mmr2,1),'r')
plt.xlabel('Time (ms)', fontsize=18)
plt.ylabel('dSPM value', fontsize=18)
plt.title('MMR wholebrain',fontsize=18)

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

stc=mne.read_source_estimate('/mnt/CBS/cbs_101/sss_fif/cbs_A101_mmr1-vl.stc')
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
