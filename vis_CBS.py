#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 16:47:33 2023

@author: tzcheng
"""
## Import library  
import mne
import matplotlib
import matplotlib.pyplot as plt 
import numpy as np
import os
from nilearn.plotting import plot_glass_brain
from scipy import stats,signal

def plot_err(group_stc,color,t):
    group_avg=np.mean(group_stc,axis=0)
   #plt.figure()
    err=np.std(group_stc,axis=0)/np.sqrt(group_stc.shape[0])
    up=group_avg+err
    lw=group_avg-err
    t=np.linspace(-100,600,3501)
    plt.plot(t,group_avg,color=color)
    plt.fill_between(t,up,lw,color=color,alpha=0.5)

#%%#######################################   
root_path='/media/tzcheng/storage/CBS/'
os.chdir(root_path)
subjects_dir = '/media/tzcheng/storage2/subjects/'

runs = ['_01','_02']
times = ['_t1','_t2']
subj = [] 
for file in os.listdir():
    if file.startswith('cbs_A'):
        subj.append(file)

run = runs[0]
time = times[0]
s = subj[2]

subject = s

#%%####################################### visualize individual sensor and source
#%% before morphing from evoked data
file_in = root_path + '/' + s + '/sss_fif/' + s 
fwd = mne.read_forward_solution(file_in + '-fwd.fif')
cov = mne.read_cov(file_in + run + '_erm_otp_raw_sss_proj_fil50-cov.fif')
proj = mne.io.read_raw_fif('/media/tzcheng/storage2/SLD/MEG/sld_101/sss_fif/sld_101_t1_01_otp_raw_sss_proj.fif')
## Visualize sensor level
## MMR
epoch = mne.read_epochs(file_in + run + '_otp_raw_sss_proj_fil50_mmr_e.fif')

evoked_s = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_fil50_evoked_substd_mmr.fif')[0]
stc1 = mne.read_source_estimate(root_path + 'cbs_A101/sss_fif/cbs_A101_mmr2_morph-vl.stc')[0]
evoked_d1 = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_fil50_evoked_dev1_mmr.fif')[0]
evoked_d2 = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_fil50_evoked_dev2_mmr.fif')[0]

# use these three lines to check for the evoked responses for babies
for i in np.arange(0,len(subj),1):
    s = subj[i]
    proj = mne.io.read_raw_fif('/media/tzcheng/storage2/SLD/MEG/sld_101/sss_fif/sld_101_t1_01_otp_raw_sss_proj.fif')
    evoked_s = mne.read_evokeds(root_path + '/' + s + '/sss_fif/' + s + '_t1_01_otp_raw_sss_proj_fil50_evoked_substd_mmr.fif')[0]
    evoked_s.plot_topomap(ch_type='mag',times=np.linspace(0, 0.5, 6),colorbar=True)
    evoked_s.plot(picks='mag',gfp=True)

evoked_s.plot_joint()
mne.viz.plot_compare_evokeds(evoked_s, picks=["MEG0721"], combine="mean")
mne.viz.plot_compare_evokeds(evoked_s, picks="meg", axes="topo") # plot all of them
epoch.plot_sensors(kind='3d', ch_type='mag', ch_groups='position')
chs = ["MEG0721","MEG0631","MEG0741","MEG1821"]
proj = mne.io.read_raw_fif('/media/tzcheng/storage2/SLD/MEG/sld_101/sss_fif/sld_101_t1_01_otp_raw_sss_proj.fif')
mne.viz.plot_compare_evokeds(evoked_s, picks=chs, combine="mean", show_sensors="upper right")

## FFR
epoch = mne.read_epochs(file_in + run + '_otp_raw_sss_proj_fil50_cABR_e.fif')
evoked_s = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_fil50_evoked_substd_cabr.fif')[0]
evoked_d1 = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_fil50_evoked_dev1_cabr.fif')[0]
evoked_d2 = mne.read_evokeds(file_in + run + '_otp_raw_sss_proj_fil50_evoked_dev2_cabr.fif')[0]
evoked_s.crop(tmin=-0.1, tmax=0.2)
evoked_d1.crop(tmin=-0.1, tmax=0.2)
evoked_d2.crop(tmin=-0.1, tmax=0.2)
mne.viz.plot_compare_evokeds(evoked_s, picks="meg", axes="topo") # plot all of them

## Visualize source level
inverse_operator = mne.minimum_norm.make_inverse_operator(epoch.info, fwd, cov,loose=1,depth=0.8)

snr = 3.0
stc_epoch = mne.minimum_norm.apply_inverse_epochs(
    epoch,
    inverse_operator,
    lambda2=1.0 / snr**2,
    verbose=False,
    method="MNE",
    pick_ori=None,proj = mne.io.read_raw_fif('/media/tzcheng/storage2/SLD/MEG/sld_101/sss_fif/sld_101_t1_01_otp_raw_sss_proj.fif')
)

stc_std = mne.minimum_norm.apply_inverse((evoked_s), inverse_operator, pick_ori='vector')
stc_dev1 = mne.minimum_norm.apply_inverse((evoked_d1), inverse_operator, pick_ori='vector')
stc_dev2 = mne.minimum_norm.apply_inverse((evoked_d2), inverse_operator, pick_ori='vector')
src = inverse_operator['src']

mmr1 = stc_dev1 - stc_std
mmr2 = stc_dev2 - stc_std

mmr1.plot(src,subject=subject, subjects_dir=subjects_dir)
mmr2.plot(src,subject=subject, subjects_dir=subjects_dir)

#%% after morphing
stc = mne.read_source_estimate(root_path + s + '/sss_fif/' + s + '_mmr2_morph-vl.stc')
src = mne.read_source_spaces(subjects_dir + '/' + s + '/bem/' + s + '-vol-5-src.fif')
# src = mne.read_source_spaces(root_path + s + '/sss_fif/' + s +'_src')  # similar
src = mne.read_source_spaces('/media/tzcheng/storage2/subjects/fsaverage/bem/fsaverage-vol-5-src.fif')

initial_time = 0.1
brain = stc.plot(
    src,
    subject='fsaverage',
    subjects_dir=subjects_dir,
    initial_time=initial_time,
    mode='glass_brain'
)

#%%######################################## visualize the group level source average for ADULTS
# load the data
stc1 = mne.read_source_estimate(root_path + 'cbs_A101/sss_fif/cbs_A101_mmr2_morph-vl.stc')
stc2 = mne.read_source_estimate(root_path + 'cbs_A101/sss_fif/cbs_A101_mmr2_morph-vl.stc')
MEG_mmr1_v = np.load(root_path + 'cbsA_meeg_analysis/group_mmr1_ba_vector_morph.npy') # with the mag or vector method
MEG_mmr2_v = np.load(root_path + 'cbsA_meeg_analysis/group_mmr2_ba_vector_morph.npy') # with the mag or vector method
MEG_mmr1_m = np.load(root_path + 'cbsA_meeg_analysis/group_mmr1_ba_None_morph.npy') # with the mag or vector method
MEG_mmr2_m = np.load(root_path + 'cbsA_meeg_analysis/group_mmr2_ba_None_morph.npy') # with the mag or vector method
EEG_mmr1 = np.load(root_path + 'cbsA_meeg_analysis/group_mmr1_ba_eeg.npy')
EEG_mmr2 = np.load(root_path + 'cbsA_meeg_analysis/group_mmr2_ba_eeg.npy')
stc1.data = MEG_mmr1_v.mean(axis=0)
stc2.data = MEG_mmr2_v.mean(axis=0)

# stcs = stc1.in_label(label = label_names[0], mri = fname_aseg, src = src) #restrict to stc to one of the label 
# stc.plot(src,clim=dict(kind="value",pos_lims=[0,2,5]),subject=subject, subjects_dir=subjects_dir)
# stc1.plot(src,subject=subject, subjects_dir=subjects_dir, bg_img='aparc+aseg.mgz', mode="glass_brain")
# stc2.plot(src,subject=subject, subjects_dir=subjects_dir, bg_img='aparc+aseg.mgz')

subject = 'fsaverage'
src = mne.read_source_spaces(subjects_dir + subject + '/bem/fsaverage-vol-5-src.fif')

## stc of ROI
fname_aseg = subjects_dir + subject + '/mri/aparc+aseg.mgz'
label_names = np.asarray(mne.get_volume_labels_from_aseg(fname_aseg))
label_names

label_tc_mmr1=mne.extract_label_time_course(stc1,fname_aseg,src,mode='mean',allow_empty=True)
label_tc_mmr2=mne.extract_label_time_course(stc2,fname_aseg,src,mode='mean',allow_empty=True)

# averaged whole brain mmr plot
plt.figure()
times = stc1.times
plot_err(stats.zscore(MEG_mmr1_v.mean(axis=1),axis=1),'r',stc1.times)
plot_err(stats.zscore(MEG_mmr2_v.mean(axis=1),axis=1),'r',stc1.times)
plot_err(stats.zscore(MEG_mmr1_m.mean(axis=1),axis=1),'b',stc1.times)
plot_err(stats.zscore(MEG_mmr2_m.mean(axis=1),axis=1),'b',stc1.times)
plot_err(stats.zscore(EEG_mmr1,axis=1),'k',stc1.times)
plot_err(stats.zscore(EEG_mmr2,axis=1),'k',stc1.times)
plt.legend(['MMR1 vector','','MMR2 vector','','MMR1 mag','','MMR2 mag',''])
plt.xlim([-100,600])
plt.xlabel('Time (ms)')
plt.ylabel('zscore')

# ROI mmr plot
lh_ROI_label = [60,61,62,72] # STG and IFG (parsopercularis, parsorbitalis, parstriangularis)
rh_ROI_label = [96,97,98,108] # STG and IFG (parsopercularis, parsorbitalis, parstriangularis)

plt.figure()
plt.plot(times,label_tc_mmr2[lh_ROI_label,].transpose())
plt.title('CBS_A MMR2 lh')
plt.legend(label_names[lh_ROI_label])
plt.xlabel('Time (s)')
plt.ylabel('Activation (AU)')
plt.xlim([-0.1, 0.5])
#    plt.savefig('/home/tzcheng/Desktop/' + 'run1_' + label[Qis_ROI_label[i]].name +'.pdf')

plt.figure()proj = mne.io.read_raw_fif('/media/tzcheng/storage2/SLD/MEG/sld_101/sss_fif/sld_101_t1_01_otp_raw_sss_proj.fif')
plt.plot(times,label_tc_mmr2[rh_ROI_label,].transpose())
plt.title('CBS_A MMR2 rh')
plt.legend(label_names[rh_ROI_label])
plt.xlabel('Time (s)')
plt.ylabel('Activation (AU)')
plt.xlim([-0.1, 0.5])
#    plt.savefig('/home/tzcheng/Desktop/' + 'run1_' + label[Qis_ROI_label[i]].name +'.pdf')

#%%######################################## visualize the group level source average for INFANTS
stc = mne.read_source_estimate(root_path + 'cbs_b106/sss_fif/cbs_b106_mmr2_None-vl.stc')

MEG_mmr1_roi_v = np.load(root_path + 'cbsb_meg_analysis/group_mmr1_ba_vector_roi.npy') # with the mag or vector method
MEG_mmr2_roi_v = np.load(root_path + 'cbsb_meg_analysis/group_mmr2_ba_vector_roi.npy') # with the mag or vector method
MEG_mmr1_roi_m = np.load(root_path + 'cbsb_meg_analysis/group_mmr1_roi.npy') # with the mag or vector method
MEG_mmr2_roi_m = np.load(root_path + 'cbsb_meg_analysis/group_mmr2_roi.npy') # with the mag or vector method

MEG_mmr1_roi_v_mean = MEG_mmr1_roi_v.mean(axis=0)
MEG_mmr2_roi_v_mean = MEG_mmr2_roi_v.mean(axis=0)
MEG_mmr1_roi_m_mean = MEG_mmr1_roi_m.mean(axis=0)
MEG_mmr2_roi_m_mean = MEG_mmr2_roi_m.mean(axis=0)

## stc of ROI
fname_aseg = subjects_dir + subject + '/mri/aparc+aseg.mgz'
label_names = np.asarray(mne.get_volume_labels_from_aseg(fname_aseg))
label_names

# averaged whole brain mmr plot
times = stc.times
plt.figure()
plot_err(MEG_mmr2_roi_m.mean(axis=1),'b',stc1.times)
plot_err(MEG_mmr2_roi_v.mean(axis=1),'r',stc1.times)
plt.legend(['MEG mag','','MEG vector',''])
plt.title('mmr 2 averaged across ROIs')
plt.xlabel('Time (ms)')
plt.ylabel('zscore')
plt.xlim([-100, 500])

# ROI mmr plot
lh_ROI_label = [49,50,51,61] # STG and IFG (parsopercularis, parsorbitalis, parstriangularis)
rh_ROI_label = [84,85,86,96] # STG and IFG (parsopercularis, parsorbitalis, parstriangularis)

plt.figure()
plt.plot(times,MEG_mmr2_roi_v_mean[lh_ROI_label,].transpose())
plt.title('CBS_b MMR2 lh')
plt.legend(label_names[lh_ROI_label])
plt.xlabel('Time (ms)')
plt.ylabel('Activation (AU)')
plt.xlim([-0.1, 0.5])
#    plt.savefig('/home/tzcheng/Desktop/' + 'run1_' + label[Qis_ROI_label[i]].name +'.pdf')

plt.figure()
plt.plot(times,MEG_mmr2_roi_v_mean[rh_ROI_label,].transpose())
plt.title('CBS_b MMR2 rh')
plt.legend(label_names[rh_ROI_label])
plt.xlabel('Time (ms)')
plt.ylabel('Activation (AU)')
plt.xlim([-0.1, 0.5])
#    plt.savefig('/home/tzcheng/Desktop/' + 'run1_' + label[Qis_ROI_label[i]].name +'.pdf')

#%%######################################## visualize ECG/EOG projection for the bad babies
evoked_s = mne.read_evokeds('/media/tzcheng/storage2/SLD/MEG/sld_101/sss_fif/sld_101_t1_01_otp_raw_sss_proj_fil50_evoked_substd_mmr.fif')[0]
evoked_s.plot_topomap(ch_type='mag',times=np.linspace(0, 0.5, 6),colorbar=True)
evoked_s.plot(picks='grad',gfp=True)
proj = mne.io.read_raw_fif('/media/tzcheng/storage2/SLD/MEG/sld_101/sss_fif/sld_101_t1_01_otp_raw_sss_proj.fif')
proj.plot_projs_topomap()