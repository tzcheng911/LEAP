#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Simulate a brainstem source and localize it using a dipole fit.
"""
from functools import partial
import os
import matplotlib.pyplot as plt
import numpy as np
import mne
from mne.minimum_norm import apply_inverse, make_inverse_operator
from mne.simulation import simulate_raw, simulate_evoked
from mne.simulation.metrics import (
    cosine_score,
    f1_score,
    peak_position_error,
    precision_score,
    recall_score,
    region_localization_error,
    spatial_deviation_error,
)

# def do_filtering(raw_data):
#     # raw_data.notch_filter(np.arange(60,500,60),filter_length='auto',notch_widths=0.5)
#     # band-pass filter between 0 and 50 Hz as in preprocessing_ME2.py
#     raw_data.filter(l_freq=0,h_freq=50,method='iir',iir_params=dict(order=4,ftype='butter'))
#     # spectrum plotting to check no usual frequency on the spectrum
#     raw_data.plot_psd()
#     return raw_data

template_subject = 'ANTS6-0Months3T'
label_list = ['Left-Putamen','Left-Caudate','Right-Putamen','Right-Caudate']

root_path = '/media/tzcheng/storage/ME2_MEG/Zoe_analyses/7mo/'
subjects_dir = '/media/tzcheng/storage2/subjects/'
os.chdir(root_path)

RLE_all = []
subj = [] 
for file in os.listdir():
    if file.startswith('me2_'):
        subj.append(file)
for subject in subj:        
    fname_raw = root_path + subject + '/raw_fif/' + subject + '_03_raw.fif'
    fname_evoked= root_path + subject + '/sss_fif/' + subject + '_03_otp_raw_sss_proj_fil50_mag6pT_evoked.fif'
    fname_cov = root_path + subject + '/sss_fif/' + subject + '_03_erm_otp_raw_sss_proj_fil50-cov.fif'
    fname_trans = root_path + subject + '/sss_fif/' + subject + '_trans.fif'
    fname_fwd = root_path + subject + '/sss_fif/' + subject + '-fwd.fif'
    fname_bem = subjects_dir + template_subject + '/bem/' + template_subject + '-5120-5120-5120-bem-sol.fif'
    fname_aseg= subjects_dir + template_subject + '/mri/aparc+aseg.mgz'
    label_names = mne.get_volume_labels_from_aseg(fname_aseg)
    
    raw = mne.io.Raw(fname_raw, allow_maxshield=True)
    raw.load_data().pick_types(meg=True, eeg=False, stim=True)
    
    ## Choose a BG source
    print('Creating a BG source')
    src_BG = mne.setup_volume_source_space(
        template_subject, mri = fname_aseg, pos=5.0, bem=fname_bem,
        volume_label = label_list, single_volume = True, add_interpolator=True)
    vertices = [src_BG[0]['vertno']]
    tmin, tmax = -0.5, 1.5 # doesn't need to be this long 
    n_samp = int(round((tmax - tmin) * raw.info['sfreq'])) + 1
    data = np.zeros([sum(len(v) for v in vertices), n_samp])
    activation = np.hanning(int(round(raw.info['sfreq'])))
    activation *= 1e-7  # nAm 
    t_offset = 0.0
    n_offset = int(round((t_offset - tmin) * raw.info['sfreq']))
    data[:, n_offset:n_offset + len(activation)] = activation
    stc = mne.VolSourceEstimate(data, vertices, tmin=tmin, tstep=1. / raw.info['sfreq'])
    # stc.plot(src=src_BG)
    # stc.plot_3d(src=src_BG,subject = template_subject)
    
    ## Simulate data
    trans = mne.read_trans(fname_trans)
    bem = mne.read_bem_solution(fname_bem)
    
    raw_stat = simulate_raw(raw.info, stc, trans, src_BG, bem, verbose=True)
    raw_stat.save(root_path + subject + '/raw_fif/' + subject + '_' + label_list[0] +'_sim_raw.fif', overwrite=True)
    
    # raw_stat = do_filtering(raw_stat)
    events = mne.find_events(raw_stat, 'STI101', verbose=False)
    events[:, 2] = 1  # all the same
    epoch = mne.Epochs(raw_stat, events, 1, tmin, tmax, verbose=False)
    evoked = epoch.average()
    evoked.plot_joint()
    
    ## Localize the simulated raw 
    noise_cov = mne.read_cov(fname_cov)  ## read the individual cov
    # noise_cov = mne.make_ad_hoc_cov(raw.info) ## which one make more sense?
    fwd =  mne.read_forward_solution(fname_fwd)  ## read the individual fwd
    
    inverse_operator = make_inverse_operator(epoch.info, fwd, noise_cov,loose=1,depth=0.8)
    stc_mne = apply_inverse((evoked), inverse_operator, pick_ori = None)
    stc_mne.plot(src=fwd['src'])
    # stc_mne.plot_3d(src=fwd['src'],subject = subject)
    
    ## Compare the simulated and ground truth source location 
    # At this particular time point, does my estimated spatial pattern look like the true spatial pattern?
    cos_similarity = mne.simulation.metrics.cosine_score(stc, stc_mne, per_sample=True)[n_offset+1]
    print('cosine similarity: ' + str(cos_similarity))
    # How far apart are the true active sources and estimated active sources in physical brain space? RLE is sensitive to spatial displacement and spatial spread
    RLE = mne.simulation.metrics.region_localization_error(stc, stc_mne, fwd['src'], threshold='50%', per_sample=True) # output in meter
    plt.figure()
    plt.plot(stc.times,RLE)
    RLE_all.append(np.min(RLE)*1000) ## in mm
