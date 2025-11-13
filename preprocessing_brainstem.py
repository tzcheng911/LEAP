#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 18:44:15 2025
Preprocessing for brainstem MEG and EEG. Need to have events file ready from evtag.py
starting after otp and sss

@author: tzcheng
"""

###### Import library 
import mne
# import mnefun
import matplotlib
from mne.preprocessing import maxwell_filter
import numpy as np
import os
import random

def do_projection(subject, condition, run):
    ###### cleaning with ecg and eog projection
    root_path = os.getcwd()
    file_in=root_path + '/' + subject + '/sss_fif/' + subject + condition + run + '_otp_raw_sss'
    file_out=file_in + '_proj'
    raw = mne.io.read_raw_fif(file_in + '.fif',allow_maxshield=True,preload=True)
    fname_erm = root_path + '/' + subject + '/sss_fif/' + subject + '_erm_otp_raw_sss'
    fname_erm_out = root_path + '/' + subject + '/sss_fif/' + subject + run + '_erm_raw_sss_proj'
    raw_erm = mne.io.read_raw_fif(fname_erm + '.fif',allow_maxshield=True,preload=True)
        
    ecg_projs, ecg_events = mne.preprocessing.compute_proj_ecg(raw, ch_name='ECG064', n_grad=1, n_mag=1, n_jobs = 4, reject=None)
    ecg_epochs = mne.preprocessing.create_ecg_epochs(raw,ch_name='ECG064').average() # don't really need to assign the ch_name
    # ecg_epochs.plot_joint()
    eog_projs, eog_events = mne.preprocessing.compute_proj_eog(raw, ch_name=['EOG061','EOG062','EOG063'], n_grad=1, n_mag=1,n_jobs = 4, reject=None)
    eog_epochs = mne.preprocessing.create_eog_epochs(raw,ch_name=['EOG061','EOG062','EOG063']).average() ## 
    # eog_epochs.plot_joint()

    raw.add_proj(ecg_projs)
    raw.add_proj(eog_projs)
    raw_erm.add_proj(ecg_projs)
    raw_erm.add_proj(eog_projs)

    raw.save(file_out + '.fif',overwrite = True)
    raw_erm.save(fname_erm_out + '.fif',overwrite = True)
    return raw, raw_erm

def do_filtering(data, lp, hp, do_cabr):
    ###### filtering
    if do_cabr == True:
        data.notch_filter(np.arange(60,2001,60),filter_length='auto',notch_widths=0.5)
        # data.notch_filter(np.arange(60,500,60),filter_length='auto',notch_widths=0.5) # for the erm files have lower sampling rates
        data.filter(l_freq=hp,h_freq=lp,method='iir',iir_params=dict(order=4,ftype='butter'))
    else:
        data.filter(l_freq=0,h_freq=50,method='iir',iir_params=dict(order=4,ftype='butter'))
    return data

def do_cov(subject,data, do_cabr,hp,lp):
    ###### noise covariance for each run based on its eog ecg proj
    root_path = os.getcwd()
    fname_erm = root_path + '/' + subject + '/sss_fif/' + subject + run + '_erm_otp_raw_sss_proj_f'
    if do_cabr == True:     
        fname_erm_out = fname_erm + str(hp) + str(lp) + '_ffr-cov'
    else: 
        fname_erm_out = fname_erm + 'il50_mmr-cov'
    noise_cov = mne.compute_raw_covariance(data, tmin=0, tmax=None)
    mne.write_cov(fname_erm_out + '.fif', noise_cov,overwrite=True)

def do_epoch_cabr(data, subject, condition, run,hp,lp): 
    root_path = os.getcwd()
    file_out = root_path + '/' + subject + '/sss_fif/' + subject + condition + run + '_otp_raw_sss_proj_f' + str(hp) + str(lp)
    cabr_events=mne.find_events(data,stim_channel=['STI101']) ###### had equal number of the positive and negative polarity (1500/block for each condition) randomized in STI003 and STI004. STI001 is just everything
    
    reject=dict(grad=4000e-13,mag=4e-12)
    picks = mne.pick_types(data.info,meg=True,eeg=False) 
    epochs = mne.Epochs(data, cabr_events, event_id=1,tmin =-0.02, tmax=0.2, baseline=(-0.02,0),reject=reject,picks=picks)
    evoked=epochs.average()  
    epochs.save(file_out + '_ffr_e.fif',overwrite=True)
    evoked.save(file_out + '_evoked_ffr.fif',overwrite=True)

    return evoked, epochs

def do_epoch_cabr_eeg(data, subject, condition, run, n_trials):  
    ###### Read the event files (generated from evtag.py) 
    root_path = os.getcwd()
    cabr_events = mne.read_events(root_path + '/' + subject + '/events/' + subject + run + '_events_cabr-eve.fif')
    file_out = root_path + '/' + subject + '/eeg/' + subject + run 
    
    event_id = {'Standardp':1,'Standardn':2, 'Deviant1p':3,'Deviant1n':5, 'Deviant2p':6,'Deviant2n':7}
    
    reject=dict(bio=35e-6)
    epochs = mne.Epochs(data, cabr_events, event_id,tmin =-0.02, tmax=0.2, baseline=(-0.02,0),reject=reject)
    new_epochs = epochs.copy().drop_bad()

    ## match the trial number for each sound
    ## get random number of sounds from all sounds
    ## neet to find p and n len after dropping bad, use the smaller one to be the full len
    if n_trials == 'all':
        evoked_substd=epochs['Standardp','Standardn'].average(picks=('bio'))
        evoked_dev1=epochs['Deviant1p','Deviant1n'].average(picks=('bio'))
        evoked_dev2=epochs['Deviant2p','Deviant2n'].average(picks=('bio'))
    else:
        rand_ind = random.sample(range(min(len(new_epochs['Standardp'].events),len(new_epochs['Standardn'].events))),n_trials//2) 
        evoked_substd_p=new_epochs['Standardp'][rand_ind].average(picks=('bio'))
        evoked_substd_n=new_epochs['Standardn'][rand_ind].average(picks=('bio'))
        evoked_substd = mne.combine_evoked([evoked_substd_p,evoked_substd_n], weights='equal')
        del rand_ind
    
        rand_ind = random.sample(range(min(len(new_epochs['Deviant1p'].events),len(new_epochs['Deviant1n'].events))),n_trials//2) 
        evoked_dev1_p=new_epochs['Deviant1p'][rand_ind].average(picks=('bio'))
        evoked_dev1_n=new_epochs['Deviant1n'][rand_ind].average(picks=('bio'))
        evoked_dev1 = mne.combine_evoked([evoked_dev1_p,evoked_dev1_n], weights='equal')
        del rand_ind
    
        rand_ind = random.sample(range(min(len(new_epochs['Deviant2p'].events),len(new_epochs['Deviant2n'].events))),n_trials//2) 
        evoked_dev2_p=new_epochs['Deviant2p'][rand_ind].average(picks=('bio'))
        evoked_dev2_n=new_epochs['Deviant2n'][rand_ind].average(picks=('bio'))
        evoked_dev2 = mne.combine_evoked([evoked_dev2_p,evoked_dev2_n], weights='equal')

    new_epochs.save(file_out + '_cabr_e_' + str(n_trials) + '.fif',overwrite=True)
    evoked_substd.save(file_out + '_evoked_substd_cabr_' + str(n_trials) + '.fif',overwrite=True)
    evoked_dev1.save(file_out + '_evoked_dev1_cabr_' + str(n_trials) + '.fif',overwrite=True)
    evoked_dev2.save(file_out + '_evoked_dev2_cabr_' + str(n_trials) + '.fif',overwrite=True)
    return evoked_substd,evoked_dev1,evoked_dev2,new_epochs

#%%########################################
mne.set_config('MNE_MEMMAP_MIN_SIZE', '10M') 
mne.set_config('MNE_CACHE_DIR', '/dev/shm')

root_path='/media/tzcheng/storage/Brainstem/' # brainstem files
os.chdir(root_path)

## parameters 
conditions = ['_p10','_n40']
runs = ['_01','_02'] 
lp = 200 # try 200 (suggested by Nike) or 450 (from Coffey paper)
hp = 80
do_cabr = True # True: use the cABR filter, cov and epoch setting; False: use the MMR filter, cov and epoch setting

subj = [] # A104 got some technical issue
for file in os.listdir():
    if file.startswith('brainstem_'): # brainstem
        subj.append(file)
# subj = ['brainstem_121','brainstem_123','brainstem_126','brainstem_129'] 
# for these four subjects empty room is sampled at 1000 Hz so the notch filter cannot go up to 2000
# run the erm files to get cov manually set to data.notch_filter(np.arange(60,500,60),filter_length='auto',notch_widths=0.5)

#%%##### do the jobs for MEG
print(subj)
for s in subj:
    print(s)
    # do_otp(s)
    # do_sss(s,st_correlation,int_order)
    for condition in conditions:
        for run in runs:
            print ('Doing ECG/EOG projection...')
            [raw,raw_erm] = do_projection(s,condition,run)
            print ('Doing filtering...')
            raw_filt = do_filtering(raw,lp,hp,do_cabr)
            raw_erm_filt = do_filtering(raw_erm,lp,hp,do_cabr)
            print ('calculate cov...')
            do_cov(s,raw_erm_filt, do_cabr,hp,lp)
            print ('Doing epoch...')
            if do_cabr == True:
                do_epoch_cabr(raw_filt, s, condition, run,hp,lp)
            else:
                print('Doing something else than cabr.')
            ## remove files with wrong filenames
            # file_in_epoch = root_path + '/' + s + '/sss_fif/' + s + condition + run + '_otp_raw_sss_proj_f' + str(hp) + str(lp) + '_ffr_e' + condition + run + '.fif'
            # file_in_evoked = root_path + '/' + s + '/sss_fif/' + s + condition + run + '_otp_raw_sss_proj_f' + str(hp) + str(lp) + '_evoked_ffr' + condition + run + '.fif'
            # epoch = mne.read_epochs(file_in_epoch)
            # evoked = mne.read_evokeds(file_in_evoked)
            # epoch.save(file_in_epoch[:-11] + '.fif',overwrite=True)
            # evoked[0].save(file_in_evoked[:-11] + '.fif',overwrite=True)
            # os.remove(file_in_epoch)
            # os.remove(file_in_epoch[:-4] + '-1.fif')
            # os.remove(file_in_evoked)

#%%##### do the jobs for EEG MMR
# for s in subj:
#     print(s)
#     for run in runs:
#         raw_file=mne.io.Raw('/media/tzcheng/storage2/CBS/'+s+'/eeg/'+s+ run +'_raw.fif',allow_maxshield=True,preload=True)
#         # raw_file = mne.io.Raw('/media/tzcheng/storage2/CBS/cbs_zoe/raw_fif/cbs_zoe' + run + '_raw.fif',allow_maxshield=True,preload=True) # for stimuli leakage test
#         raw_file.filter(l_freq=0,h_freq=50,picks=('bio'),method='iir',iir_params=dict(order=4,ftype='butter'))
#         raw_file.pick_channels(['BIO004'])
#         do_epoch_mmr_eeg(raw_file, s, run, direction)

#%%##### do the jobs for EEG FFR
# for s in subj:
#     print(s)
#     for run in runs:
#         raw_file=mne.io.Raw('/media/tzcheng/storage2/CBS/'+s+'/eeg/'+s+ run +'_raw.fif',allow_maxshield=True,preload=True)
#         # raw_file = mne.io.Raw('/media/tzcheng/storage2/CBS/cbs_zoe/raw_fif/cbs_zoe' + run + '_raw.fif',allow_maxshield=True,preload=True) # for stimuli leakage test
#         raw_file.notch_filter(np.arange(60,2001,60),filter_length='auto',notch_widths=0.5,picks=('bio'))
#         raw_file.filter(l_freq=80,h_freq=2000,picks=('bio'),method='iir',iir_params=dict(order=4,ftype='butter'))
#         raw_file.pick_channels(['BIO004'])
#         do_epoch_cabr_eeg(raw_file, s, run, n_trials)

#%%##### check the number of avg in evoked files
# nave_mmr_std = []
# nave_mmr_dev1 = []
# nave_mmr_dev2 = []
# nave_ffr_std = []
# nave_ffr_dev1 = []
# nave_ffr_dev2 = []

# for s in subj:
#     print(s)
#     for run in runs:
#         std_mmr =mne.read_evokeds('/media/tzcheng/storage2/CBS/'+s+'/eeg/'+s+ run +'_evoked_substd_mmr.fif',allow_maxshield=True)
#         dev1_mmr =mne.read_evokeds('/media/tzcheng/storage2/CBS/'+s+'/eeg/'+s+ run +'_evoked_dev1_mmr.fif',allow_maxshield=True)
#         dev2_mmr =mne.read_evokeds('/media/tzcheng/storage2/CBS/'+s+'/eeg/'+s+ run +'_evoked_dev2_mmr.fif',allow_maxshield=True)
#         std_ffr =mne.read_evokeds('/media/tzcheng/storage2/CBS/'+s+'/eeg/'+s+ run +'_evoked_substd_cabr_all.fif',allow_maxshield=True)
#         dev1_ffr =mne.read_evokeds('/media/tzcheng/storage2/CBS/'+s+'/eeg/'+s+ run +'_evoked_dev1_cabr_all.fif',allow_maxshield=True)
#         dev2_ffr =mne.read_evokeds('/media/tzcheng/storage2/CBS/'+s+'/eeg/'+s+ run +'_evoked_dev2_cabr_all.fif',allow_maxshield=True)

#         nave_mmr_std.append(std_mmr[0].nave)
#         nave_mmr_dev1.append(dev1_mmr[0].nave)
#         nave_mmr_dev2.append(dev2_mmr[0].nave)
#         nave_ffr_std.append(std_ffr[0].nave)
#         nave_ffr_dev1.append(dev1_ffr[0].nave)
#         nave_ffr_dev2.append(dev2_ffr[0].nave)

# print(np.mean(nave_mmr_std))
# print(np.mean(nave_mmr_dev1))
# print(np.mean(nave_mmr_dev2))