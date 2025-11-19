#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 13:58:36 2025
Extract misc channels for the same preprocessing steps as M/EEG for the CBS files
@author: tzcheng
"""
import mne
import mnefun
import matplotlib
from mne.preprocessing import maxwell_filter
import numpy as np
import os
import copy
import random

def do_filtering(data, lp, hp, do_cabr):
    ###### filtering
    if do_cabr == True:
        data.notch_filter(np.arange(60,2001,60),filter_length='auto',notch_widths=0.5)
        data.filter(l_freq=hp,h_freq=lp,method='iir',iir_params=dict(order=4,ftype='butter'))
    else:
        data.filter(l_freq=0,h_freq=50,method='iir',iir_params=dict(order=4,ftype='butter'))
    return data

def do_epoch_cabr(data, s, run,n_trials,hp,lp): 
    random.seed(15) # add for replication
    ###### Read the event files (generated from evtag.py) 
    root_path = os.getcwd()
    cabr_events = mne.read_events(root_path + '/' + s + '/events/' + s + run + '_events_cabr-eve.fif')
    file_out = root_path + '/' + s + '/sss_fif/' + s + run + '_otp_raw_sss_proj_f' + str(hp) + str(lp)
    
    event_id = {'Standardp':1,'Standardn':2, 'Deviant1p':3,'Deviant1n':5, 'Deviant2p':6,'Deviant2n':7}
    
    picks = mne.pick_types(data.info,misc=True) 
    epochs = mne.Epochs(data, cabr_events, event_id,tmin =-0.02, tmax=0.2, baseline=(-0.02,0),picks=picks)
    new_epochs = epochs.copy().drop_bad()
    
    ## match the trial number for each sound
    ## get random number of sounds from all sounds
    ## neet to find p and n len after dropping bad, use the smaller one to be the full len
    if n_trials == 'all':
        evoked_substd=epochs['Standardp','Standardn'].average(picks='MISC001')
        evoked_dev1=epochs['Deviant1p','Deviant1n'].average(picks='MISC001')
        evoked_dev2=epochs['Deviant2p','Deviant2n'].average(picks='MISC001')
    else:
        rand_ind = random.sample(range(min(len(new_epochs['Standardp'].events),len(new_epochs['Standardn'].events))),n_trials//2) 
        evoked_substd_p=epochs['Standardp'][rand_ind].average(picks='MISC001')
        evoked_substd_n=epochs['Standardn'][rand_ind].average(picks='MISC001')
        evoked_substd = mne.combine_evoked([evoked_substd_p,evoked_substd_n], weights='equal')
        del rand_ind
    
        rand_ind = random.sample(range(min(len(new_epochs['Deviant1p'].events),len(new_epochs['Deviant1n'].events))),n_trials//2) 
        evoked_dev1_p=new_epochs['Deviant1p'][rand_ind].average(picks='MISC001')
        evoked_dev1_n=new_epochs['Deviant1n'][rand_ind].average(picks='MISC001')
        evoked_dev1 = mne.combine_evoked([evoked_dev1_p,evoked_dev1_n], weights='equal')
        del rand_ind
    
        rand_ind = random.sample(range(min(len(new_epochs['Deviant2p'].events),len(new_epochs['Deviant2n'].events))),n_trials//2) 
        evoked_dev2_p=new_epochs['Deviant2p'][rand_ind].average(picks='MISC001')
        evoked_dev2_n=new_epochs['Deviant2n'][rand_ind].average(picks='MISC001')
        evoked_dev2 = mne.combine_evoked([evoked_dev2_p,evoked_dev2_n], weights='equal')
        
    epochs.save(file_out + '_misc_e_' + str(n_trials) + '.fif',overwrite=True)
    evoked_substd.save(file_out + '_evoked_substd_misc_' + str(n_trials) + '.fif',overwrite=True)
    evoked_dev1.save(file_out + '_evoked_dev1_misc_' + str(n_trials) + '.fif',overwrite=True)
    evoked_dev2.save(file_out + '_evoked_dev2_misc_' + str(n_trials) + '.fif',overwrite=True)

#%%########################################
## parameters 
run = ['_01']
lp = 200 # try 200 (suggested by Nike) or 450 (from Coffey paper)
hp = 80
do_cabr = True # True: use the cABR filter, cov and epoch setting; False: use the MMR filter, cov and epoch setting
n_trials = 200

## path
root_path='/media/tzcheng/storage2/CBS/'
os.chdir(root_path)
subj = [] # A104 got some technical issue
for file in os.listdir():
    # if file.startswith('cbs_b'): # cbs_A for the adults and cbs_b for the infants
    if file.startswith('cbs_A'): # brainstem
        subj.append(file)
        
print(subj)
for s in subj:
    print(s)
    filename = root_path + s + '/sss_fif/' + s + run[0] + '_otp_raw_sss_proj.fif'
    raw = mne.io.read_raw_fif(filename, allow_maxshield=True,preload=True)
    print ('Doing filtering...')
    raw_filt = do_filtering(raw,lp,hp,do_cabr)
    print ('Doing epoch...')
    do_epoch_cabr(raw_filt, s, run[0], n_trials,hp,lp)
    
## Get the misc channel recording from all subj (essentially the "group" function)
group_ba = []
group_mba = []
group_pa = []

for s in subj:    
    print(s)
    file_in = root_path + s + '/sss_fif/' + s + run[0]
    std = mne.read_evokeds(file_in + '_otp_raw_sss_proj_f' + str(hp) + str(lp) + '_evoked_substd_misc_200.fif')[0]
    dev1 = mne.read_evokeds(file_in + '_otp_raw_sss_proj_f' + str(hp) + str(lp) + '_evoked_dev1_misc_200.fif')[0]
    dev2 = mne.read_evokeds(file_in + '_otp_raw_sss_proj_f' + str(hp) + str(lp) + '_evoked_dev2_misc_200.fif')[0]
    group_ba.append(std.data)
    group_mba.append(dev1.data)
    group_pa.append(dev2.data)
group_ba = np.squeeze(np.asarray(group_ba))
group_mba = np.squeeze(np.asarray(group_mba))
group_pa = np.squeeze(np.asarray(group_pa))
np.save(root_path + 'cbsA_meeg_analysis/misc/adult_group_substd_misc_200.npy',group_ba)
np.save(root_path + 'cbsA_meeg_analysis/misc/adult_group_dev1_misc_200.npy',group_mba)
np.save(root_path + 'cbsA_meeg_analysis/misc/adult_group_dev2_misc_200.npy',group_pa)