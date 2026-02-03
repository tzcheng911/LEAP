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
    
def find_eeg(raw_file,subject,condition,run):
    if 'EEG030' in raw_file.info['ch_names']:
        raw_file.pick_channels(['EEG030','MISC001','STI001','STI003','STI004','STI101'])
        print("EEG030 for " + subject)
    else:
        raw_file.pick_channels(['EEG034','MISC001','STI001','STI003','STI004','STI101'])
        print("EEG034 for " + subject)
    raw_file.save('/media/tzcheng/storage/Brainstem/EEG/raw/' + subject + condition + run + '_raw.fif',overwrite=True)
    return raw_file

def find_events(raw_file):
    STI1 = mne.find_events(raw_file,stim_channel='STI001') 
    STI3 = mne.find_events(raw_file,stim_channel='STI003')
    STI4 = mne.find_events(raw_file,stim_channel='STI004')
    STI1[:,2] = 1 # 3000
    STI3[:,2] = 4 # 1500 
    STI4[:,2] = 8 # 1500
    events_raw = np.concatenate((STI1,STI3,STI4),axis=0)
    events_raw = events_raw[events_raw[:,0].argsort()] # sort by the latency
    
    events = events_raw
    e=[]
    for event in events_raw:
        e.append(event[2])
    ## dealing with the event codes
    ind1=[i for i, x in enumerate(e) if x==1] ## find all the indices for 1 because it marks the precise timing
    
    # should always be 4-1-8-1-4-...etc. the 4 and 8 mark the polarity, 1 marks the accurate timing
    for i in ind1:
        if e[i-1]==4:
             events[i][2] = 44 # use this to do the epoch
        elif e[i-1]==8:
            events[i][2] = 88 # use this to do the epoch
    return events

def do_epoch_cabr_meg(data, events, subject, condition, run, n_trials, hp,lp): 
    root_path = os.getcwd()
    file_out = root_path + '/' + subject + '/sss_fif/' + subject + condition + run + '_otp_raw_sss_proj_f' + str(hp) + str(lp) + '_ntrial' + str(n_trials)
    
    reject=dict(grad=4000e-13,mag=4e-12)
    picks = mne.pick_types(data.info,meg=True,eeg=False) 
    epochs = mne.Epochs(data, events, event_id = [44,88], tmin =-0.02, tmax=0.2, baseline=(-0.02,0),reject=reject,picks=picks)
    new_epochs = epochs.copy().drop_bad()
    if n_trials == 'all':
        evoked=new_epochs.average()
    else:
        rand_ind = random.Random().sample(range(min(len(new_epochs['44'].events),len(new_epochs['88'].events))),n_trials//2) 
        evoked_p=new_epochs['44'][rand_ind].average()
        evoked_n=new_epochs['88'][rand_ind].average()
        evoked = mne.combine_evoked([evoked_p,evoked_n], weights='equal')
    # epochs.save(file_out + '_ffr_e.fif',overwrite=True)
    evoked.save(file_out + '_evoked_ffr.fif',overwrite=True)
    return evoked, epochs

def do_epoch_cabr_eeg(data, events, subject, condition, run, n_trials):  
    ###### Read the event files (generated from evtag.py) 
    random.seed(15)
    root_path = os.getcwd()
    file_out = root_path + '/EEG/preprocessed/ntrial_' + str(n_trials) +'/brainstem_' + subject + condition + run 
    reject=dict(eeg=100e-6) ## follow the 2018 paper
    epochs = mne.Epochs(data, events, event_id = [44,88],tmin =-0.02, tmax=0.20, baseline=(-0.02,0),reject=reject) ## the paper tmin tmax is -0.01, 0.15 but it cuts out the signal, so use -0.02 and 0.2 
    new_epochs = epochs.copy().drop_bad()
    ## match the trial number for each sound
    ## get random number of sounds from all sounds
    ## neet to find p and n len after dropping bad, use the smaller one to be the full len
    if n_trials == 'all':
        evoked=new_epochs.average(picks=('eeg'))
    else:
        rand_ind = random.Random().sample(range(min(len(new_epochs['44'].events),len(new_epochs['88'].events))),n_trials//2) 
        evoked_p=new_epochs['44'][rand_ind].average(picks=('eeg'))
        evoked_n=new_epochs['88'][rand_ind].average(picks=('eeg'))
        evoked = mne.combine_evoked([evoked_p,evoked_n], weights='equal')
        
    new_epochs.save(file_out + '_cabr_e_' + str(n_trials) + '.fif',overwrite=True)
    evoked.save(file_out + '_evoked_cabr_' + str(n_trials) + '.fif',overwrite=True)
    return evoked, new_epochs

#%%######################################## 
# mne.set_config('MNE_MEMMAP_MIN_SIZE', '10M') 
# mne.set_config('MNE_CACHE_DIR', '/dev/shm')
root_path='/media/tzcheng/storage/Brainstem/' # brainstem files
os.chdir(root_path)

## parameters 
conditions = ['_p10','_n40']
runs = ['_01','_02'] 
lp = 200 # try 200 (suggested by Nike) or 450 (from Coffey paper)
hp = 80
n_trials = 200 ## 'all' or 200 or any number

do_cabr = True # True: use the cABR filter, cov and epoch setting; False: use the MMR filter, cov and epoch setting

subj = [] # A104 got some technical issue
for file in os.listdir():
    if file.startswith('brainstem_1'): # brainstem
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
            # print ('Doing ECG/EOG projection...')
            # [raw,raw_erm] = do_projection(s,condition,run)
            # print ('Doing filtering...')
            # raw_filt = do_filtering(raw,lp,hp,do_cabr)
            # raw_erm_filt = do_filtering(raw_erm,lp,hp,do_cabr)
            # print ('calculate cov...')
            # do_cov(s,raw_erm_filt, do_cabr,hp,lp)
            print ('Doing epoch...')
            file_in=root_path + '/' + s + '/sss_fif/' + s + condition + run + '_otp_raw_sss_proj.fif'
            raw_file = mne.io.read_raw_fif(file_in,preload=True,allow_maxshield=True)
            events = find_events(raw_file)
            if do_cabr == True: 
                do_epoch_cabr_meg(raw_file, events, s, condition, run,n_trials,hp,lp)
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

#%%##### save EEG from the MEG recording file
## brainstem_107,brainstem_113 EEG030, the rest 100x use EEG034
for s in subj:
    print(s)
    for condition in conditions:
        for run in runs:
            file_in=root_path + '/' + s + '/sss_fif/' + s + condition + run + '_otp_raw_sss.fif'
            raw_file = mne.io.read_raw_fif(file_in,preload=True,allow_maxshield=True)
            eeg = find_eeg(raw_file,s,condition,run)
            
#%%##### do the jobs for EEG 
n_trials = 200 ## 'all' or 200 or any number

## These are the sbujects selected by their categorical perception that showed a clear categorization
subjects_eng=['104','106','107','108','110','111','112','113','118','121','123','124','126','129','133']
subjects_spa=['203','204','205','206','211','212','213','214','215','220','221','222','223','224','225','226'] ## 202 event code has some issues

for s in subjects_eng:
    print(s)
    for condition in conditions:
        for run in runs:
            input_file = root_path + 'EEG/raw/brainstem_' + s + condition + run +'_raw.fif'
            raw_file = mne.io.Raw(input_file,preload=True,allow_maxshield=True)
            # raw_file.copy().pick(picks="stim").plot()
            events = find_events(raw_file)
            raw_file.notch_filter(np.arange(60,2001,60),filter_length='auto',notch_widths=0.5,picks=('eeg'))
            raw_file.filter(l_freq=80,h_freq=2000,picks=('eeg'),method='iir',iir_params=dict(order=4,ftype='butter'))
            [evoked,epochs] = do_epoch_cabr_eeg(raw_file, events, s, condition, run, n_trials)
            
#%%####################################### Save the group npy file
root_path='/media/tzcheng/storage/Brainstem/EEG/'
t=np.linspace(-0.02,0.25,1001) ## need to double check the epoch tmin and tmax

## only include the subjects that have both p10 and n40
# subjects_eng=['104','106','107','108','110','112','113','118','121','123','124','126','129','133']
# subjects_spa=['202','203','204','205','206','212','214','215','220','221','222','223','224','225','226']

subjects_eng=['104','106','107','108','110','111','112','113','118','121','123','124','126','129','133']
subjects_spa=['203','204','205','206','211','212','213','214','215','220','221','222','223','224','225','226'] ## 202 event code has some issues

p10_eng = []
n40_eng = []
p10_spa = []
n40_spa = []

# for subj in subjects_eng:
#     evoked_p10 = np.loadtxt(root_path + str(subj)+'_p10_evoked_avg.txt')
#     evoked_n40 = np.loadtxt(root_path + str(subj)+'_n40_evoked_avg.txt')
#     p10_eng.append(evoked_p10)
#     n40_eng.append(evoked_n40)

# for subj in subjects_spa:
#     evoked_p10 = np.loadtxt(root_path + str(subj)+'_p10_evoked_avg.txt')
#     evoked_n40 = np.loadtxt(root_path + str(subj)+'_n40_evoked_avg.txt')
#     p10_spa.append(evoked_p10)
#     n40_spa.append(evoked_n40)
    
# for subj in subjects_eng:
#     evoked_p10 = np.loadtxt(root_path + str(subj)+'_p10_evoked_avg.txt')
#     evoked_n40 = np.loadtxt(root_path + str(subj)+'_n40_evoked_avg.txt')
#     p10_eng.append(evoked_p10)
#     n40_eng.append(evoked_n40)
    
for subj in subjects_spa:
    evoked_p10 = mne.read_evokeds(root_path + 'preprocessed/ntrial_all/spa/brainstem_' + str(subj)+'_p10_01_evoked_cabr_all.fif')
    evoked_n40 = mne.read_evokeds(root_path + 'preprocessed/ntrial_all/spa/brainstem_' + str(subj)+'_n40_01_evoked_cabr_all.fif')
    p10_spa.append(evoked_p10[0].data)
    n40_spa.append(evoked_n40[0].data)

for subj in subjects_eng:
    evoked_p10 = mne.read_evokeds(root_path + 'preprocessed/ntrial_all/eng/brainstem_' + str(subj)+'_p10_01_evoked_cabr_all.fif')
    evoked_n40 = mne.read_evokeds(root_path + 'preprocessed/ntrial_all/eng/brainstem_' + str(subj)+'_n40_01_evoked_cabr_all.fif')
    p10_eng.append(evoked_p10[0].data)
    n40_eng.append(evoked_n40[0].data)

p10_eng = np.squeeze(np.asarray(p10_eng))
n40_eng = np.squeeze(np.asarray(n40_eng))
p10_spa = np.squeeze(np.asarray(p10_spa))
n40_spa = np.squeeze(np.asarray(n40_spa))

np.save(root_path + 'p10_eng_eeg_ntrall_01.npy',p10_eng)
np.save(root_path + 'n40_eng_eeg_ntrall_01.npy',n40_eng)
np.save(root_path + 'p10_spa_eeg_ntrall_01.npy',p10_spa)
np.save(root_path + 'n40_spa_eeg_ntrall_01.npy',n40_spa)