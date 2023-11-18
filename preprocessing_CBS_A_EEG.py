#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 14:51:27 2021 by zhaotc, modified by tzcheng on Tue Oct  17

This script is for CBS EEG channel analysis. Need to use the event files output from evtag.py

@author: zhaotc
"""
import mne
import numpy as np
import itertools
import random

def find_eeg(raw_file,subj,block):
    raw_file.pick_channels(['BIO004','STI101'])
    raw_file.save('/media/tzcheng/storage/CBS/' + str(subj) +'/eeg/' + str(subj) + '_' + str(block) + '_raw.fif')
    return raw_file

def find_events(raw_file,subj,block):
    #find events
    events = mne.find_events(raw_file,stim_channel='STI101')
    root_path='/media/tzcheng/storage/CBS/'+str(subj)+'/events/'
    file_name_raw=root_path + str(subj)+ '_' + str(block) +'_events_raw-eve.fif'
    mne.write_events(file_name_raw,events)  ###write out raw events for double checking
    
def process_events(subj,block):
     #find events
    root_path='/media/tzcheng/storage/CBS/'+str(subj)+'/events/'
    file_name_raw=root_path + str(subj) + '_' + str(block) +'_events_raw-eve.fif'
    events=mne.read_events(file_name_raw)  ###write out raw events for double checking
    
    ###
    e=[]
    for event in events:
        e.append(event[2])
    ## dealing with the event codes
    ind1=[i for i, x in enumerate(e) if x==1] ## find all the indices for 1
    
    # the units are all 3 in length, first change them to 1, 3 and 6
    for i in ind1:
        if e[i-3]==4 and e[i-2]==4 and e[i-1]==8:
             events[i][2] = 1
        elif e[i-3]==4 and e[i-2]==8 and e[i-1]==4:
            events[i][2] = 3
        elif e[i-3]==4 and e[i-2]==8 and e[i-1]==8:
            events[i][2] = 6
    
    for i in ind1:
        if events[i-1][2] == 3:
            events[i][2] = 3
        elif events[i-1][2] == 6:
            events[i][2] = 6
            
    alt_set_length = {8: [3, 5], 10: [3, 5, 7],  # "alt" index by soa length
                 12: [3, 5, 7, 9], 14: [3, 5, 7, 9, 11]}
    
    ## figure out the alt ones
    # figure out the beginning and the end of the trials
    ind2=[0]
    for i in range(len(ind1))[1:]:
        if ind1[i]-ind1[i-1]>1: ## looking at if they are consecutive
            ind2.append(i-1)
            ind2.append(i)
    
    ind2.append(len(ind1)-1)
    
    #figure out length of each trial and the indx of alt sounds
    trial_length=[]
    alt_ind=[]
    for i in range(1,len(ind2),2):
        tl=ind2[i]-ind2[i-1]+1
        trial_length.append(tl)
        ai= [x + ind2[i-1] for x in alt_set_length[tl]]
        alt_ind.append(ai)
    
    # change the alt event code
    for i in alt_ind:
        if events[ind1[i[0]]][2] == 1:
            for m in i:
                events[ind1[m]][2]=2
        elif events[ind1[i[0]]][2] == 3:
            for m in i:
                events[ind1[m]][2]=5
        elif events[ind1[i[0]]][2] == 6:
            for m in i:
                events[ind1[m]][2]=7 
       
    root_path='/media/tzcheng/storage/CBS/'+str(subj)+'/events/'
    file_name_new=root_path + str(subj) + '_' + block +'_events_processed-eve.fif'
    mne.write_events(file_name_new,events)  ###read in raw events
    return events


def check_events(events,condition):  ## processed events
    ####
    #event check#
    e2=[]
    for event in events:
        if event[2]==1 or event[2]==2:
            e2.append(1)
        elif event[2]==3 or event[2]==5:
            e2.append(2)
        elif event[2]==6 or event[2]==7:
            e2.append(3)
            
           
    path='/media/tzcheng/storage/CBS/'
    seq_file=path + 'seq'+ condition+'_200.npy'
    seq=np.load(seq_file)
    
    soa_file=path + 'soas' + condition + '_200.npy'
    soa=np.load(soa_file, allow_pickle=True)
    
    length=[]
    for s in soa:
        length.append(s.shape[0])
        
    echeck=[]
    for i in range(seq.shape[0]):
        a=np.repeat(seq[i],length[i])
        echeck.append(a)
        
    echeck=list(itertools.chain(*echeck))
    
    check=[]
    if np.array_equal(e2,echeck[0:len(e2)]):
        print('all events are correct')
        check.append('correct')
    else:
        print ('something wrong with the event file!!!')
        check.append('incorrect')
    ####   
    return check


def select_mmr_events(events,subj,block): ## load the processed events
    e=[]
    for event in events:
        if event[2] in [1,2,3,5,6,7]:
            e.append(event)
    std=[]
    dev1=[]
    for i in range(len(e)):
        if e[i][2]==3 and e[i-1][2]==1:
            dev1.append(e[i])
            std.append(e[i-1])
            
    dev2=[]
    for i in range(len(e)):
        if e[i][2]==6 and e[i-1][2]==1:
            dev2.append(e[i])
            std.append(e[i-1])
    
        ###
    #randomly select standard trials from the ones that precedes deviants#
    ###
    sample_size = int(len(std)/2)
    substd = [std[i] for i in sorted(random.sample(range(len(std)), sample_size))]
    
    mmr_event=np.concatenate((substd,dev1,dev2),axis=0)
    
    root_path='/media/tzcheng/storage/CBS/'+str(subj)+'/events/'
    file_name_new=root_path + str(subj) + '_' + block + '_events_mmr-eve.fif'
    mne.write_events(file_name_new,mmr_event)
    return mmr_event


def select_cabr_events(events,subj,block): ## load the processed events
    e=[]
    for event in events:
        if event[2] in [1,2,3,5,6,7]:
            e.append(event)
            
    std1=[]
    std2=[]
    for i in range(len(e)):
        if e[i][2] == 2:
            std1.append(e[i-1])
            std2.append(e[i])
            
    dev1=[]
    for i in range(len(e)):
        if e[i][2] == 5:
            dev1.append(e[i-1])
            dev1.append(e[i])
    
    dev2=[]
    for i in range(len(e)):
        if e[i][2] == 7:
            dev2.append(e[i-1])
            dev2.append(e[i])
        
        #randomly select standard trials from the ones that precedes deviants#
    ###
    sample_size = int(len(std1)/2)
    substd1 = [std1[i] for i in sorted(random.sample(range(len(std1)), sample_size))]
    substd2 = [std2[i] for i in sorted(random.sample(range(len(std2)), sample_size))]
    
    cabr_event=np.concatenate((substd1,substd2,dev1,dev2),axis=0)
    
    root_path='/media/tzcheng/storage/CBS/'+str(subj)+'/events/'
    file_name_new=root_path + str(subj) + '_' + block + '_events_cabr-eve.fif'
    mne.write_events(file_name_new, cabr_event)
    return cabr_event

def average_mmr_trials(raw_file,mmr_event):    
    event_id = {'Standard':1,'Deviant1':3,'Deviant2':6}
    
    reject=dict(bio=100e-6)
    epochs = mne.Epochs(raw_file, mmr_event, event_id,tmin =-0.1, tmax=0.6,baseline=(-0.1,0),reject=reject)

    evoked_substd=epochs['Standard'].average(picks=('bio'))
    evoked_dev1=epochs['Deviant1'].average(picks=('bio'))
    evoked_dev2=epochs['Deviant2'].average(picks=('bio'))
    return evoked_substd,evoked_dev1,evoked_dev2, epochs

def average_cabr_trials(raw_file,cabr_events):    
    event_id = {'Standardp':1,'Standardn':2, 'Deviant1p':3,'Deviant1n':5, 'Deviant2p':6,'Deviant2n':7}
    
    reject=dict(bio=35e-6)
    epochs = mne.Epochs(raw_file, cabr_event, event_id,tmin =-0.02, tmax=0.20,baseline=(-0.02,0),reject=reject)

    evoked_substd=epochs['Standardp','Standardn'].average(picks=('bio'))
    evoked_dev1=epochs['Deviant1p','Deviant1n'].average(picks=('bio'))
    evoked_dev2=epochs['Deviant2p','Deviant2n'].average(picks=('bio'))
    return evoked_substd,evoked_dev1,evoked_dev2, epochs

#%% main
subj='cbs_A123'
block='01'
#condition='4'
#%%
raw_file=mne.io.Raw('/media/tzcheng/storage/CBS/'+subj+'/raw_fif/'+subj+'_'+ block+'_raw.fif',allow_maxshield=True,preload=True)
raw_file=find_eeg(raw_file,subj,block)
find_events(raw_file, subj,block)
events=process_events(subj,block)
check=check_events(events,condition)
mmr_event=select_mmr_events(events, subj, block)
cabr_event=select_cabr_events(events, subj, block)

#%% check the sound presentation
raw_file.pick_channels(['MISC001'])
event_id = {'Standardp':1,'Standardn':2, 'Deviant1p':3,'Deviant1n':5, 'Deviant2p':6,'Deviant2n':7}
epochs = mne.Epochs(raw_file, cabr_event, event_id,tmin =-0.02, tmax=0.2,baseline=(-0.02,0))
evoked_substd=epochs['Standardp'].average(picks=('misc'))
evoked_dev1=epochs['Deviant1p'].average(picks=('misc'))
evoked_dev2=epochs['Deviant2p'].average(picks=('misc'))
evoked_substd.plot()
evoked_dev1.plot()
evoked_dev2.plot()

event_id = {'Standard':1,'Deviant1':3,'Deviant2':6}
epochs = mne.Epochs(raw_file, mmr_event, event_id,tmin =-0.05, tmax=0.15,baseline=(-0.05,0))
evoked_substd=epochs['Standard'].average(picks=('misc'))
evoked_substd.plot()
#%% analyze cabr data
cabr_event=mne.read_events('/media/tzcheng/storage/CBS/'+subj+'/events/'+subj+'_'+block+'_events_cabr-eve.fif')

raw_file=mne.io.Raw('/mnt/CBS/'+subj+'/eeg/'+subj+'_'+block+'_raw.fif',allow_maxshield=True,preload=True)
raw_file.notch_filter(np.arange(60,2001,60),filter_length='auto',notch_widths=0.5,picks=('bio'))
raw_file.filter(l_freq=80,h_freq=2000,picks=('bio'),method='iir',iir_params=dict(order=4,ftype='butter'))
raw_file.pick_channels(['BIO004'])

evoked_substd,evoked_dev1,evoked_dev2,epochs_subc=average_cabr_trials(raw_file,cabr_event)  
evoked_substd.plot()
evoked_dev1.plot()
evoked_dev2.plot()

evoked_substd.save('/mnt/CBS/'+subj+'/eeg/cbs_' +str(subj)+'_' + str(block) + '_evoked_substd_cabr.fif')
evoked_dev1.save('/mnt/CBS/'+subj+'/eeg/cbs_' +str(subj)+'_' + str(block) + '_evoked_dev1_cabr.fif')
evoked_dev2.save('/mnt/CBS/'+subj+'/eeg/cbs_' +str(subj)+'_' + str(block) + '_evoked_dev2_cabr.fif')
epochs_subc.save('/mnt/CBS/'+subj+'/eeg/cbs_' +str(subj)+'_' + str(block) + '_epochs_subcortical.fif')

#%% analyze mmr data
mmr_event=mne.read_events('/mnt/CBS/'+subj+'/events/'+subj+'_'+block+'_events_mmr-eve.fif')

raw_file=mne.io.Raw('/mnt/CBS/'+subj+'/eeg/'+subj+'_'+block+'_raw.fif',allow_maxshield=True,preload=True)
raw_file.filter(l_freq=0,h_freq=50,picks=('bio'),method='iir',iir_params=dict(order=4,ftype='butter'))
raw_file.pick_channels(['BIO004'])

evoked_substd,evoked_dev1,evoked_dev2,epochs_cortical=average_mmr_trials(raw_file,mmr_event)  
evoked_substd.plot()
evoked_dev1.plot()
evoked_dev2.plot()

evoked_substd.save('/mnt/CBS/'+subj+'/eeg/cbs_' +str(subj)+'_' + str(block) + '_evoked_substd_mmr.fif')
evoked_dev1.save('/mnt/CBS/'+subj+'/eeg/cbs_' +str(subj)+'_' + str(block) + '_evoked_dev1_mmr.fif')
evoked_dev2.save('/mnt/CBS/'+subj+'/eeg/cbs_' +str(subj)+'_' + str(block) + '_evoked_dev2_mmr.fif')
epochs_cortical.save('/mnt/CBS/'+subj+'/eeg/cbs_' +str(subj)+'_' + str(block) + '_epochs_cortical.fif')

#%%
import matplotlib.pyplot as plt
mmr1=evoked_dev1.data-evoked_substd.data
mmr2=evoked_dev2.data-evoked_substd.data
time=evoked_substd.times
plt.plot(time,mmr1[0,:],label='mmr1')
plt.plot(time,mmr2[0,:],label='mmr2')
plt.legend()