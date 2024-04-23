#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 11:47:27 2023

This script is for CBS event processing FOR ADULTS ONLY. 
It is used for 
1. extract the /ba/ as standard and /mba/ as dev1 /pa/ as dev 2
2. extract the /mba/ as standard 1 /pa/ as standard 2 and /ba/ as dev
3. extract the last /ba/ as standard, the first /ba/ (follow pa, or mba seperately) as deviants (address the N1 disparity issue)
see eeg_analysis_final.py for the original script to extract adult MMR and cABR

Some notes: A119 raw event is messed up (STI101), subtracting 1024 from the second and the third column

The correspondance between event tage and sound are

                        TTl                  event code
ba                      448                  1 and 2 (alt)
mba                     484                  3 and 5 (alt)
pa                      488                  6 and 7 (alt)

@author: tzcheng
"""
import mne
import numpy as np
import itertools
import random
import os

def find_events(raw_file,subj,block):
    #find events
    events = mne.find_events(raw_file,stim_channel='STI101')
    if subj == 'cbs_A119':
        events[:,2] = events[:,2] - 1024
    root_path='/media/tzcheng/storage/CBS/'+str(subj)+'/events/'
    file_name_raw=root_path + str(subj)+ str(block) +'_events_raw-eve.fif'
    # mne.write_events(file_name_raw,events,overwrite=True)  ###write out raw events for double checking
    
def process_events(subj,block):
     #find events
    root_path='/media/tzcheng/storage/CBS/'+str(subj)+'/events/'
    file_name_raw=root_path + str(subj) + str(block) +'_events_raw-eve.fif'
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
    file_name_new=root_path + str(subj) + block +'_events_processed-eve.fif'
    # mne.write_events(file_name_new,events,overwrite=True)  ###read in raw events
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

def select_mmr_events(events,subj,block, direction): ## load the processed events
    e=[]
    for event in events:
        if event[2] in [1,2,3,5,6,7]:
            e.append(event)
    ## For the /ba/ as standard and /mba/ as dev1 /pa/ as dev 2
    if direction == 'ba_to_pa':
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
        
        #randomly select standard trials from the ones that precedes deviants#
        ###
        sample_size = int(len(std)/2)
        substd = [std[i] for i in sorted(random.sample(range(len(std)), sample_size))]
        
        mmr_event=np.concatenate((substd,dev1,dev2),axis=0)
        
        root_path='/media/tzcheng/storage/CBS/'+str(subj)+'/events/'
        file_name_new=root_path + str(subj) + block + '_events_mmr-eve.fif'
        mne.write_events(file_name_new,mmr_event,overwrite=True)

    ## For the last /ba/ as standard first /ba/ after /mba/ as dev1 (50 trials) and after /pa/ as dev2 (only 49 cuz the last one is mba or pa)
    elif direction == 'first_last_ba':
        std=[] # last /ba/
        dev1=[] # first /ba/ after /mba/ 31
        dev2=[] # first /ba/ after /pa/ 61
        for i in range(len(e)-1):
            if e[i][2]==3 and e[i+1][2]==1: 
                e[i+1][2]=31
                dev1.append(e[i+1])

        for i in range(len(e)-1):
            if e[i][2]==6 and e[i+1][2]==1: 
                e[i+1][2]=61
                dev2.append(e[i+1])    
        
        for i in range(len(e)-1):
            if (e[i][2]==1 and e[i+1][2]==3) or (e[i][2]==1 and e[i+1][2]==6): 
                std.append(e[i])
        #randomly select standard trials from the ones that precedes deviants#
        ###
        sample_size = int(len(std)/2)
        substd = [std[i] for i in sorted(random.sample(range(len(std)), sample_size))]
        
        mmr_event=np.concatenate((substd,dev1,dev2),axis=0)
        
        root_path='/media/tzcheng/storage/CBS/'+str(subj)+'/events/'
        file_name_new=root_path + str(subj) + block + '_events_mmr_firstlastba-eve.fif'
        mne.write_events(file_name_new,mmr_event,overwrite=True)

    ## For the /mba/ as standard 1 /pa/ as standard 2 and /ba/ as dev
    elif direction == 'pa_to_ba':
        std1=[] # /mba/ to /ba/
        dev=[]
        for i in range(len(e)-1):
            if e[i][2]==3 and e[i+1][2]==1: # change from -1 to +1 so get the std and dev order reverse
                std1.append(e[i])
                dev.append(e[i+1]) # change from -1 to +1 so get the std and dev order reverse
                
        std2=[] # /pa/ to /ba/
        for i in range(len(e)-1):
            if e[i][2]==6 and e[i+1][2]==1: # change from -1 to +1 so get the std and dev order reverse
                std2.append(e[i])
                dev.append(e[i+1]) # change from -1 to +1 so get the std and dev order reverse
        
        #randomly select dev trials from the ones that after standards#
        ###
        sample_size = int(len(dev)/2)
        dev_sample = [dev[i] for i in sorted(random.sample(range(len(dev)), sample_size))]
        
        mmr_event=np.concatenate((std1,std2,dev_sample),axis=0)
        
        root_path='/media/tzcheng/storage/CBS/'+str(subj)+'/events/'
        file_name_new=root_path + str(subj) + block + '_events_mmr_reverse-eve.fif'
        mne.write_events(file_name_new,mmr_event,overwrite=True)
    else:
        print('Select a direction to do MMR!')
    return mmr_event

#%% 
########################################
root_path='/media/tzcheng/storage/CBS/'
os.chdir(root_path)

## parameters 
run = '_01' # ['_01','_02'] for adults and ['_01'] for infants
# conditions = ['6','1','2','1','5','6','1','3','6','3','5','6','2','1','4','1','2','3'] # run1: follow the order of subj
# conditions = ['4','6','4','3','4','2','4','6','3','3','4','1','5','2','2','4','4','3']# run2: follow the order of subj
direction = 'first_last_ba' # traditional direction 'ba_to_pa': ba to pa and ba to mba
# reverse direction 'pa_to_ba' : is pa to ba and mba to ba; 
# only comparing /ba/ 'first_last_ba': only comparing /ba/ before and after habituation 
subj = [] 
check_all= []
for file in os.listdir():
    if file.startswith('cbs_A'):
        subj.append(file)

###### do the jobs
for n,s in enumerate(subj):
#    condition = conditions[n]
    isExist = os.path.exists(root_path + s + '/events')
    if not isExist:
        os.makedirs(root_path + s + '/events')
        
    raw_file=mne.io.Raw('/media/tzcheng/storage/CBS/' + s + '/raw_fif/' + s + run +'_raw.fif',allow_maxshield=True,preload=True)
    find_events(raw_file, s,run)
    events=process_events(s,run)
    # check=check_events(events,condition)
    # check_all.append(check)
    mmr_event=select_mmr_events(events, s, run, direction)

# %% get the MEG soa from the event files
events=mne.read_events('/media/tzcheng/storage2/CBS/cbs_A107/events/cbs_A107_01_events_raw-eve.fif') 
onset = mne.pick_events(events, include=1)
soa = np.diff(onset[:,0])/5000
print(np.min(soa))
print(np.max(soa))