#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 13:21:55 2023
Deal with the trigger issue of SLD. Splitting triggers occasionally happened. 
Need to take the first one as the events and get rid of the second half. 

@author: tzcheng
"""

import mne 
import numpy as np
import itertools


#%%######################################## SLD112
soa = np.load('/media/tzcheng/storage2/SLD/MEG/soas2_cbsb_200.npy',allow_pickle=True)
seq = np.load('/media/tzcheng/storage2/SLD/MEG/seq2_200.npy')
raw_file=mne.io.Raw('/media/tzcheng/storage2/SLD/MEG/check_events_sld_112/raw_fif/sld_112_t1_01_raw.fif',allow_maxshield=True,preload=True)
raw_file.copy().pick(picks="stim").plot(start=3, duration=6)

min_soa = 0.15
srate = raw_file.info['sfreq']

length=[]
for s in soa:
    length.append(s.shape[0])
n_events = np.sum(length)
        
echeck=[]
for i in range(seq.shape[0]):
    a=np.repeat(seq[i],length[i])
    echeck.append(a)
    
STI1 = mne.find_events(raw_file,stim_channel='STI001') 
STI3 = mne.find_events(raw_file,stim_channel='STI003')
STI4 = mne.find_events(raw_file,stim_channel='STI004')
STI1[:,2] = 1 # suppose to be 2184 events
STI3[:,2] = 4 # suppose to be 350 events
STI4[:,2] = 8 # suppose to be 250 events

diff_STI1 = np.diff(STI1,axis = 0) # anything shorter than 750 is probably wrong
diff_STI3 = np.diff(STI3,axis = 0) # anything shorter than 145 is probably wrong
diff_STI4 = np.diff(STI4,axis = 0) # anything shorter than 145 is probably wrong

## clean1 for the splitting triggers
STI1_s_ind = np.where(diff_STI1[:,0] < min_soa*srate)
STI3_s_ind = np.where(diff_STI3[:,0] < 100)
STI4_s_ind = np.where(diff_STI4[:,0] < 100) # still has some problems (4 more events than expected)
STI1 = np.delete(STI1,np.array(STI1_s_ind) + 1,axis=0) # delete the next item
STI3 = np.delete(STI3,np.array(STI3_s_ind) + 1,axis=0) # delete the next item
STI4 = np.delete(STI4,np.array(STI4_s_ind) + 1,axis=0) # delete the next item
events = np.concatenate((STI1,STI3,STI4),axis=0)
events = events[events[:,0].argsort()] # sort by the latency

## clean2 for the cross-talk (STI2 to STI3)
i = np.where(np.isin(events[:,2],[4,8]))
ii = np.where(np.diff(events[i,0]) < 50)[1]
events = np.delete(events,np.array(i[0])[ii+1],axis=0)

# check how well the code delete the broken ones
raw_file.copy().pick(picks="stim").plot(
    events=events,
    color="gray",
    event_color={1:"r",4:"b",8:"y"})

# check the ground truth in evtag.py


#%%######################################## vMMR 104 108 109
## three main issues
# 1. cross-talk duplicate deviant events
# 2. random standard events in the middle of the trial
# 3. trigger splitting
import mne 
import numpy as np

subj = 'vMMR_109'
pulse_dur = 50
condition_all = {"A":['1','4','2','3'],
             "B":['2','3','1','4'],
             "C":['4','1','3','2'],
             "D":['3','2','4','1']}
condition = {"vMMR_901":'A',
             "vMMR_902":'A',
             "vMMR_102":'B',
             "vMMR_103":'C',
             "vMMR_104":'D',
             "vMMR_108":'B',
             "vMMR_109":'A'}
sequence = {"vMMR_901":['1','1','1','1'],
             "vMMR_902":['1','1','1','1'],
             "vMMR_102":['7','2','8','9'],
             "vMMR_103":['3','4','6','2'],
             "vMMR_104":['5','6','7','8'],
             "vMMR_108":['2','5','3','9'],
             "vMMR_109":['3','7','6','4']}

cond = condition[subj]
seq = sequence[subj]

nrun = 3


raw=mne.io.Raw('/media/tzcheng/storage/vmmr/' + subj + '/' + subj + '_' + condition_all[cond][nrun] + '_raw.fif',allow_maxshield=True,preload=True)
# raw.copy().pick(picks="stim").plot(start=3, duration=6)
SOA = np.load('/media/tzcheng/storage/vmmr/npy/soas' + seq[nrun] + '_900.npy',allow_pickle=True)
SEQ = np.load('/media/tzcheng/storage/vmmr/npy/full_seq' + seq[nrun] + '_900.npy')
CHANGE = np.load('/media/tzcheng/storage/vmmr/npy/change_ind' + seq[nrun] + '_900.npy')

cross = mne.find_events(raw,stim_channel='STI001') # 900
std = mne.find_events(raw,stim_channel='STI002') # 768
d = mne.find_events(raw,stim_channel='STI003') # 132
r = mne.find_events(raw,stim_channel='STI005')
cross[:,2] = 1
std[:,2] = 2
d[:,2] = 4
r[:,2] = 16

## Get the number of events presented
print('Cross events:' + str(len(np.where(np.isin(SEQ, ['c','C']))[0])))
print('Stadard events:' + str(len(np.where(np.isin(SEQ, ['s','S']))[0])))
print('Deviant events:' + str(len(np.where(np.isin(SEQ, ['d','D']))[0])))
print('Change events:' + str(len(CHANGE)))

## clean1 for the splitting triggers
cross_s_ind = np.where(np.diff(cross,axis = 0)[:,0] < pulse_dur)
std_s_ind = np.where(np.diff(std,axis = 0)[:,0]  < pulse_dur)
d_s_ind = np.where(np.diff(d,axis = 0)[:,0]  < pulse_dur) # still has some problems (4 more events than expected)
r_s_ind = np.where(np.diff(r,axis = 0)[:,0]  < pulse_dur) # still has some problems (4 more events than expected)
cross = np.delete(cross,np.array(cross_s_ind) + 1,axis=0) # delete the next item
std = np.delete(std,np.array(std_s_ind) + 1,axis=0) # delete the next item
d = np.delete(d,np.array(d_s_ind) + 1,axis=0) # delete the next item
r = np.delete(r,np.array(r_s_ind) + 1,axis=0) # delete the next item
events_stim = np.concatenate((cross,std,d),axis=0)
events_stim = events_stim[events_stim[:,0].argsort()] # sort by the latency

## clean2 for the cross-talk (STI2 to STI3 same time stamp)
i = np.where(np.diff(events_stim[:,0]) < 100)[0]
events_stim = np.delete(events_stim,np.array(i+1),axis=0) # delete the second one (follow the order of 2 and 4)
events = np.concatenate((events_stim,r),axis=0)
events = events[events[:,0].argsort()] # sort by the latency

## clean3 for the random intruders e.g. 591.6 s

# check how well the code delete the broken ones
# raw.copy().pick(picks="stim").plot(
#     events=events,
#     color="gray",
#     event_color={1:"r",2:"g",4:"b",16:"y"})
len(np.where(events_stim[:,2] == 1)[0]) == len(np.where(np.isin(SEQ, ['c','C']))[0]) # check cross
len(np.where(events_stim[:,2] == 2)[0]) == len(np.where(np.isin(SEQ, ['s','S']))[0]) # check std
len(np.where(events_stim[:,2] == 4)[0]) == len(np.where(np.isin(SEQ, ['d','D']))[0]) # check dev

## check the ground truth
SEQ[np.where(np.isin(SEQ, ['c','C']))[0]] = 1 # ground truth
SEQ[np.where(np.isin(SEQ, ['s','S']))[0]] = 2 # ground truth
SEQ[np.where(np.isin(SEQ, ['d','D']))[0]] = 4 # ground truth
SEQ = SEQ.astype('int64')
compare = events_stim[:,2] - SEQ
print(np.sum(events_stim[:,2] == SEQ))
print('subject:' + subj + 'run' + str(nrun+1))




# %%
