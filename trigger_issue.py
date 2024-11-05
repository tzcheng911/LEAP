#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 13:21:55 2023
Deal with the trigger issue of SLD. Splitting triggers occasionally happened. 
Need to take the first one as the events and get rid of the second half. 
Need to deal with the cross-talk using the direction (2 -> 4)

@author: tzcheng
"""

import mne 
import numpy as np
import itertools


#%%######################################## SLD112
s = 'sld_112'
soa = np.load('/media/tzcheng/storage2/SLD/MEG/soas2_cbsb_200.npy',allow_pickle=True)
seq = np.load('/media/tzcheng/storage2/SLD/MEG/seq2_200.npy')
raw_file=mne.io.Raw('/media/tzcheng/storage2/SLD/MEG/' + s + '/raw_fif/' + s +'_t1_01_raw.fif',allow_maxshield=True,preload=True)
raw_file.copy().pick(picks="stim").plot()

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