###### Import library
import mne
import os
import numpy as np
import itertools
import matplotlib
matplotlib.use('QtAgg') 

###### Load data
SOA = np.load('/media/tzcheng/storage/vmmr/vMMR_901/exp/soas1_900.npy')
seq = np.load('/media/tzcheng/storage/vmmr/vMMR_901/exp/full_seq1_900.npy')
change = np.load('/media/tzcheng/storage/vmmr/vMMR_901/exp/change_ind1_900.npy')
root_path='/media/tzcheng/storage/vmmr/vMMR_901/'
file_name_raw=root_path + 'raw_fif/' + 'vMMR_901_1_raw.fif'
raw = mne.io.read_raw_fif(file_name_raw,allow_maxshield=True)
ch = raw.copy().pick("stim")
ch.plot()

###### Find events 
cross = mne.find_events(raw,stim_channel='STI001') 
s = mne.find_events(raw,stim_channel='STI002')
d = mne.find_events(raw,stim_channel='STI003')
r = mne.find_events(raw,stim_channel='STI005')
cross[:,2] = 1
s[:,2] = 2
d[:,2] = 4
r[:,2] = 16
events = np.concatenate((cross,s,d,r),axis=0)
events = events[events[:,0].argsort()] # sort by the latency
# events = mne.find_events(raw,stim_channel='STI101') # sum of all channels

## Write event file
path = "events"
isExist = os.path.exists(root_path + path)
if not isExist:
    os.makedirs(root_path + path)
file_name_events = root_path + path + '/vMMR_901_1_raw-eve.fif'
mne.write_events(file_name_events,events) 

## Read event file
file_name_events = '/media/tzcheng/storage/vmmr/vMMR_901/events/vMMR_901_1_raw-eve.fif'
events = mne.read_events(file_name_events)

## Select relevant events 
r = mne.pick_events(events, include = 16)
c = mne.pick_events(events, include = 1)
s = mne.pick_events(events, include = 2)
d = mne.pick_events(events, include = 4)
no_response = mne.pick_events(events, include = [1,2,4])
sd = mne.pick_events(events, include = [2,4])

## MMR Code s before d to 6
for n in range(len(sd)):
    if sd[n,2] == 4 and sd[n-1,2] == 2:
        sd[n-1,2] = 6
sd = mne.pick_events(sd, include = [6,4])

file_name_events = '/media/tzcheng/storage/vmmr/vMMR_901/events/vMMR_901_1_raw-eve2.fif'
mne.write_events('/media/tzcheng/storage/vmmr/vMMR_901/events/vMMR_901_1_mmr-eve.fif',sd) 

## Calculate correct response (development calculate d prime)
stim = mne.pick_events(events, include = [1,2,4])
resp = mne.pick_events(events, include = [16])
resp[:,0] - stim[change,0] # RT of key press after the stim change in ms; such a good subject! 79/80 accuracy