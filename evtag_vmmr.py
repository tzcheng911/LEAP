###### Import library
import mne
import os
import numpy as np
import itertools
import matplotlib

###### Load data
root_path='/media/tzcheng/storage/vmmr/'
os.chdir(root_path)
runs = ['_1','_2','_3','_4']
condition = {"vMMR_901":['1','1','1','1'],
             "vMMR_902":['1','1','1','1'],
             "vMMR_102":['7','2','8','9'],
             "vMMR_103":['3','4','6','2'],
             "vMMR_104":['5','6','7','8'],
             "vMMR_108":['2','5','3','9'],
             "vMMR_109":['3','7','6','4']}
subj = [] 
for file in os.listdir():
    if file.startswith('vMMR_'):
        subj.append(file)

for s in subj:
    for run in runs:
        ###### Load files
        root_path = '/media/tzcheng/storage/vmmr/'
        file_in = root_path + '/' + s + '/' + s + run + '_raw.fif'
        file_out = root_path + '/' + s + '/events/'
        raw = mne.io.read_raw_fif(file_in,allow_maxshield=True)
        ch = raw.copy().pick("stim")

        ###### Find events 
        cross = mne.find_events(raw,stim_channel='STI001') 
        std = mne.find_events(raw,stim_channel='STI002')
        d = mne.find_events(raw,stim_channel='STI003')
        r = mne.find_events(raw,stim_channel='STI005')
        cross[:,2] = 1
        std[:,2] = 2
        d[:,2] = 4
        r[:,2] = 16
        events = np.concatenate((cross,std,d,r),axis=0)
        events = events[events[:,0].argsort()] # sort by the latency
        # events = mne.find_events(raw,stim_channel='STI101') # sum of all channels
        
        ###### Check events
        # SOA = np.load('/media/tzcheng/storage/vmmr/exp/soas1_900.npy')
        # seq = np.load('/media/tzcheng/storage/vmmr/exp/full_seq1_900.npy')
        # change = np.load('/media/tzcheng/storage/vmmr/exp/change_ind1_900.npy')
        
        ###### Write event file
        path = "events"
        isExist = os.path.exists(root_path + path)
        if not isExist:
            os.makedirs(root_path + path)
        mne.write_events(file_out + s + run + '_raw-eve.fif',events,overwrite=True) 

        ## Read event file
        # file_name_events = '/media/tzcheng/storage/vmmr/vMMR_901/events/vMMR_901_1_raw-eve.fif'
        # events = mne.read_events(file_name_events)

        ###### Select relevant events 
        r = mne.pick_events(events, include = 16)
        c = mne.pick_events(events, include = 1)
        std = mne.pick_events(events, include = 2)
        d = mne.pick_events(events, include = 4)
        no_response = mne.pick_events(events, include = [1,2,4])
        sd = mne.pick_events(events, include = [2,4])

        ## MMR Code std before d to 6
        for n in range(len(sd)):
            if sd[n,2] == 4 and sd[n-1,2] == 2:
                sd[n-1,2] = 6
        sd = mne.pick_events(sd, include = [6,4])
        mne.write_events(file_out + s + run + '_mmr-eve.fif',sd,overwrite=True) 





## Calculate correct response (development calculate d prime)
stim = mne.pick_events(events, include = [1,2,4])
resp = mne.pick_events(events, include = [16])
resp[:,0] - stim[change,0] # RT of key press after the stim change in ms; such a good subject! 79/80 accuracy