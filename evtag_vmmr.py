###### Import library
import mne
import os
import numpy as np
import itertools
import matplotlib
from scipy.stats import norm

###### Load data
subj = 'vMMR_201'

root_path='/media/tzcheng/storage/vmmr/'
os.chdir(root_path)
pulse_dur = 50

###### Dictionary for different subjects
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
             "vMMR_109":'A',
             "vMMR_201":'A'}
sequence = {"vMMR_901":['1','1','1','1'],
             "vMMR_902":['1','1','1','1'],
             "vMMR_102":['7','2','8','9'],
             "vMMR_103":['3','4','6','2'],
             "vMMR_104":['5','6','7','8'],
             "vMMR_108":['2','5','3','9'],
             "vMMR_109":['3','7','6','4'],
             "vMMR_201":['6','5','7','9']}

cond = condition[subj]
seq = sequence[subj]

###### Do the jobs
for nrun in np.arange(0,4,1):
    ###### Load files
    file_in = root_path + subj + '/' + subj + '_' + condition_all[cond][nrun] + '_raw.fif'
    file_out = root_path + subj + '/events/'
    raw = mne.io.read_raw_fif(file_in, allow_maxshield=True)

    ###### Find events
    cross = mne.find_events(raw, stim_channel='STI001')
    std = mne.find_events(raw, stim_channel='STI002')
    d = mne.find_events(raw, stim_channel='STI003')
    r = mne.find_events(raw, stim_channel='STI005')
    cross[:, 2] = 1
    std[:, 2] = 2
    d[:, 2] = 4
    r[:, 2] = 16

    ###### Repair broken triggers
    # step 1: delete the second splitting triggers
    cross_s_ind = np.where(np.diff(cross,axis = 0)[:,0] < pulse_dur)
    std_s_ind = np.where(np.diff(std,axis = 0)[:,0]  < pulse_dur)
    d_s_ind = np.where(np.diff(d,axis = 0)[:,0]  < pulse_dur) 
    r_s_ind = np.where(np.diff(r,axis = 0)[:,0]  < pulse_dur) 
    cross = np.delete(cross,np.array(cross_s_ind) + 1,axis=0) # delete the next item
    std = np.delete(std,np.array(std_s_ind) + 1,axis=0) # delete the next item
    d = np.delete(d,np.array(d_s_ind) + 1,axis=0) # delete the next item
    r = np.delete(r,np.array(r_s_ind) + 1,axis=0) # delete the next item
    events_stim = np.concatenate((cross,std,d),axis=0)
    events_stim = events_stim[events_stim[:,0].argsort()] # sort by the latency
    
    # step 2: deal with the cross-talk (leaking from STI1 to STI2, and STI2 to STI3)
    i = np.where(np.diff(events_stim[:,0]) < pulse_dur)[0]
    i_del = [] # delete the larger one cuz the cross-talk happen in this direction (2 followed 1 or 4 followed 2)
    for ni in i:
        if np.argmax(events_stim[ni:ni+2,2]) == 0:
            i_del.append(ni)
        elif np.argmax(events_stim[ni:ni+2,2]) == 1:
            i_del.append(ni+1)
       
    if i_del: 
        events_stim = np.delete(events_stim,np.array([i_del])[0],axis=0) 

    # step 3: for the random intruders e.g. 430 s (event 953) for vMMR_108 run 2
    if subj == 'vMMR_108' and nrun == 1:
        events_stim = np.delete(events_stim,953,axis=0) # manually delete this one random intruder 

    events = np.concatenate((events_stim,r),axis=0)
    events = events[events[:,0].argsort()] # sort by the latency

    ###### Check events
    # raw.copy().pick(picks="stim").plot(
    #     events=events,
    #     color="gray",
    #     event_color={1:"r",2:"g",4:"b",16:"y"})
    
    SOA = np.load(root_path + 'npy/soas' + seq[nrun] + '_900.npy',allow_pickle=True)
    SEQ = np.load(root_path + 'npy/full_seq' + seq[nrun] + '_900.npy')
    CHANGE = np.load(root_path + 'npy/change_ind' + seq[nrun] + '_900.npy')

    print(len(np.where(events_stim[:,2] == 1)[0]) == len(np.where(np.isin(SEQ, ['c','C']))[0])) # check cross
    print(len(np.where(events_stim[:,2] == 2)[0]) == len(np.where(np.isin(SEQ, ['s','S']))[0])) # check std
    print(len(np.where(events_stim[:,2] == 4)[0]) == len(np.where(np.isin(SEQ, ['d','D']))[0])) # check dev

    SEQ[np.where(np.isin(SEQ, ['c','C']))[0]] = 1 # ground truth
    SEQ[np.where(np.isin(SEQ, ['s','S']))[0]] = 2 # ground truth
    SEQ[np.where(np.isin(SEQ, ['d','D']))[0]] = 4 # ground truth
    SEQ = SEQ.astype('int64')
    compare = events_stim[:,2] - SEQ
    print('Matching ones (out of 1800):' + str(np.sum(events_stim[:,2] == SEQ)))
    print('subject:' + subj + ' run' + str(nrun+1))

    ###### Write event file
    path = "/events"
    isExist = os.path.exists(root_path + subj + path)
    if not isExist:
        os.makedirs(root_path + subj + path)
        mne.write_events(file_out + subj + condition_all[cond][nrun] + '_raw-eve.fif',events, overwrite=True)

    ###### Select relevant events for MMR
    # Code std before d to 6
    # make sure no responses within 100 ms before and 500 ms after the std or dev
    sdr_del = []
    
    # get the s, d and r events
    sd = mne.pick_events(events, include=[2, 4])
    for n in range(len(sd)):
        if sd[n, 2] == 4 and sd[n-1, 2] == 2:
            sd[n-1, 2] = 6
    sd = mne.pick_events(sd, include=[6, 4])
    sdr = np.concatenate((sd,r),axis=0)
    sdr = sdr[sdr[:,0].argsort()]
    
    # add first and last padding for calculation
    sdr = np.concatenate(([[1,1,1]],sdr,[[1,1,1]]),axis=0)
    
    # catch the button press within [-100 500] window of a std or dev events
    for n in np.arange(0,len(sdr),1):
        if sdr[n,2] == 16 and sdr[n-1,2] != 16:
            RT_pre = sdr[n+1,0]-sdr[n,0]
            RT_post = sdr[n,0]-sdr[n-1,0]
            if abs(RT_pre) < 100:
                sdr_del.append(n+1)
                print("Press during the std or dev trials!")
            elif abs(RT_post) < 500:
                sdr_del.append(n-1)
                print("Press during the std or dev trials!")
    print('Trials contaminated by button press:' + str(len(sdr_del)))
    sdr = np.delete(sdr,sdr_del,axis=0) # delete the events contaminated by button press
    mne.write_events(file_out + subj + condition_all[cond][nrun] +'_mmr-eve.fif', sdr, overwrite = True)

    ###### Evaluate the behavioral performance
    # 1. count as correct response if less than 1 s after the change
    # 2. calculate d prime
    events_stim[CHANGE,2] = 166
    resp = np.concatenate((events_stim[CHANGE,:],r),axis=0)
    resp = resp[resp[:,0].argsort()] # 166 is the change, 16 is the button press

    correct = 0
    
    # calculate the Hit rate
    for i in np.arange(0,len(resp)-1,1):
        if resp[i,2] == 166 and resp[i+1,2] == 16: # press after the change
            RT = resp[i+1,0] - resp[i,0]
            if RT < 1000:
                correct += 1
        elif all(resp[i:i+2,2] == 166) and all(resp[i+2:i+4,2] == 16): ## consider close change 166, 166, 16, 16
            RT_double = resp[i+4,0] - resp[i+1,0] # second 16 should be less than 1s of the second 166
            # print(i)
            # print(resp[i:i+4])
            if RT_double < 1000:
                correct += 1
        elif resp[i,2] == 16 and resp[i+1,2] == 166:
            pass
        # else:
        #     print("something is wrong!")
        #     print(i)
    
    # calculate the FA rate
    FA = 0
    for i in np.arange(0,len(resp)-1,1):
        if resp[i,2] == 16 and (resp[i,0] - resp[i-1,0] > 1000): # find the ones longer than 1s from previous event (i.e. 16 or 166)
            FA += 1
            # print(resp[i-2:i+2])

    # Calculate accuracy and dprime
    accuracy = correct/80
    
    hit = correct/len(r)
    FA = FA/len(r)
    dprime = norm.ppf(hit) - norm.ppf(FA)

    print("Accuracy:" + str(accuracy))
    print("d prime:" + str(dprime))