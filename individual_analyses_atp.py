#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 13:19:12 2023

ATP analysis based on preprocessing done by Ashley Drew

'condition code vs. condition vs. event code '
'condition 1 = ba/touch = 48 == 1'
'condition 2 = ba/no-touch = 44 === 2'
'condition 3 = ga/touch = 88 === 3'
'condition 4 = ga/no-touch = 84 ===5'

@author: zhaotc
"""

import mne
import numpy as np


def find_events(raw_file,subj):
    #find events
    events_stim = mne.find_events(raw_file, stim_channel='STI101',mask=15)
    events_touch = mne.find_events(raw_file, stim_channel='STI011')
    for event in events_touch:
        event[2] = 16
    root_path='/mnt/storage/ATP/'+str(subj)+'/events/'
    file_name_stim = root_path + str(subj)+ '_events_stim-eve.fif'
    file_name_touch = root_path + str(subj)+ '_events_touch-eve.fif'
    
    mne.write_events(file_name_stim, events_stim)
    mne.write_events(file_name_touch, events_touch)
    return events_stim, events_touch
    
    
def process_events_stim(subj):
     #find events
    root_path='/mnt/storage/ATP/'+str(subj)+'/events/'
    file_name_stim=root_path + str(subj) + '_events_stim-eve.fif'
    events_stim=mne.read_events(file_name_stim)  ###write out raw events for double checking
    
    ###
    e=[]
    for event in events_stim:
        e.append(event[2])
    ## dealing with the event codes
    ind1=[i for i, x in enumerate(e) if x==1] ## find all the indices for 1
    
    # see the code map in the beginning
    for i in ind1:
        if  e[i-2]==4 and e[i-1]==8:
             events_stim[i][2] = 1
        elif e[i-2]==4 and e[i-1]==4:
            events_stim[i][2] = 2
        elif e[i-2]==8 and e[i-1]==8:
            events_stim[i][2] = 3
        elif e[i-2]==8 and e[i-1]==4:
            events_stim[i][2] = 5
    
       
    root_path='/mnt/storage/ATP/'+str(subj)+'/events/'
    file_name_stim_new=root_path + str(subj) + '_events_stim_processed-eve.fif'
    mne.write_events(file_name_stim_new,events_stim)  ###read in raw events
    return events_stim


def check_events(events_stim,condition):  ## processed events
    ####
    #event check#
    e2=[]
    for event in events_stim:
        if event[2]==1:
            e2.append(1)
        elif event[2]==2:
            e2.append(2)
        elif event[2]==3:
            e2.append(3)
        elif event[2]==5:
            e2.append(4)
            
           
    path='/mnt/storage/ATP/'
    seq_file=path + 'seq'+ condition+'_atp_60.npy'
    seq=np.load(seq_file, allow_pickle=True)

    #faltten seq list
    echeck=[]
    for sublist in seq:
        for item in sublist:
            echeck.append(item)
        
    if np.array_equal(e2,echeck[0:len(e2)]):
        print('all events are correct')
    else:
        print ('something wrong with the event file!!!')
    ####   
    return
        
#event fixing is on a separate file

def get_evoked(epochs,subj):
    evoked_ba_touch=epochs['ba-touch'].average()
    evoked_ga_touch=epochs['ga-touch'].average()
    evoked_ba_notouch=epochs['ba-notouch'].average()
    evoked_ga_notouch=epochs['ga-notouch'].average()

    evoked_ba_touch.plot(spatial_colors=True, gfp=True)
    evoked_ba_notouch.plot(spatial_colors=True, gfp=True)
    evoked_ga_touch.plot(spatial_colors=True, gfp=True)
    evoked_ga_notouch.plot(spatial_colors=True, gfp=True)
    
    return evoked_ba_notouch,evoked_ba_touch,evoked_ga_notouch,evoked_ga_touch
    
def do_forward(subj,raw):
    src = mne.read_source_spaces('/mnt/storage/Subjects/' + subj + '/bem/' + subj + '-ico-5-src.fif')
    trans = mne.read_trans('/mnt/storage/Subjects/' + subj + '/' + subj +'-trans.fif')
    bem = mne.read_bem_solution('/mnt/storage/Subjects/' + subj + '/bem/' + subj + '-5120-5120-5120-bem-sol.fif')

    fwd = mne.make_forward_solution(raw.info, trans, src, bem, eeg = False)

    mne.write_forward_solution('/mnt/storage/ATP/'+subj+'/sss_fif/'+subj+'_fwd.fif', fwd)
    return fwd

#%% open file
subj='atp_a122'
raw = mne.io.Raw('/mnt/storage/ATP/'+subj+'/sss_fif/'+subj+'_01_allclean_fil50_raw_sss.fif')
#%% epochs and evoked
events_stim=mne.read_events('/mnt/storage/ATP/'+subj+'/events/'+subj+'_events_stim_fixed-eve.fif')
event_id = {'ba-touch':1,'ba-notouch':2, 'ga-touch':3,'ga-notouch':5}
epochs = mne.Epochs(raw, events_stim, event_id,tmin =-0.1, tmax=0.8,baseline=(-0.1,0))

# calculate evoked and write out epochs and evoked files
evoked_ba_notouch,evoked_ba_touch,evoked_ga_notouch,evoked_ga_touch = get_evoked(epochs, subj)

epochs.save('/mnt/storage/ATP/'+subj+'/sss_fif/'+subj+'_epochs.fif',overwrite=True)
evoked_ba_touch.save('/mnt/storage/ATP/'+subj+'/sss_fif/'+subj+'_ba_touch_evoked.fif',overwrite=True)
evoked_ga_touch.save('/mnt/storage/ATP/'+subj+'/sss_fif/'+subj+'_ga_touch_evoked.fif',overwrite=True)
evoked_ba_notouch.save('/mnt/storage/ATP/'+subj+'/sss_fif/'+subj+'_ba_notouch_evoked.fif',overwrite=True)
evoked_ga_notouch.save('/mnt/storage/ATP/'+subj+'/sss_fif/'+subj+'_ga_notouch_evoked.fif',overwrite=True)

#%% source localization
fwd = do_forward(subj,raw)
#fwd = mne.read_forward_solution()

cov = mne.read_cov('/mnt/storage/ATP/' + subj + '/sss_fif/' + subj + '_erm_50_sss-cov.fif' )
#cov.save('/mnt/storage/ATP/' + subj + '/sss/' + subj + '_cov.fif')

inverse_operator = mne.minimum_norm.make_inverse_operator(raw.info, fwd, cov,loose=0.2,depth=0.8)

evoked_ba_touch = mne.read_evokeds('/mnt/storage/ATP/'+subj+'/sss_fif/'+subj+'_ba_touch_evoked.fif')[0]
evoked_ba_notouch = mne.read_evokeds('/mnt/storage/ATP/'+subj+'/sss_fif/'+subj+'_ba_notouch_evoked.fif')[0]
evoked_ga_touch = mne.read_evokeds('/mnt/storage/ATP/'+subj+'/sss_fif/'+subj+'_ga_touch_evoked.fif')[0]
evoked_ga_notouch = mne.read_evokeds('/mnt/storage/ATP/'+subj+'/sss_fif/'+subj+'_ga_notouch_evoked.fif')[0]

stc_ba_touch = mne.minimum_norm.apply_inverse((evoked_ba_touch), inverse_operator)
stc_ba_notouch = mne.minimum_norm.apply_inverse((evoked_ba_notouch), inverse_operator)
stc_ga_touch = mne.minimum_norm.apply_inverse((evoked_ga_touch), inverse_operator)
stc_ga_notouch = mne.minimum_norm.apply_inverse((evoked_ga_notouch), inverse_operator)

stc_ba_diff = stc_ba_notouch - stc_ba_touch
stc_ga_diff = stc_ga_notouch - stc_ga_touch

stc_ba_touch.save('/mnt/storage/ATP/'+subj+'/sss_fif/'+subj+'_ba_touch_stc.fif')
stc_ba_notouch.save('/mnt/storage/ATP/'+subj+'/sss_fif/'+subj+'_ba_notouch_stc.fif')
stc_ga_touch.save('/mnt/storage/ATP/'+subj+'/sss_fif/'+subj+'_ga_touch_stc.fif')
stc_ga_notouch.save('/mnt/storage/ATP/'+subj+'/sss_fif/'+subj+'_ga_notouch_stc.fif')
stc_ba_diff.save('/mnt/storage/ATP/'+subj+'/sss_fif/'+subj+'_ba_diff_stc.fif')
stc_ga_diff.save('/mnt/storage/ATP/'+subj+'/sss_fif/'+subj+'_ga_diff_stc.fif')

#%% induced time-frequency
# We are mostly interested in the beta band since it has been shown to be
# active for somatosensory stimulation
freqs = np.linspace(13, 31, 5)

# Use Morlet wavelets to compute sensor-level time-frequency (TFR)
# decomposition for each epoch. We must pass ``output='complex'`` if we wish to
# use this TFR later with a DICS beamformer. We also pass ``average=False`` to
# compute the TFR for each individual epoch.
epochs_tfr = tfr_morlet(
    epochs, freqs, n_cycles=5, return_itc=False, output="complex", average=False
)

# crop either side to use a buffer to remove edge artifact
epochs_tfr.crop(tmin=-0.5, tmax=2)

#%% beamformer
# Compute the Cross-Spectral Density (CSD) matrix for the sensor-level TFRs.
# We are interested in increases in power relative to the baseline period, so
# we will make a separate CSD for just that period as well.
csd = csd_tfr(epochs_tfr, tmin=-0.5, tmax=2)
baseline_csd = csd_tfr(epochs_tfr, tmin=-0.5, tmax=-0.1)

# use the CSDs and the forward model to build the DICS beamformer
fwd = mne.read_forward_solution(fname_fwd)

# compute scalar DICS beamfomer
filters = make_dics(
    epochs.info,
    fwd,
    csd,
    noise_csd=baseline_csd,
    pick_ori="max-power",
    reduce_rank=True,
    real_filter=True,
)

# project the TFR for each epoch to source space
epochs_stcs = apply_dics_tfr_epochs(epochs_tfr, filters, return_generator=True)

# average across frequencies and epochs
data = np.zeros((fwd["nsource"], epochs_tfr.times.size))
for epoch_stcs in epochs_stcs:
    for stc in epoch_stcs:
        data += (stc.data * np.conj(stc.data)).real

stc.data = data / len(epochs) / len(freqs)

# apply a baseline correction
stc.apply_baseline((-0.5, -0.1))
    