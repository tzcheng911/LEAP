# -*- coding: utf-8 -*-
"""
Created on Tue Aug 8 11:28:12 2023

Preprocessing for CBS MEG and EEG. Need to have events file ready from evtag.py
Need to manually enter bad channels for sss from the experiment notes. 
Need to check parameters st_correlation and int_order in sss for adult/infants
Didn't save the product from filtering to save some space

Could be used on cbsb too after otp and sss!

@author: tzcheng
"""

###### Import library 
import mne
import mnefun
import matplotlib
from mne.preprocessing import maxwell_filter
import numpy as np
import os
import random

def do_otp(subject):
    root_path='/media/tzcheng/storage2/CBS/'+ subject +'/raw_fif/'
    os.chdir(root_path)
    #find all the raw files
    runs=['01','02','erm'] # ['01','02','erm'] for the adults and ['01','erm'] for the infants
    for run in runs:
        # file_in=root_path+'cbs'+str(subj)+'_'+str(run)+'_raw.fif'
        # file_out=root_path+'cbs'+str(subj)+'_'+str(run)+'_otp_raw.fif'
        file_in=root_path+subject+'_'+run+'_raw.fif'
        file_out=root_path+subject+'_'+run+'_otp_raw.fif'
        raw=mne.io.Raw(file_in,allow_maxshield=True)
        picks=mne.pick_types(raw.info,meg=True,eeg=False,eog=False, ecg=False,exclude='bads')
        raw_otp=mne.preprocessing.oversampled_temporal_projection(raw,duration=1,picks=picks)
        raw_otp.save(file_out,overwrite=True)

def do_sss(subject,st_correlation,int_order):
    root_path='/media/tzcheng/storage2/CBS/'
    os.chdir(root_path)
    params = mnefun.Params(n_jobs=6, n_jobs_mkl=1, proj_sfreq=200, n_jobs_fir='cuda',
                       n_jobs_resample='cuda', filter_length='auto')

    params.subjects = [subject]

    params.work_dir = '/media/tzcheng/storage2/CBS/'
    params.run_names = ['%s_01_otp','%s_02_otp'] # ['%s_01_otp','%s_02_otp'] for the adults and ['%s_01_otp'] for the infants
    params.runs_empty = ['%s_erm_otp']
    params.subject_indices = [0] #to run individual participants
    #params.subject_indices = np.arange(0,len(params.subjects)) #to run all subjects


    params.plot_drop_logs = True  # turn off for demo or plots will block
    #params.sws_ssh = 'christina@kasga.ilabs.uw.edu'
    #params.sws_dir = '/data05/christina'

    # SSS options
    params.sss_type = 'python'
    params.hp_type = 'python'
    params.sss_regularize = 'in'
    params.trans_to = 'twa'
    params.cal_file = 'sss_cal_truix.dat'
    params.ct_file = 'ct_sparse_triux2.fif'
    params.coil_t_window = 'auto'  # use the smallest reasonable window size
    params.st_correlation = st_correlation # 0.98 for adults and 0.9 for infants
    params.int_order = int_order # 8 for adults and 6 for infants
    params.movecomp = 'inter'
    # params.mf_prebad['cbs_A101'] = ['MEG0122', 'MEG0333', 'MEG1612', 'MEG1643']
    params.mf_prebad = {
    'cbs_A101': ['MEG0122', 'MEG0333', 'MEG1612', 'MEG1643'],
    'cbs_A103': ['MEG0122', 'MEG0333', 'MEG1612', 'MEG1643', 'MEG0911'],
    'cbs_A104': ['MEG0122', 'MEG0333', 'MEG1612', 'MEG1643'],
    'cbs_A105': ['MEG0122', 'MEG0333', 'MEG1612', 'MEG1643'],
    'cbs_A106': ['MEG0122', 'MEG0333', 'MEG1612', 'MEG1643'],
    'cbs_A107': ['MEG0122', 'MEG0333', 'MEG1612', 'MEG1643'],
    'cbs_A108': ['MEG0122', 'MEG0333', 'MEG1612', 'MEG1643', 'MEG0911'],
    'cbs_A109': ['MEG0122', 'MEG0333', 'MEG1612', 'MEG1643', 'MEG0313'],
    'cbs_A110': ['MEG0122', 'MEG0333', 'MEG1612', 'MEG1643'],
    'cbs_A111': ['MEG0122', 'MEG0333', 'MEG1612', 'MEG1643', 'MEG2012'],
    'cbs_A117': ['MEG0122', 'MEG0333', 'MEG1612', 'MEG1643'],
    'cbs_A118': ['MEG0122', 'MEG0333', 'MEG1612', 'MEG1643', 'MEG2012'],
    'cbs_A119': ['MEG0122', 'MEG0333', 'MEG1612', 'MEG1643'],
    'cbs_A121': ['MEG0122', 'MEG0333', 'MEG1612', 'MEG1643', 'MEG2622'],
    'cbs_A122': ['MEG0122', 'MEG0333', 'MEG1612', 'MEG1643'],
    'cbs_A123': ['MEG0122', 'MEG0333', 'MEG1612', 'MEG1643'],
    'cbs_b101': ['MEG0312', 'MEG1712', 'MEG1831', 'MEG1841', 'MEG2021', 'MEG2231'],
    'cbs_b102': ['MEG0312', 'MEG1712'],
    'cbs_b103': ['MEG0312', 'MEG1712'],
    'cbs_b106': ['MEG0312', 'MEG1712'],
    'cbs_b107': ['MEG0312', 'MEG1712', 'MEG0441'],
    'cbs_b110': ['MEG0312', 'MEG1712', 'MEG1942'],
    'cbs_b111': ['MEG0312', 'MEG1712'],
    'cbs_b112': ['MEG0312', 'MEG1712'],
    'cbs_b113': ['MEG0312', 'MEG1712'],
    'cbs_b114': ['MEG0312', 'MEG1712', 'MEG2612'],
    'cbs_b116': ['MEG0312', 'MEG1712'],
    'cbs_b117': ['MEG0312', 'MEG1712', 'MEG2011', 'MEG2241'],
    'cbs_b118': ['MEG0312', 'MEG1712']
    }
    # make sure you cd to the working directory that have ct and cal files
    mnefun.do_processing(
        params,
        do_score=False,  # do scoring
        fetch_raw=False,  # Fetch raw recording files from acq machine XX
        do_sss=True,  # Run SSS remotely
        gen_ssp=False,  # Generate SSP vectors XX(ECG)
        apply_ssp=False,  # Apply SSP vectors and filtering XX
        write_epochs=False,  # Write epochs to disk XX200
        gen_covs=False,  # Generate covariances
        gen_fwd=False,  # Generate forward solutions (do co-registration first)
        gen_inv=False,  # Generate inverses
        gen_report=False,
        print_status=False,
    )

def do_projection(subject, run):
    ###### cleaning with ecg and eog projection
    root_path = os.getcwd()
    file_in=root_path + '/' + subject + '/sss_fif/' + subject + run + '_otp_raw_sss'
    file_out=file_in + '_proj'
    raw = mne.io.read_raw_fif(file_in + '.fif',allow_maxshield=True,preload=True)
    fname_erm = root_path + '/' + subject + '/sss_fif/' + subject + '_erm_otp_raw_sss'
    fname_erm_out = root_path + '/' + subject + '/sss_fif/' + subject + run + '_erm_raw_sss_proj'
    raw_erm = mne.io.read_raw_fif(fname_erm + '.fif',allow_maxshield=True,preload=True)
        
    ecg_projs, ecg_events = mne.preprocessing.compute_proj_ecg(raw, ch_name='ECG001', n_grad=1, n_mag=1, reject=None)
    ecg_epochs = mne.preprocessing.create_ecg_epochs(raw,ch_name='ECG001').average() # don't really need to assign the ch_name
    eog_projs, eog_events = mne.preprocessing.compute_proj_eog(raw, ch_name=['EOG002','EOG003'], n_grad=1, n_mag=1, reject=None)
    eog_epochs = mne.preprocessing.create_eog_epochs(raw,ch_name=['EOG002','EOG003']).average() ## 

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
        data.filter(l_freq=hp,h_freq=lp,method='iir',iir_params=dict(order=4,ftype='butter'))
    else:
        data.filter(l_freq=0,h_freq=lp,method='iir',iir_params=dict(order=4,ftype='butter'))
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

def do_epoch_mmr(data, subject, run, direction):
    ###### Read the event files to do epoch    
    root_path = os.getcwd()
    file_out = root_path + '/' + subject + '/sss_fif/' + subject + run + '_otp_raw_sss_proj_fil50'
    
    reject=dict(grad=4000e-13,mag=4e-12)
    picks = mne.pick_types(data.info,meg=True,eeg=False) 
    
    if direction == 'ba_to_pa':
        mmr_events = mne.read_events(root_path + '/' + subject + '/events/' + subject + run + '_events_mmr-eve.fif')
        event_id = {'Standard':1,'Deviant1':3,'Deviant2':6}
        
        epochs_cortical = mne.Epochs(data, mmr_events, event_id,tmin =-0.1, tmax=0.6,baseline=(-0.1,0),preload=True,proj=True,reject=reject,picks=picks)

        evoked_substd=epochs_cortical['Standard'].average()
        evoked_dev1=epochs_cortical['Deviant1'].average()
        evoked_dev2=epochs_cortical['Deviant2'].average()

        epochs_cortical.save(file_out + '_mmr_e.fif',overwrite=True)
        evoked_substd.save(file_out + '_evoked_substd_mmr.fif',overwrite=True)
        evoked_dev1.save(file_out + '_evoked_dev1_mmr.fif',overwrite=True)
        evoked_dev2.save(file_out + '_evoked_dev2_mmr.fif',overwrite=True)
    
    elif direction == 'pa_to_ba':
        mmr_events = mne.read_events(root_path + '/' + subject + '/events/' + subject + run + '_events_mmr_reverse-eve.fif')
        event_id = {'Standard1':3,'Standard2':6,'Deviant':1}
        
        epochs_cortical = mne.Epochs(data, mmr_events, event_id,tmin =-0.1, tmax=0.6,baseline=(-0.1,0),preload=True,proj=True,reject=reject,picks=picks)

        evoked_substd1=epochs_cortical['Standard1'].average()
        evoked_substd2=epochs_cortical['Standard2'].average()
        evoked_dev=epochs_cortical['Deviant'].average()

        epochs_cortical.save(file_out + '_mmr_reverse_e.fif',overwrite=True)
        evoked_substd1.save(file_out + '_evoked_substd1_reverse_mmr.fif',overwrite=True)
        evoked_substd2.save(file_out + '_evoked_substd2_reverse_mmr.fif',overwrite=True)
        evoked_dev.save(file_out + '_evoked_dev_reverse_mmr.fif',overwrite=True)
    
    elif direction == 'first_last_ba':
        mmr_events = mne.read_events(root_path + '/' + subject + '/events/' + subject + run + '_events_mmr_firstlastba-eve.fif')
        event_id = {'Standard':1,'Deviant1':31,'Deviant2':61}
        
        epochs_cortical = mne.Epochs(data, mmr_events, event_id,tmin =-0.1, tmax=0.6,baseline=(-0.1,0),preload=True,proj=True,reject=reject,picks=picks)

        evoked_substd=epochs_cortical['Standard'].average()
        evoked_dev1=epochs_cortical['Deviant1'].average()
        evoked_dev2=epochs_cortical['Deviant2'].average()

        epochs_cortical.save(file_out + '_mmr_ba_e.fif',overwrite=True)
        evoked_substd.save(file_out + '_evoked_substd_ba_mmr.fif',overwrite=True)
        evoked_dev1.save(file_out + '_evoked_dev1_ba_mmr.fif',overwrite=True)
        evoked_dev2.save(file_out + '_evoked_dev2_ba_mmr.fif',overwrite=True)

def do_epoch_mmr_eeg(data, subject, run, direction):
    ###### Read the event files to do epoch    
    root_path = os.getcwd()    
    reject=dict(bio=100e-6)
    file_out = root_path + '/' + subject + '/eeg/' + subject + run 

 
    if direction == 'ba_to_pa':
        mmr_events = mne.read_events(root_path + '/' + subject + '/events/' + subject + run + '_events_mmr-eve.fif')
        event_id = {'Standard':1,'Deviant1':3,'Deviant2':6}
        
        epochs_cortical = mne.Epochs(data, mmr_events, event_id,tmin =-0.1, tmax=0.6,baseline=(-0.1,0),preload=True,proj=True,reject=reject)

        evoked_substd=epochs_cortical['Standard'].average(picks=('bio'))
        evoked_dev1=epochs_cortical['Deviant1'].average(picks=('bio'))
        evoked_dev2=epochs_cortical['Deviant2'].average(picks=('bio'))

        epochs_cortical.save(file_out + '_mmr_e.fif',overwrite=True)
        evoked_substd.save(file_out + '_evoked_substd_mmr.fif',overwrite=True)
        evoked_dev1.save(file_out + '_evoked_dev1_mmr.fif',overwrite=True)
        evoked_dev2.save(file_out + '_evoked_dev2_mmr.fif',overwrite=True)
    
    elif direction == 'pa_to_ba':
        mmr_events = mne.read_events(root_path + '/' + subject + '/events/' + subject + run + '_events_mmr_reverse-eve.fif')
        event_id = {'Standard1':3,'Standard2':6,'Deviant':1}
        
        epochs_cortical = mne.Epochs(data, mmr_events, event_id,tmin =-0.1, tmax=0.6,baseline=(-0.1,0),preload=True,proj=True,reject=reject)

        evoked_substd1=epochs_cortical['Standard1'].average(picks=('bio'))
        evoked_substd2=epochs_cortical['Standard2'].average(picks=('bio'))
        evoked_dev=epochs_cortical['Deviant'].average(picks=('bio'))

        epochs_cortical.save(file_out + '_mmr_reverse_e.fif',overwrite=True)
        evoked_substd1.save(file_out + '_evoked_substd1_reverse_mmr.fif',overwrite=True)
        evoked_substd2.save(file_out + '_evoked_substd2_reverse_mmr.fif',overwrite=True)
        evoked_dev.save(file_out + '_evoked_dev_reverse_mmr.fif',overwrite=True)
    
    elif direction == 'first_last_ba':
        mmr_events = mne.read_events(root_path + '/' + subject + '/events/' + subject + run + '_events_mmr_firstlastba-eve.fif')
        event_id = {'Standard':1,'Deviant1':31,'Deviant2':61}
        
        epochs_cortical = mne.Epochs(data, mmr_events, event_id,tmin =-0.1, tmax=0.6,baseline=(-0.1,0),preload=True,proj=True,reject=reject)

        evoked_substd=epochs_cortical['Standard'].average(picks=('bio'))
        evoked_dev1=epochs_cortical['Deviant1'].average(picks=('bio'))
        evoked_dev2=epochs_cortical['Deviant2'].average(picks=('bio'))

        epochs_cortical.save(file_out + '_mmr_ba_e.fif',overwrite=True)
        evoked_substd.save(file_out + '_evoked_substd_ba_mmr.fif',overwrite=True)
        evoked_dev1.save(file_out + '_evoked_dev1_ba_mmr.fif',overwrite=True)
        evoked_dev2.save(file_out + '_evoked_dev2_ba_mmr.fif',overwrite=True)

def do_epoch_cabr(data, subject, run, n_trials ,hp,lp):  
    ###### Read the event files (generated from evtag.py) 
    root_path = os.getcwd()
    cabr_events = mne.read_events(root_path + '/' + subject + '/events/' + subject + run + '_events_cabr-eve.fif')
    file_out = root_path + '/' + subject + '/sss_fif/' + subject + run + '_otp_raw_sss_proj_f' + str(hp) + str(lp)
    
    event_id = {'Standardp':1,'Standardn':2, 'Deviant1p':3,'Deviant1n':5, 'Deviant2p':6,'Deviant2n':7}
    
    reject=dict(grad=4000e-13,mag=4e-12)
    picks = mne.pick_types(data.info,meg=True,eeg=False) 
    epochs = mne.Epochs(data, cabr_events, event_id,tmin =-0.02, tmax=0.2, baseline=(-0.02,0),reject=reject,picks=picks)
    new_epochs = epochs.copy().drop_bad()
    
    ## match the trial number for each sound
    ## get random number of sounds from all sounds
    ## neet to find p and n len after dropping bad, use the smaller one to be the full len
    if n_trials == 'all':
        evoked_substd=epochs['Standardp','Standardn'].average()
        evoked_dev1=epochs['Deviant1p','Deviant1n'].average()
        evoked_dev2=epochs['Deviant2p','Deviant2n'].average()
    else:
        rand_ind = random.sample(range(min(len(new_epochs['Standardp'].events),len(new_epochs['Standardn'].events))),n_trials//2) 
        evoked_substd_p=epochs['Standardp'][rand_ind].average()
        evoked_substd_n=epochs['Standardn'][rand_ind].average()
        evoked_substd = mne.combine_evoked([evoked_substd_p,evoked_substd_n], weights='equal')
        del rand_ind
    
        rand_ind = random.sample(range(min(len(new_epochs['Deviant1p'].events),len(new_epochs['Deviant1n'].events))),n_trials//2) 
        evoked_dev1_p=new_epochs['Deviant1p'][rand_ind].average()
        evoked_dev1_n=new_epochs['Deviant1n'][rand_ind].average()
        evoked_dev1 = mne.combine_evoked([evoked_dev1_p,evoked_dev1_n], weights='equal')
        del rand_ind
    
        rand_ind = random.sample(range(min(len(new_epochs['Deviant2p'].events),len(new_epochs['Deviant2n'].events))),n_trials//2) 
        evoked_dev2_p=new_epochs['Deviant2p'][rand_ind].average()
        evoked_dev2_n=new_epochs['Deviant2n'][rand_ind].average()
        evoked_dev2 = mne.combine_evoked([evoked_dev2_p,evoked_dev2_n], weights='equal')
        
    epochs.save(file_out + '_ffr_e_' + str(n_trials) + '.fif',overwrite=True)
    evoked_substd.save(file_out + '_evoked_substd_ffr_' + str(n_trials) + '.fif',overwrite=True)
    evoked_dev1.save(file_out + '_evoked_dev1_ffr_' + str(n_trials) + '.fif',overwrite=True)
    evoked_dev2.save(file_out + '_evoked_dev2_ffr_' + str(n_trials) + '.fif',overwrite=True)
    return evoked_substd,evoked_dev1,evoked_dev2,epochs

def do_epoch_cabr_eeg(data, subject, run, n_trials):  
    ###### Read the event files (generated from evtag.py) 
    root_path = os.getcwd()
    cabr_events = mne.read_events(root_path + '/' + subject + '/events/' + subject + run + '_events_cabr-eve.fif')
    file_out = root_path + '/' + subject + '/eeg/' + subject + run 
    
    event_id = {'Standardp':1,'Standardn':2, 'Deviant1p':3,'Deviant1n':5, 'Deviant2p':6,'Deviant2n':7}
    
    reject=dict(bio=35e-6)
    epochs = mne.Epochs(data, cabr_events, event_id,tmin =-0.02, tmax=0.2, baseline=(-0.02,0),reject=reject)
    new_epochs = epochs.copy().drop_bad()

    ## match the trial number for each sound
    ## get random number of sounds from all sounds
    ## neet to find p and n len after dropping bad, use the smaller one to be the full len
    if n_trials == 'all':
        evoked_substd=epochs['Standardp','Standardn'].average(picks=('bio'))
        evoked_dev1=epochs['Deviant1p','Deviant1n'].average(picks=('bio'))
        evoked_dev2=epochs['Deviant2p','Deviant2n'].average(picks=('bio'))
    else:
        rand_ind = random.sample(range(min(len(new_epochs['Standardp'].events),len(new_epochs['Standardn'].events))),n_trials//2) 
        evoked_substd_p=new_epochs['Standardp'][rand_ind].average(picks=('bio'))
        evoked_substd_n=new_epochs['Standardn'][rand_ind].average(picks=('bio'))
        evoked_substd = mne.combine_evoked([evoked_substd_p,evoked_substd_n], weights='equal')
        del rand_ind
    
        rand_ind = random.sample(range(min(len(new_epochs['Deviant1p'].events),len(new_epochs['Deviant1n'].events))),n_trials//2) 
        evoked_dev1_p=new_epochs['Deviant1p'][rand_ind].average(picks=('bio'))
        evoked_dev1_n=new_epochs['Deviant1n'][rand_ind].average(picks=('bio'))
        evoked_dev1 = mne.combine_evoked([evoked_dev1_p,evoked_dev1_n], weights='equal')
        del rand_ind
    
        rand_ind = random.sample(range(min(len(new_epochs['Deviant2p'].events),len(new_epochs['Deviant2n'].events))),n_trials//2) 
        evoked_dev2_p=new_epochs['Deviant2p'][rand_ind].average(picks=('bio'))
        evoked_dev2_n=new_epochs['Deviant2n'][rand_ind].average(picks=('bio'))
        evoked_dev2 = mne.combine_evoked([evoked_dev2_p,evoked_dev2_n], weights='equal')

    new_epochs.save(file_out + '_ffr_e_' + str(n_trials) + '.fif',overwrite=True)
    evoked_substd.save(file_out + '_evoked_substd_ffr_' + str(n_trials) + '.fif',overwrite=True)
    evoked_dev1.save(file_out + '_evoked_dev1_ffr_' + str(n_trials) + '.fif',overwrite=True)
    evoked_dev2.save(file_out + '_evoked_dev2_ffr_' + str(n_trials) + '.fif',overwrite=True)
    return evoked_substd,evoked_dev1,evoked_dev2,new_epochs
########################################
root_path='/media/tzcheng/storage2/CBS/'
os.chdir(root_path)

## parameters 
direction = 'ba_to_pa' # traditional direction 'ba_to_pa': ba to pa and ba to mba
# reverse direction 'pa_to_ba' : is pa to ba and mba to ba; 
# only comparing /ba/ 'first_last_ba': only comparing /ba/ before and after habituation 
runs = ['_01'] # ['_01','_02'] for the adults and ['_01'] for the infants
st_correlation = 0.98 # 0.98 for adults and 0.9 for infants
int_order = 8 # 8 for adults and 6 for infants
lp = 450 
hp = 80
do_cabr = True # True: use the cABR filter, cov and epoch setting; False: use the MMR filter, cov and epoch setting

subj = [] # A104 got some technical issue
for file in os.listdir():
    if file.startswith('cbs_b'): # cbs_A for the adults and cbs_b for the infants
        subj.append(file)

#%%##### do the jobs for MEG
n_trials =  200 # can be an integer or 'all' using all the sounds
# randomly select k sounds from each condition
# each trial has 4-8 sounds, there are 100 /ba/ and 50 /pa/ and 50 /mba/ trials
# we have at least 200 sounds for each condition 

for s in subj:
    print(s)
    # do_otp(s)
    # do_sss(s,st_correlation,int_order)
    for run in runs:
        filename = root_path + s + '/sss_fif/' + s + run + '_otp_raw_sss_proj.fif'

        if os.path.exists(filename):
            print ('ECG/EOG projection exists, loading...')
            raw = mne.io.read_raw_fif(filename, allow_maxshield=True,preload=True)
            raw_erm = mne.io.read_raw_fif(root_path + s + '/sss_fif/' + s + run + '_erm_raw_sss_proj.fif', allow_maxshield=True,preload=True)
        else:
            print ('Doing ECG/EOG projection...')
            [raw,raw_erm] = do_projection(s,run)

        print ('Doing filtering...')
        raw_filt = do_filtering(raw,lp,hp,do_cabr)
        raw_erm_filt = do_filtering(raw_erm,lp,hp,do_cabr)
        print ('calculate cov...')
        do_cov(s,raw_erm_filt, do_cabr,hp,lp)
        print ('Doing epoch...')
        if do_cabr == True:
            do_epoch_cabr(raw_filt, s, run, n_trials,hp,lp)
        else:
            do_epoch_mmr(raw_filt, s, run, direction)

#%%##### do the jobs for EEG MMR
# for s in subj:
#     print(s)
#     for run in runs:
#         raw_file=mne.io.Raw('/media/tzcheng/storage/CBS/'+s+'/eeg/'+s+ run +'_raw.fif',allow_maxshield=True,preload=True)
#         raw_file.filter(l_freq=0,h_freq=50,picks=('bio'),method='iir',iir_params=dict(order=4,ftype='butter'))
#         raw_file.pick_channels(['BIO004'])
#         do_epoch_mmr_eeg(raw_file, s, run, direction)

#%%##### do the jobs for EEG FFR

# for s in subj:
#     print(s)
#     for run in runs:
#         raw_file=mne.io.Raw('/media/tzcheng/storage/CBS/'+s+'/eeg/'+s+ run +'_raw.fif',allow_maxshield=True,preload=True)
#         raw_file.notch_filter(np.arange(60,2001,60),filter_length='auto',notch_widths=0.5,picks=('bio'))
#         raw_file.filter(l_freq=80,h_freq=2000,picks=('bio'),method='iir',iir_params=dict(order=4,ftype='butter'))
#         raw_file.pick_channels(['BIO004'])
#         do_epoch_cabr_eeg(raw_file, s, run, n_trials)