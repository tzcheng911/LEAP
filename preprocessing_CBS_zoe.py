# -*- coding: utf-8 -*-
"""
Created on Tue Aug 8 11:28:12 2023

Preprocessing for vMMR. Need to have events file ready from evtag.py
Need to change the input file name "cbs_A" to "cbs_b" for infants
Need to manually enter bad channels for sss from the experiment notes. 
Need to change parameters st_correlation and int_order in sss for adult/infants
Didn't save the product from ecg, eog project and filtering to save some space

@author: zcheng
"""

###### Import library 
import mne
import mnefun
import matplotlib
from mne.preprocessing import maxwell_filter
import numpy as np
import os

def do_otp(subject):
    root_path='/media/tzcheng/storage/CBS/'+ subject +'/raw_fif/'
    os.chdir(root_path)
    #find all the raw files
    runs=['01','02'] # with movecomp or no movecomp; A104 has run1,2 and 3 need to preprocess differently
    for run in runs:
        # file_in=root_path+'cbs'+str(subj)+'_'+str(run)+'_raw.fif'
        # file_out=root_path+'cbs'+str(subj)+'_'+str(run)+'_otp_raw.fif'
        file_in=root_path+subject+'_'+run+'_raw.fif'
        file_out=root_path+subject+'_'+run+'_otp_raw.fif'
        raw=mne.io.Raw(file_in,allow_maxshield=True)
        picks=mne.pick_types(raw.info,meg=True,eeg=False,eog=False, ecg=False,exclude='bads')
        raw_otp=mne.preprocessing.oversampled_temporal_projection(raw,duration=1,picks=picks)
        raw_otp.save(file_out)

def do_sss(subject,st_correlation,int_order):

    params = mnefun.Params(n_jobs=6, n_jobs_mkl=1, proj_sfreq=200, n_jobs_fir='cuda',
                       n_jobs_resample='cuda', filter_length='auto')

    params.subjects = [subject]

    params.work_dir = '/media/tzcheng/storage/CBS/'
    params.run_names = ['%s_02_otp','%s_03_otp'] # A104 has run1,2 and 3 need to preprocess differently
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
    'cbs_A111': ['MEG0122', 'MEG0333', 'MEG1612', 'MEG1643', 'MEG0522'],
    'cbs_A114': ['MEG0122', 'MEG0333', 'MEG1612', 'MEG1643'],
    'cbs_A115': ['MEG0122', 'MEG0333', 'MEG1612', 'MEG1643'],
    'cbs_A116': ['MEG0122', 'MEG0333', 'MEG1612', 'MEG1643', 'MEG2012'],
    'cbs_A117': ['MEG0122', 'MEG0333', 'MEG1612', 'MEG1643'],
    'cbs_A118': ['MEG0122', 'MEG0333', 'MEG1612', 'MEG1643', 'MEG2012'],
    'cbs_A119': ['MEG0122', 'MEG0333', 'MEG1612', 'MEG1643'],
    'cbs_A121': ['MEG0122', 'MEG0333', 'MEG1612', 'MEG1643', 'MEG2622'],
    'cbs_A122': ['MEG0122', 'MEG0333', 'MEG1612', 'MEG1643'],
    'cbs_A123': ['MEG0122', 'MEG0333', 'MEG1612', 'MEG1643']
    }
    # make sure you cd to the working directory that have ct and cal files
    mnefun.do_processing(
        params,
        do_score=False,  # do scoring
        fetch_raw=False,  # Fetch raw recording files from acq machine XX
        do_sss=True,  # Run SSS remotely
        gen_ssp=False,  # Generate SSP vectors XX(ECG)
        apply_ssp=False,  # Apply SSP vectors and filtering XX
        write_epochs=False,  # Write epochs to disk XX
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

    return raw, raw_erm

def do_filtering(data, lp):
    ###### filtering
    data.filter(l_freq=0,h_freq=lp,method='iir',iir_params=dict(order=4,ftype='butter'))
    return data

def do_cov(subject,data):
    ###### noise covariance for each run based on its eog ecg proj
    root_path = os.getcwd()
    fname_erm = root_path + '/' + subject + '/sss_fif/' + subject + run + '_erm_otp_raw_sss_proj_fil50'
    fname_erm_out = fname_erm + '-cov'
    noise_cov = mne.compute_raw_covariance(data, tmin=0, tmax=None)
    mne.write_cov(fname_erm_out + '.fif', noise_cov,overwrite=True)

def do_epoch_mmr(data, subject, run):
    ###### Read the event files to do epoch    
    root_path = os.getcwd()
    mmr_events = mne.read_events(root_path + '/' + subject + '/events/' + subject + run + '_events_mmr-eve.fif')
    file_out = root_path + '/' + subject + '/sss_fif/' + subject + run + '_otp_raw_sss_proj_fil50'

    event_id = {'Standard':1,'Deviant1':3,'Deviant2':6}
    
    reject=dict(grad=4000e-13,mag=4e-12)
    picks = mne.pick_types(data.info,meg=True,eeg=False) 
    epochs_cortical = mne.Epochs(data, mmr_events, event_id,tmin =-0.1, tmax=0.6,baseline=(-0.1,0),preload=True,proj=True,reject=reject,picks=picks)

    evoked_substd=epochs_cortical['Standard'].average()
    evoked_dev1=epochs_cortical['Deviant1'].average()
    evoked_dev2=epochs_cortical['Deviant2'].average()

    epochs_cortical.save(file_out + '_mmr_e.fif',overwrite=True)
    evoked_substd.save(file_out + '_evoked_substd_mmr.fif',overwrite=True)
    evoked_dev1.save(file_out + '_evoked_dev1_mmr.fif',overwrite=True)
    evoked_dev2.save(file_out + '_evoked_dev2_mmr.fif',overwrite=True)
    
    return evoked_substd,evoked_dev1,evoked_dev2,epochs_cortical

def do_epoch_cabr(data, subject, run):  
    ###### Read the event files (generated from evtag.py) 
    root_path = os.getcwd()
    cabr_events = mne.read_events(root_path + '/' + subject + '/events/' + subject + run + '_events_cabr-eve.fif')
    file_out = root_path + '/' + subject + '/sss_fif/' + subject + run + '_otp_raw_sss_proj_fil50'
    
    event_id = {'Standardp':1,'Standardn':2, 'Deviant1p':3,'Deviant1n':5, 'Deviant2p':6,'Deviant2n':7}
    
    reject=dict(grad=4000e-13,mag=4e-12)
    picks = mne.pick_types(data.info,meg=True,eeg=False) 
    epochs = mne.Epochs(data, cabr_events, event_id,tmin =-0.01, tmax=0.18, baseline=(-0.01,0),reject=reject,picks=picks)

    evoked_substd=epochs['Standardp','Standardn'].average()
    evoked_dev1=epochs['Deviant1p','Deviant1n'].average()
    evoked_dev2=epochs['Deviant2p','Deviant2n'].average()
    epochs.save(file_out + '_cABR_e.fif',overwrite=True)
    evoked_substd.save(file_out + '_evoked_substd_cabr.fif',overwrite=True)
    evoked_dev1.save(file_out + '_evoked_dev1_cabr.fif',overwrite=True)
    evoked_dev2.save(file_out + '_evoked_dev2_cabr.fif',overwrite=True)
    return evoked_substd,evoked_dev1,evoked_dev2,epochs

########################################
root_path='/media/tzcheng/storage/CBS/'
os.chdir(root_path)

## parameters 
runs = ['_01','_02']
st_correlation = 0.98 # 0.98 for adults and 0.9 for infants
int_order = 8 # 8 for adults and 6 for infants
lp = 50 
subj = [] # A104 got some technical issue
for file in os.listdir():
    if file.startswith('cbs_A'):
        subj.append(file)

###### do the jobs
for s in subj:
    print(s)
    # do_otp
    # do_sss(s,st_correlation,int_order)
    for run in runs:
        print ('Doing ECG/EOG projection...')
        [raw,raw_erm] = do_projection(s,run)
        print ('Doing filtering...')
        raw_filt = do_filtering(raw,lp)
        raw_erm_filt = do_filtering(raw_erm,lp)
        print ('calculate cov...')
        do_cov(s,raw_erm_filt)
        print ('Doing epoch...')
        do_epoch_mmr(raw_filt, s, run)
        do_epoch_cabr(raw_filt, s, run)
