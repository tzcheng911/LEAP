# -*- coding: utf-8 -*-
"""
Created on Tue Aug 8 11:28:12 2023

Preprocessing for vMMR. Need to have events file ready from evtag.py
Need to change the input file name "vMMR_" to others if needed
Need to manually enter bad channels for sss from the experiment notes. 
Need to change parameters st_correlation and int_order in sss for adult/infants

@author: zcheng
"""

###### Import library 
import mne
import mnefun
import matplotlib
from mne.preprocessing import maxwell_filter
import numpy as np
import os
def otp_preprocess(subject,st_correlation,int_order):
    root_path='/media/tzcheng/storage/vmmr/'+ subject +'/raw_fif/'
    os.chdir(root_path)
    #find all the raw files
    runs=['_1','_2','_3','_4'] # with movecomp or no movecomp; A104 has run1,2 and 3 need to preprocess differently
    for run in runs:
        file_in=root_path+subject+run+'_raw.fif'
        file_out=root_path+subject+run+'_otp_raw.fif'
        raw=mne.io.Raw(file_in,allow_maxshield=True)
        picks=mne.pick_types(raw.info,meg=True,eeg=False,eog=False, ecg=False,exclude='bads')
        raw_otp=mne.preprocessing.oversampled_temporal_projection(raw,duration=1,picks=picks)
        raw_otp.save(file_out,overwrite = True)

def do_sss(subject):
    params = mnefun.Params(n_jobs=6, n_jobs_mkl=1, proj_sfreq=200, n_jobs_fir='cuda',
                       n_jobs_resample='cuda', filter_length='auto')

    params.subjects = [subject]

    params.work_dir = '/media/tzcheng/storage/vmmr/'
    params.run_names = ['%s_1','%s_2','%s_3','%s_4'] # A104 has run 1,2 and 3 need to preprocess differently
    params.runs_empty = ['%s_erm']
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
    params.mf_prebad = {'vMMR_901': ['MEG0312','MEG2241','MEG1712'],
                        'vMMR_902': ['MEG0312','MEG1712']
    }
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
    file_in=root_path + '/' + subject + '/sss_fif/' + subject + run + '_raw_sss'
    file_out=file_in + '_proj'
    raw = mne.io.read_raw_fif(file_in + '.fif',allow_maxshield=True,preload=True)
    fname_erm = root_path + '/' + subject + '/sss_fif/' + subject + '_erm_raw_sss'
    fname_erm_out = root_path + '/' + subject + '/sss_fif/' + subject + run + '_erm_raw_sss_proj'
    raw_erm = mne.io.read_raw_fif(fname_erm + '.fif',allow_maxshield=True,preload=True)
        
    ecg_projs, ecg_events = mne.preprocessing.compute_proj_ecg(raw, ch_name='ECG001', n_grad=1, n_mag=1, reject=None)
    ecg_epochs = mne.preprocessing.creatcondae_ecg_epochs(raw,ch_name='ECG001').average() # don't really need to assign the ch_name
    eog_projs, eog_events = mne.preprocessing.compute_proj_eog(raw, ch_name=['EOG002','EOG003'], n_grad=1, n_mag=1, reject=None)
    eog_epochs = mne.preprocessing.create_eog_epochs(raw,ch_name=['EOG002','EOG003']).average() ## 

    raw.add_proj(ecg_projs)
    raw.add_proj(eog_projs)
    raw_erm.add_proj(ecg_projs)
    raw_erm.add_proj(eog_projs)

    raw.save(file_out + '.fif',overwrite = True)
    raw_erm.save(fname_erm_out + '.fif',overwrite = True)

def do_filtering(subject, run, lp):
    ###### filtering
    root_path = os.getcwd()
    file_in=root_path + '/' + subject + '/sss_fif/' + subject + run + '_raw_sss_proj'
    file_out=file_in + '_fil' + str(lp)
    raw = mne.io.read_raw_fif(file_in + '.fif',allow_maxshield=True,preload=True)
    
    fname_erm = root_path + '/' + subject + '/sss_fif/' + subject + run + '_erm_raw_sss_proj'
    fname_erm_out = fname_erm + '_fil' + str(lp)
    raw_erm = mne.io.read_raw_fif(fname_erm + '.fif',allow_maxshield=True,preload=True)
    raw.filter(l_freq=0,h_freq=lp,method='iir',iir_params=dict(order=4,ftype='butter'))
    raw_erm.filter(l_freq=0,h_freq=lp,method='iir',iir_params=dict(order=4,ftype='butter'))
    
    raw.save(file_out + '.fif',overwrite = True)
    raw_erm.save(fname_erm_out + '.fif',overwrite = True)

def do_cov(subject, run):
    ###### noise covariance
    root_path = os.getcwd()
    fname_erm = root_path + '/' + subject + '/sss_fif/' + subject + run + '_erm_raw_sss_proj_fil50'
    raw_erm = mne.io.read_raw_fif(fname_erm + '.fif',allow_maxshield=True,preload=True)
    fname_erm_out = fname_erm + '-cov'
    noise_cov = mne.compute_raw_covariance(raw_erm, tmin=0, tmax=None)
    mne.write_cov(fname_erm_out, noise_cov,overwrite = True)

def do_epoch(subject, run):
    ###### Read the event files (generated from evtag.py)
    root_path = os.getcwd()
    mmr_events = mne.read_events(root_path + '/' + subject + '/events/' + subject + run + '_mmr-eve.fif')
    events_dict = {
    "standard": 6,
    "deviant": 4,
    }
    file_in=root_path + '/' + subject + '/sss_fif/' + subject + run + '_raw_sss_proj_fil50'
    raw = mne.io.read_raw_fif(file_in + '.fif',allow_maxshield=True,preload=True)

    # remove blinking trials (can't see)
    eog_events = mne.preprocessing.find_eog_events(raw)
    onsets = eog_events[:, 0] / raw.info["sfreq"] - 0.25
    durations = [0.5] * len(eog_events)
    descriptions = ["bad blink"] * len(eog_events)
    blink_annot = mne.Annotations(
        onsets, durations, descriptions, orig_time=raw.info["meas_date"])
    raw.set_annotations(blink_annot)

    # remove flat or jumping data
    wildch_criteria = dict(
        mag=3000e-15,  # 3000 fT
        grad=3000e-13,  # 3000 fT/cm
        eog=200e-6,
    )  # 200 µV

    flat_criteria = dict(mag=1e-15, grad=1e-13)  # 1 fT  # 1 fT/cm  # 1 µV

    # do epoch
    epochs = mne.Epochs(
        raw,
        mmr_events,
        events_dict,
        tmin=-0.1,
        tmax=0.5,
        baseline=(-0.1,0),
        reject_tmax=0,
        reject=wildch_criteria,
        flat=flat_criteria,
        reject_by_annotation=True,
        preload=True,
    )
    evoked_s = epochs["standard"].average()
    evoked_d = epochs["deviant"].average()
    epochs.save(file_in + '_e.fif',overwrite=True)
    evoked_s.save(file_in + '_evoked_s.fif',overwrite=True)
    evoked_d.save(file_in + '_evoked_d.fif',overwrite=True)

########################################
root_path='/media/tzcheng/storage/vmmr/'
os.chdir(root_path)

## parameters 
runs = ['_1','_2','_3','_4']
st_correlation = 0.98 # 0.98 for adults and 0.9 for infants
int_order = 8 # 8 for adults and 6 for infants
lp = 50 
subj = [] 
for file in os.listdir():
    if file.startswith('vMMR_902'):
        subj.append(file)

###### do the jobs
for s in subj:
    print(s)
    do_sss(s)
    for run in runs:
        print ('Doing ECG/EOG projection...')
        do_projection(s,run)
        print ('Doing filtering...')
        do_filtering(s,run,lp)
        print ('calculate cov...')
        do_cov(s,run)
        print ('Doing epoch...')
        do_epoch(s,run)
        
