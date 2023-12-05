#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 23:03:10 2023
Preprocessing for ME2. Need to have events file ready from evtag.py
See a list of problemetic subjects from notion page.
Focus on getting 7 mo from 100 and 200, 11 mo from 300
Need manually fix:
1. 108_7m, 202_7m, 208_7m, 316_11m only have run 1,2,4

@author: tzcheng
"""

###### Import library 
import mne
import mnefun
import matplotlib
from mne.preprocessing import maxwell_filter
import numpy as np
import os

def do_otp(subject):
    root_path='/media/tzcheng/storage/ME2_MEG/'+ subject +'/raw_fif/'

    os.chdir(root_path)
    #find all the raw files
    runs=['01','02','03','04','erm']
    for run in runs:
        file_in=root_path+subject+'_'+run+'_raw.fif'
        file_out=root_path+subject+'_'+run+'_otp_raw.fif'
        raw=mne.io.Raw(file_in,allow_maxshield=True)
        picks=mne.pick_types(raw.info,meg=True,eeg=False,eog=False, ecg=False,exclude='bads')
        raw_otp=mne.preprocessing.oversampled_temporal_projection(raw,duration=1,picks=picks)
        raw_otp.save(file_out,overwrite=True)

def do_sss(subject,st_correlation,int_order):
    root_path='/media/tzcheng/storage/ME2_MEG/'+ subject +'/raw_fif/'

    os.chdir(root_path)
    params = mnefun.Params(n_jobs=6, n_jobs_mkl=1, proj_sfreq=200, n_jobs_fir='cuda',
                       n_jobs_resample='cuda', filter_length='auto')

    params.subjects = [subject]

    # params.work_dir = '/media/tzcheng/storage/CBS/'
    params.work_dir = '/media/tzcheng/storage/ME2_MEG/'
    params.run_names = ['%s_01_otp','%s_02_otp','%s_03_otp','%s_04_otp']
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
    params.cal_file = 'sss_cal.dat'
    params.ct_file = 'ct_sparse.fif'
    params.coil_t_window = 'auto'  # use the smallest reasonable window size
    params.st_correlation = st_correlation # 0.98 for adults and 0.9 for infants
    params.int_order = int_order # 8 for adults and 6 for infants
    params.movecomp = 'inter'
    params.mf_prebad = open(root_path + subject + '_prebad.txt').read().split()
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
    file_in=root_path + subject + '/sss_fif/' + subject + '_'+run + '_otp_raw_sss'
    file_out=file_in + '_proj'
    raw = mne.io.read_raw_fif(file_in + '.fif',allow_maxshield=True,preload=True)
    fname_erm = root_path + subject + '/sss_fif/' + subject + '_erm_otp_raw_sss'
    fname_erm_out = root_path + subject + '/sss_fif/' + subject + '_'+run + '_erm_raw_sss_proj'
    raw_erm = mne.io.read_raw_fif(fname_erm + '.fif',allow_maxshield=True,preload=True)
        
    ecg_projs, ecg_events = mne.preprocessing.compute_proj_ecg(raw, ch_name='ECG001', n_grad=1, n_mag=1, reject=None)
    ecg_epochs = mne.preprocessing.create_ecg_epochs(raw,ch_name='ECG001').average() # don't really need to assign the ch_name
    eog_projs, eog_events = mne.preprocessing.compute_proj_eog(raw, ch_name=['EOG002'], n_grad=1, n_mag=1, reject=None) ## adult ['EOG002','EOG003'], infant ['EOG002']
    eog_epochs = mne.preprocessing.create_eog_epochs(raw,ch_name=['EOG002']).average() ## adult ['EOG002','EOG003'], infant ['EOG002']

    raw.add_proj(ecg_projs)
    raw.add_proj(eog_projs)
    raw_erm.add_proj(ecg_projs)
    raw_erm.add_proj(eog_projs)

    raw.save(file_out + '.fif',overwrite = True)
    raw_erm.save(fname_erm_out + '.fif',overwrite = True)

    return raw, raw_erm

def do_filtering(subject, data, lp,run):
    ###### filtering
    root_path = os.getcwd()
    file_in=root_path + subject + '/sss_fif/' + subject + '_'+run + '_otp_raw_sss_proj'
    file_out=file_in + '_fil50'
    data.filter(l_freq=0,h_freq=lp,method='iir',iir_params=dict(order=4,ftype='butter'))
    data.save(file_out + '.fif',overwrite = True)

    return data

def do_cov(subject,data,run):
    ###### noise covariance for each run based on its eog ecg proj
    root_path = os.getcwd()
    fname_erm = root_path + subject + '/sss_fif/' + subject + '_'+run + '_erm_otp_raw_sss_proj_fil50'
    fname_erm_out = fname_erm + '-cov'
    noise_cov = mne.compute_raw_covariance(data, tmin=0, tmax=None)
    mne.write_cov(fname_erm_out + '.fif', noise_cov,overwrite=True)

########################################
root_path='/media/tzcheng/storage/ME2_MEG/'
os.chdir(root_path)

#%%## parameters 
runs = ['_01','_02','_03','_04'] # ['_01','_02'] for the adults and ['_01'] for the infants
st_correlation = 0.9 # 0.98 for adults and 0.9 for infants
int_order = 6 # 8 for adults and 6 for infants
lp = 50 
subjects = []

for file in os.listdir():
    if file.startswith('me2'): 
        subjects.append(file)
subjects = subjects[66:]

subj_11mo = []
for file in os.listdir():
    if file.endswith('11m'): 
        subj_11mo.append(file)

subj_7mo = []
for file in os.listdir():
    if file.endswith('7m'): 
        subj_7mo.append(file)

## check if there is prebad txt with the raw data: 49 subjects don't have it
no_prebadstxt = []
s = subjects[0]
prebads_exist = os.path.exists(root_path + '/' + s + '/raw_fif/' + s + '_prebad.txt')
if not prebads_exist:
    print(s + ' doesnt have prebads txt')
    no_prebadstxt.append(s)

# subjects = ['me2_108_7m', 'me2_202_7m', 'me2_208_7m', 'me2_316_11m'] # problemetic subjects
#%%###### do the jobs
for s in subjects:
    print(s)
    do_otp(s)
    do_sss(s,st_correlation,int_order)
    # for run in runs:
    #     print ('Doing ECG/EOG projection...')
    #     [raw,raw_erm] = do_projection(s,run)
    #     print ('Doing filtering...')
    #     raw_filt = do_filtering(s, raw,lp,run)
    #     raw_erm_filt = do_filtering(s, raw_erm,lp)
    #     print ('calculate cov...')
    #     do_cov(s,raw_erm_filt,run)
    #     print ('Doing epoch...')
    #     # do_epoch_mmr(raw_filt, s, run)
    #     # do_epoch_cabr(raw_filt, s, run)
