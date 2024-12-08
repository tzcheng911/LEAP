#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 23:03:10 2023
Preprocessing for ME2. Could apply to adults too (br_xx), call their prebads.txt. 
See a list of problemetic subjects from notion page.
Focus on getting 7 mo from 100 and 200, 11 mo from 300
Need manually fix:
1. 108_7m, 202_7m, 208_7m, 316_11m only have run 1,2,4
2. me2_104_7m does not have STI001, need to use STI101 to get event timing; sampled at 2000 Hz instead of 1000 Hz -> resample
3. me2_320_11m and me2_324_11m have very small epoch files → dropped all the bad epochs 
4. me2_101_7m doesn't have ECG063, has ECG064
@author: tzcheng
"""

###### Import library 
import mne
# import mnefun
import matplotlib
from mne.preprocessing import maxwell_filter,ICA, corrmap, create_ecg_epochs, create_eog_epochs
import numpy as np
import os

def do_otp(subject):
    #find all the raw files
    runs=['01','02','03','04','erm']
    for run in runs:
        file_in=root_path+'/'+subject+'/raw_fif/'+subject+'_'+run+'_raw.fif'
        file_out=root_path+subject+'_'+run+'_otp_raw.fif'
        raw=mne.io.Raw(file_in,allow_maxshield=True)
        picks=mne.pick_types(raw.info,meg=True,eeg=False,eog=False, ecg=False,exclude='bads')
        raw_otp=mne.preprocessing.oversampled_temporal_projection(raw,duration=1,picks=picks)
        raw_otp.save(file_out,overwrite=True)

def do_sss(subject,st_correlation,int_order):
    params = mnefun.Params(n_jobs=6, n_jobs_mkl=1, proj_sfreq=200, n_jobs_fir='cuda',
                       n_jobs_resample='cuda', filter_length='auto')

    params.subjects = [subject]
    params.work_dir = root_path
    params.run_names = ['%s_01_otp','%s_02_otp','%s_03_otp','%s_04_otp']
    params.runs_empty = ['%s_erm_otp']
    params.subject_indices = [0] #to run individual participants
    #params.subject_indices = np.arange(0,len(params.subjects)) #to run all subjects

    params.plot_drop_logs = True  # turn off for demo or plots will block
    #params.sws_ssh = 'christina@kasga.ilabs.uw.edu'
    #params.sws_dir = '/data05/christina'

    # SSS options
    params.movecomp = 'inter' # 'inter' for standard processing # None for br_04_04 and br_17_04 
    # (need to comment out params.trans_to = 'twa' too) 
    # do it manually by changign this line and Line 40 for different runs 
    params.sss_type = 'python'
    params.hp_type = 'python'
    params.sss_regularize = 'in'
    params.trans_to = 'twa'
    params.cal_file = 'sss_cal.dat'
    params.ct_file = 'ct_sparse.fif'
    params.coil_t_window = 'auto'  # use the smallest reasonable window size
    params.st_correlation = st_correlation # 0.98 for adults and 0.9 for infants
    params.int_order = int_order # 8 for adults and 6 for infants
    
    ## based on the excel runsheet
    # prebad = {
    # 'me2_101_7m': ['MEG1743', 'MEG1842'],
    # 'me2_101_11m': ['MEG1842'],
    # 'me2_102_7m': ['MEG1743', 'MEG1842'],
    # 'me2_102_11m': ['MEG1433'],
    # 'me2_103_7m': ['MEG1743', 'MEG1842'],
    # 'me2_103_11m': ['MEG1733', 'MEG1811', 'MEG1842'],
    # 'me2_104_7m': ['MEG1743', 'MEG1842'],
    # 'me2_104_11m': ['MEG1433'], # 'MEG1453' in mf_prebad['me2_104_11m'] is not a valid channel name -> change to 1433
    # 'me2_106_7m': ['MEG1743', 'MEG1842'],
    # 'me2_106_11m': ['MEG1433'],
    # 'me2_108_7m': ['MEG1743', 'MEG1842'],
    # 'me2_108_11m': ['MEG1433'],
    # 'me2_109_7m': ['MEG1842'],
    # 'me2_110_7m': ['MEG1842'],
    # 'me2_110_11m': ['MEG1433', 'MEG1743', 'MEG1842'],
    # 'me2_111_7m': ['MEG1842'],
    # 'me2_112_7m': ['MEG1842'],
    # 'me2_112_11m': ['MEG1433','MEG1843'],
    # 'me2_113_7m': ['MEG1842'],
    # 'me2_113_11m': ['MEG1433', 'MEG1743', 'MEG1842'],
    # 'me2_114_7m': ['MEG1842'],
    # 'me2_115_7m': ['MEG1842'],
    # 'me2_116_7m': ['MEG1842'],
    # 'me2_116_11m': ['MEG1433', 'MEG1743', 'MEG1842'],
    # 'me2_117_7m': ['MEG1842'],
    # 'me2_118_7m': ['MEG1842'],
    # 'me2_119_7m': ['MEG1842'],
    # 'me2_119_11m': ['MEG1433', 'MEG1743'],
    # 'me2_120_7m': ['MEG1842', 'MEG1431', 'MEG2431'],
    # 'me2_122_7m': ['MEG1842'],
    # 'me2_122_11m':  ['MEG1433', 'MEG1743', 'MEG1842'],vents = mne.find_events(raw_file,stim_channel='STI001') 
    # 'me2_124_7m': ['MEG1842'],
    # 'me2_124_11m': ['MEG1433', 'MEG1743'],
    # 'me2_125_7m': ['MEG1842'],
    # 'me2_127_7m': ['MEG1842'],
    # 'me2_127_11m': ['MEG1433', 'MEG1743', 'MEG1842'],
    # 'me2_128_7m': ['MEG1842'],
    # 'me2_129_7m': ['MEG1842'],
    # 'me2_129_11m': ['MEG1433', 'MEG1743', 'MEG0313'],
    # 'me2_202_7m': ['MEG1433'],
    # 'me2_203_7m': ['MEG1433'],
    # 'me2_204_7m': ['MEG1433','MEG1321','MEG1141','MEG1322','MEG1323'],
    # 'me2_205_7m': ['MEG1433'],
    # 'me2_206_7m': ['MEG1433'],
    # 'me2_207_7m': ['MEG1433'],
    # 'me2_208_7m': ['MEG1433'],
    # 'me2_209_7m': ['MEG1433'],
    # 'me2_211_7m': ['MEG1433'],
    # 'me2_212_7m': ['MEG1433'],
    # 'me2_213_7m': ['Mimport mneEG1433','MEG1743', 'MEG1842'],
    # 'me2_215_7m': ['MEG1433','MEG1743', 'MEG1842'],
    # 'me2_216_7m': ['MEG1433','MEG1743', 'MEG1872'], # 'MEG1872' in mf_prebad['me2_216_7m'] is not a valid channel name -> change to 1842
    # 'me2_216_7m': ['MEG1433','MEG1743', 'MEG1842'], 
    # 'me2_217_7m': ['MEG1433','MEG1743', 'MEG1842'],
    # 'me2_218_7m': ['MEG1433','MEG1743', 'MEG1842', 'MEG2011', 'MEG2041', 'MEG0621'],
    # 'me2_220_7m': ['MEG1433','MEG1743', 'MEG1842'],
    # 'me2_221_7m': ['MEG1433','MEG1743', 'MEG1842'],
    # 'me2_202_11m': ['MEG1433','MEG1743', 'MEG1842'],
    # 'me2_204_11m': ['MEG1433','MEG1243', 'MEG1842'],
    # 'me2_205_11m': ['MEG1433','MEG1743', 'MEG1842'],
    # 'me2_206_11m': ['MEG1433','MEG1753','MEG1243'],
    # 'me2_207_11m': ['MEG1433','MEG1743', 'MEG1842'],
    # 'me2_208_11m': ['MEG1433','MEG1243', 'MEG1842'],
    # 'me2_209_11m': ['MEG1743', 'MEG1843'],
    # 'me2_211_11m': ['MEG1433','MEG1743', 'MEG1842'],
    # 'me2_212_11m': ['MEG1433','MEG1743', 'MEG1842'],
    # 'me2_213_11m': ['MEG1433','MEG1811'],
    # 'me2_215_11m': ['MEG1433','MEG1811'],
    # 'me2_216_11m': ['MEG1433','MEG1811'],
    # 'me2_217_11m': ['MEG1433','MEG1811'],
    # 'me2_218_11m': ['MEG1433','MEG1811'],
    # 'me2_220_11m': ['MEG1433','MEG1811'],
    # 'me2_221_11m': ['MEG1433','MEG1811'],
    # 'me2_301_11m': ['Mimport mneEG1842'],
    # 'me2_302_11m': ['MEG1842'],
    # 'me2_303_11m': ['MEG1842'],
    # 'me2_304_11m': ['MEG1842'],
    # 'me2_305_11m': ['MEG1842'],
    # 'me2_306_11m': ['MEG1842'],
    # 'me2_307_11m': ['MEG1842'],
    # 'me2_308_11m': ['MEG1842'],
    # 'me2_309_11m': ['MEG1743','MEG1842'],
    # 'me2_310_11m': ['MEG1842'],
    # 'me2_311_11m': ['MEG1842'],
    # 'me2_312_11m': ['MEG1842'],
    # 'me2_313_11m': ['MEG1842'],
    # 'me2_314_11m': ['MEG1842'],
    # 'me2_315_11m': ['MEG1842'],
    # 'me2_316_11m': ['MEG1433','MEG1842'],
    # 'me2_318_11m': ['MEG1433','MEG1742','MEG1811','MEG1842'],
    # 'me2_319_11m': ['MEG1433','MEG1743'],
    # 'me2_320_11m': ['MEG1433','MEG1743'],
    # 'me2_321_11m': ['MEG1433','MEG1743','MEG1842'],
    # 'me2_322_11m': ['MEG1433','MEG1743','MEG1842'],
    # 'me2_323_11m': ['MEG1433','MEG1'_01','_03',743','MEG1842'],
    # 'me2_324_11m': ['MEG1433','MEG1743','MEG1842'],
    # 'me2_325_11m': ['MEG1433','MEG1743','MEG1842'],
    # 'me2_326_11m': ['MEG1433','MEG1743','MEG1842'],
    # }
    
    # params.mf_prebad = prebad
    prebad = open(root_path + subject + '/raw_fif/' + subject + '_prebad.txt').read().split()
    params.mf_prebad[subject] = prebad
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
    file_in=root_path + '/' +subject + '/sss_fif/' + subject + run + '_otp_raw_sss'
    file_out=file_in + '_proj'
    raw = mne.io.read_raw_fif(file_in + '.fif',allow_maxshield=True,preload=True)
    fname_erm = root_path + '/' + subject + '/sss_fif/' + subject + '_erm_otp_raw_sss'
    fname_erm_out = root_path + '/' + subject + '/sss_fif/' + subject +run + '_erm_raw_sss_proj'
    raw_erm = mne.io.read_raw_fif(fname_erm + '.fif',allow_maxshield=True,preload=True)
    
    if subject == 'me2_101_7m':
        ECG_ch = 'ECG064'
    else: 
        ECG_ch = 'ECG063'
    # me2_101_7m used ECG064, the rest used ECG063
    ecg_projs, ecg_events = mne.preprocessing.compute_proj_ecg(raw, ch_name=ECG_ch, n_grad=1, n_mag=1, reject=None)
    # ecg_epochs = mne.preprocessing.create_ecg_epochs(raw,ch_name=ECG_ch).average() # don't really need to assign the ch_name
    # eog_projs, eog_events = mne.preprocessing.compute_proj_eog(raw, ch_name=['EOG002'], n_grad=1, n_mag=1, reject=None) ## adult ['EOG002','EOG003'], infant ['EOG002']
    # eog_epochs = mne.preprocessing.create_eog_epochs(raw,ch_name=['EOG002']).average() ## adult ['EOG002','EOG003'], infant ['EOG002']

    raw.add_proj(ecg_projs)
    # raw.add_proj(eog_projs)
    raw_erm.add_proj(ecg_projs)
    # raw_erm.add_proj(eog_projs)

    raw.save(file_out + '.fif',overwrite = True)
    raw_erm.save(fname_erm_out + '.fif',overwrite = True)

    return raw, raw_erm

def do_ica(subject, run):
    ###### cleaning with ICA
    root_path = os.getcwd()
    file_in=root_path + '/' +subject + '/sss_fif/' + subject + run + '_otp_raw_sss'
    file_out=file_in + '_ica'
    raw = mne.io.read_raw_fif(file_in + '.fif',allow_maxshield=True,preload=True)
    fname_erm = root_path + '/' + subject + '/sss_fif/' + subject + '_erm_otp_raw_sss'
    fname_erm_out = root_path + '/' + subject + '/sss_fif/' + subject +run + '_erm_raw_sss_ica'
    raw_erm = mne.io.read_raw_fif(fname_erm + '.fif',allow_maxshield=True,preload=True)
   
    ica = ICA(n_components=15, max_iter="auto", random_state=97)
    ica.fit(raw)
    ica.exclude = []
    ecg_indices, ecg_scores = ica.find_bads_ecg(raw) # find which ICs match the ECG pattern
    ica.exclude = ecg_indices    
    ica.apply(raw)
    ica.apply(raw_erm)
    raw.save(file_out + '.fif',overwrite = True)
    raw_erm.save(fname_erm_out + '.fif',overwrite = True)
    return raw, raw_erm

def do_filtering(subject, data, lp, run):
    ###### filtering
    root_path = os.getcwd()
    file_in=root_path + '/' + subject + '/sss_fif/' + subject + run + '_otp_raw_sss_proj'
    file_out=file_in + '_fil50'
    data.filter(l_freq=0,h_freq=lp,method='iir',iir_params=dict(order=4,ftype='butter'))
    return data

def do_cov(subject,data,run):
    ###### noise covariance for each run based on its eog ecg proj
    root_path = os.getcwd()
    fname_erm = root_path + '/' + subject + '/sss_fif/' + subject + run + '_erm_otp_raw_sss_proj_fil50'
    fname_erm_out = fname_erm + '-cov'
    noise_cov = mne.compute_raw_covariance(data, tmin=0, tmax=None)
    mne.write_cov(fname_erm_out + '.fif', noise_cov,overwrite=True)

def do_evtag(raw_file,subj,run):
    if subj == 'me2_104_7m':
        events_all = mne.find_events(raw_file,stim_channel='STI101') 
        events =  events_all[events_all[:,2]==1]
        events[:,2] = 5
    else:
        events = mne.find_events(raw_file,stim_channel='STI001') 
    return events 

def do_epoch(data, subject, run, events):
    root_path = os.getcwd()
    file_out = root_path + '/' + subject + '/sss_fif/' + subject + run + '_otp_raw_sss_proj_fil50'

    ###### Read the event files to do epoch    
    event_id = {'Trial_Onset':5}
    # reject=dict(grad=4000e-13,mag=4e-12) # Christina's MMR criteria
    reject=dict(grad=4000e-13,mag=6e-12) # Zoe's ME2 criteria
    picks = mne.pick_types(data.info,meg=True,eeg=False) 
    epochs_cortical = mne.Epochs(data, events, event_id,tmin =-0.5, tmax=10.5,baseline=(-0.1,0),preload=True,proj=True,reject=reject,picks=picks)
    # epochs_cortical.plot_drop_log()
    evoked=epochs_cortical['Trial_Onset'].average()

    # epochs_cortical.save(file_out + '_epoch.fif',overwrite=True)
    # evoked.save(file_out + '_evoked.fif',overwrite=True)
    epochs_cortical.save(file_out + '_mag6pT_epoch.fif',overwrite=True)
    evoked.save(file_out + '_mag6pT_evoked.fif',overwrite=True)    
    return evoked,epochs_cortical

########################################
root_path='/media/tzcheng/storage/BabyRhythm/'
# root_path='/media/tzcheng/storage/ME2_MEG/Zoe_analyses/7mo/'
os.chdir(root_path)

#%%## parameters 
runs = ['_01','_02','_03','_04'] 
st_correlation = 0.9 # 0.98 for adults and 0.9 for infants
int_order = 6 # 8 for adults and 6 for infants
lp = 50 
subjects = []

for file in os.listdir():
    if file.startswith('br_'): 
        subjects.append(file)

## check if there is prebad txt with the raw data: 49 subjects don't have it
# no_prebadstxt = []

# prebads_exist = os.path.exists(root_path + '/' + s + '/raw_fif/' + s + '_prebad.txt')
# if not prebads_exist:
#     print(s + ' doesnt have prebads txt')
#     no_prebadstxt.append(s)

# subjects = ['me2_108_7m', 'me2_202_7m', 'me2_208_7m', 'me2_316_11m'] # problemetic subjects
# subjects = ['me2_108_11m', 'me2_122_11m'] # the two 11 mo from the 100 that I can use
# subjects = ['me2_103_11m', 'me2_306_11m', 'me2_316_11m', 'me2_322_11m'] # the incomplete but qualified 11 mo

#%%###### do the jobs
for s in subjects:
    print(s)
    # do_otp(s)
    # do_sss(s,st_correlation,int_order)
    for run in runs:
        print ('Doing ECG projection...')
        # [raw,raw_erm] = do_projection(s, run)
        filename = root_path + s + '/sss_fif/' + s + run + '_otp_raw_sss_proj.fif'
        if os.path.exists(filename):
            print ('ECG/EOG projection exists, loading...')
            raw = mne.io.read_raw_fif(filename, allow_maxshield=True,preload=True)
            raw_erm = mne.io.read_raw_fif(root_path + s + '/sss_fif/' + s + run + '_erm_raw_sss_proj.fif', allow_maxshield=True,preload=True)
        else:
            print ('something is wrong')
        if s == 'me2_104_7m':
            print ('Doing resampling...')
            raw = raw.copy().resample(sfreq=1000)
            raw_erm = raw_erm.copy().resample(sfreq=1000)
        # print ('Doing ICA...')
        # [raw,raw_erm] = do_ica(s,run)
        print ('Doing filtering...')
        raw_filt = do_filtering(s, raw,lp,run)
        raw_erm_filt = do_filtering(s, raw_erm,lp,run)
        # print ('calculate cov...')
        # do_cov(s,raw_erm_filt,run)
        print ('Doing epoch...')
        events = do_evtag(raw_filt,s,run)
        evoked, epochs_cortical = do_epoch(raw_filt, s, run, events)

#%%###### do manual sensor rejection
# s = 'me2_305_11m'
# run = runs[1]

# print ('Doing manual sensor rejection...')
# file_in=root_path + '/' + s + '/sss_fif/' + s + run + '_otp_raw_sss_proj'
# raw_file = mne.io.read_raw_fif(file_in + '.fif',allow_maxshield=True,preload=True)
# raw_file.filter(l_freq=0,h_freq=50,method='iir',iir_params=dict(order=4,ftype='butter'))
# events = mne.find_events(raw_file,stim_channel='STI001') 
# event_id = {'Trial_Onset':5}
# reject=dict(grad=4000e-13,mag=6e-12)
# picks = mne.pick_types(raw_file.info,meg=True,eeg=False) 
# epochs_cortical = mne.Epochs(raw_file, events, event_id,tmin =-0.5, tmax=10.5,baseline=(-0.1,0),preload=True,proj=True,reject=reject,picks=picks)
# epochs_cortical.plot_drop_log()
# raw_file.plot()