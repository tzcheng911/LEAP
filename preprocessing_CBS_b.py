# -*- coding: utf-8 -*-
"""
Created on Tue Aug 8 11:28:12 2023

Preprocessing for CBS_b. Need to have events file ready from evtag.py
Need to change the input file name "cbs_A" to "cbs_b" for infants
Need to manually enter bad channels for sss from the experiment notes. 
Need to change parameters st_correlation and int_order in sss for adult/infants
Didn't save the product from ecg, eog project and filtering to save some space
Could be used to run SLD too (change the root path, subject name, add the pre_bads)
1. cbs_b118 emptyroom is sampled at 1000 Hz instead of 5000 Hz
@author: tzcheng
"""

###### Import library 
import mne
import mnefun
import matplotlib
from mne.preprocessing import maxwell_filter
import numpy as np
import os

def do_otp(subject,time):
    # root_path='/media/tzcheng/storage2/CBS/'+ subject +'/raw_fif/'
    root_path='/media/tzcheng/storage2/SLD/MEG/'+ subject +'/raw_fif/'

    os.chdir(root_path)
    #find all the raw files
    runs=['01','erm'] # ['01','02','erm'] for the adults and ['01','erm'] for the infants
    for run in runs:
        # file_in=root_path+'cbs'+str(subj)+'_'+str(run)+'_raw.fif'
        # file_out=root_path+'cbs'+str(subj)+'_'+str(run)+'_otp_raw.fif'
        file_in=root_path+subject+time+'_'+run+'_raw.fif'
        file_out=root_path+subject+time+'_'+run+'_otp_raw.fif'
        raw=mne.io.Raw(file_in,allow_maxshield=True)
        picks=mne.pick_types(raw.info,meg=True,eeg=False,eog=False, ecg=False,exclude='bads')
        raw_otp=mne.preprocessing.oversampled_temporal_projection(raw,duration=1,picks=picks)
        raw_otp.save(file_out,overwrite=True)

def do_sss(subject,st_correlation,int_order,time):
    # root_path='/media/tzcheng/storage2/CBS/'
    root_path='/media/tzcheng/storage2/SLD/MEG/'

    os.chdir(root_path)
    params = mnefun.Params(n_jobs=6, n_jobs_mkl=1, proj_sfreq=200, n_jobs_fir='cuda',
                       n_jobs_resample='cuda', filter_length='auto')

    params.subjects = [subject]

    # params.work_dir = '/media/tzcheng/storage/CBS/'
    params.work_dir = '/media/tzcheng/storage2/SLD/MEG/'
    params.run_names = ['%s' + time + '_01_otp'] # ['%s_01_otp','%s_02_otp'] for the adults and ['%s_01_otp'] for the infants
    params.runs_empty = ['%s' + time + '_erm_otp']
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
    
    t1_prebad = {
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
    'cbs_b901': ['MEG0312', 'MEG2411'],
    'cbs_b902': ['MEG0312', 'MEG1712'],
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
    'cbs_b118': ['MEG0312', 'MEG1712'],
    'sld_105': ['MEG0312', 'MEG1712'],
    'sld_101': ['MEG0312', 'MEG1712'],
    'sld_103': ['MEG0312', 'MEG1712','MEG1013'],
    'sld_102': ['MEG0312', 'MEG1712','MEG1013'],
    'sld_104': ['MEG0312', 'MEG1712'],
    'sld_107': ['MEG0312', 'MEG1712','MEG0921'],
    'sld_108': ['MEG0312', 'MEG1712'],
    'sld_110': ['MEG0312', 'MEG1712'],
    'sld_113': ['MEG0312', 'MEG1712','MEG1831'],
    'sld_112': ['MEG0312', 'MEG1712'],
    'sld_111': ['MEG0312', 'MEG1712'],
    'sld_114': ['MEG0312', 'MEG1712'],
    'sld_115': ['MEG0312', 'MEG1712'],
    'sld_116': ['MEG0312', 'MEG1712'],
    'sld_117': ['MEG0312', 'MEG1712', 'MEG0631'],
    'sld_118': ['MEG0312', 'MEG1712'],
    'sld_119': ['MEG0312', 'MEG1712'],
    'sld_121': ['MEG0312', 'MEG1712'],
    'sld_122': ['MEG0312', 'MEG1712'],
    'sld_123': ['MEG0312', 'MEG1712'],
    'sld_124': ['MEG0312', 'MEG1712','MEG2512', 'MEG2641'],
    'sld_125': ['MEG0312', 'MEG1712','MEG0911', 'MEG1721', 'MEG0642','MEG0441', 'MEG2543'],
    'sld_126': ['MEG0312', 'MEG1712'],
    'sld_127': ['MEG0312', 'MEG1712']
    }
    
    t2_prebad = {
    'sld_101': ['MEG0312', 'MEG1712'],
    'sld_102': ['MEG0312', 'MEG1712'],
    'sld_103': ['MEG0312', 'MEG1712'],
    'sld_104': ['MEG0312', 'MEG1712'],
    'sld_105': ['MEG0312', 'MEG1712'],
    'sld_107': ['MEG0312', 'MEG1712'],
    'sld_108': ['MEG0312', 'MEG1712','MEG2612'],
    'sld_112': ['MEG0312', 'MEG1712','MEG1721', 'MEG0642','MEG0441', 'MEG2543'],
    'sld_113': ['MEG0312', 'MEG1712','MEG0642','MEG2543'],
    }
    
    t3_prebad = {
    'sld_102': ['MEG0312', 'MEG1712'],
    'sld_103': ['MEG0312', 'MEG1712','MEG1133', 'MEG2612', 'MEG2433'],
    'sld_107': ['MEG0312', 'MEG1712'],
    'sld_108': ['MEG0312', 'MEG1712','MEG2512'],
    }
    if time == '_t1':
        params.mf_prebad = t1_prebad
    elif time == '_t2':
        params.mf_prebad = t2_prebad
    elif time == '_t3':
        params.mf_prebad = t3_prebad
    else: 
        print("Check the t1 or t2")
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

def do_projection(subject, run,time):
    ###### cleaning with ecg and eog projection
    if time == 0:
        time =''
    root_path = os.getcwd()
    file_in=root_path + '/' + subject + '/sss_fif/' + subject + time +run + '_otp_raw_sss'
    file_out=file_in + '_proj'
    raw = mne.io.read_raw_fif(file_in + '.fif',allow_maxshield=True,preload=True)
    fname_erm = root_path + '/' + subject + '/sss_fif/' + subject + time + '_erm_otp_raw_sss'
    fname_erm_out = root_path + '/' + subject + '/sss_fif/' + subject + time + run + '_erm_raw_sss_proj'
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

def do_filtering(data, lp, do_cabr):
    ###### filtering
    if do_cabr == True:
        data.notch_filter(np.arange(60,2001,60),filter_length='auto',notch_widths=0.5)
        data.filter(l_freq=80,h_freq=2000,method='iir',iir_params=dict(order=4,ftype='butter'))
    else:
        data.filter(l_freq=0,h_freq=lp,method='iir',iir_params=dict(order=4,ftype='butter'))
    return data

def do_cov(subject,data,time, do_cabr):
    ###### noise covariance for each run based on its eog ecg proj
    if time == 0:
        time =''
    root_path = os.getcwd()
    fname_erm = root_path + '/' + subject + '/sss_fif/' + subject + time +run + '_erm_otp_raw_sss_proj_f'
    if do_cabr == True:     
        fname_erm_out = fname_erm + '_ffr-cov'
    else: 
        fname_erm_out = fname_erm + 'il50_mmr-cov'
    noise_cov = mne.compute_raw_covariance(data, tmin=0, tmax=None)
    mne.write_cov(fname_erm_out + '.fif', noise_cov,overwrite=True)

def do_epoch_mmr(data, subject, run, time, direction):
    ###### Read the event files to do epoch    
    if time == 0:
        time =''
    root_path = os.getcwd()
    mmr_events = mne.read_events(root_path + '/' + subject + '/events/' + subject +time+run + '_events_mmr-eve.fif')
    file_out = root_path + '/' + subject + '/sss_fif/' + subject + time +run + '_otp_raw_sss_proj_fil50'

    event_id = {'Standard':1,'Deviant1':3,'Deviant2':6}
    
    reject=dict(grad=4000e-13,mag=4e-12)
    picks = mne.pick_types(data.info,meg=True,eeg=False) 
    
    
    if direction == 'ba_to_pa':
        mmr_events = mne.read_events(root_path + '/' + subject + '/events/' + subject + time + run + '_events_mmr-eve.fif')
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
        mmr_events = mne.read_events(root_path + '/' + subject + '/events/' + subject + time + run + '_events_mmr_reverse-eve.fif')
        event_id = {'Standard1':3,'Standard2':6,'Deviant':1}
        
        epochs_cortical = mne.Epochs(data, mmr_events, event_id,tmin =-0.1, tmax=0.6,baseline=(-0.1,0),preload=True,proj=True,reject=reject,picks=picks)

        evoked_substd1=epochs_cortical['Standard1'].average()
        evoked_substd2=epochs_cortical['Standard2'].average()
        evoked_dev=epochs_cortical['Deviant'].average()

        epochs_cortical.save(file_out + '_mmr_reverse_e.fif',overwrite=True)
        evoked_substd1.save(file_out + '_evoked_substd1_reverse_mmr.fif',overwrite=True)
        evoked_substd2.save(file_out + '_evoked_substd2_reverse_mmr.fif',overwrite=True)
        evoked_dev.save(file_out + '_evoked_dev_reverse_mmr.fif',overwrite=True)
    

def do_epoch_cabr(data, subject, run,time):  
    ###### Read the event files (generated from evtag.py) 
    if time == 0:
        time =''
    root_path = os.getcwd()
    cabr_events = mne.read_events(root_path + '/' + subject + '/events/' + subject + time + run + '_events_cabr-eve.fif')
    file_out = root_path + '/' + subject + '/sss_fif/' + subject + time + run + '_otp_raw_sss_proj_f'
    
    event_id = {'Standardp':1,'Standardn':2, 'Deviant1p':3,'Deviant1n':5, 'Deviant2p':6,'Deviant2n':7}
    
    reject=dict(grad=4000e-13,mag=4e-12)
    picks = mne.pick_types(data.info,meg=True,eeg=False) 
    epochs = mne.Epochs(data, cabr_events, event_id,tmin =-0.02, tmax=0.2, baseline=(-0.01,0),reject=reject,picks=picks)

    evoked_substd=epochs['Standardp','Standardn'].average()
    evoked_dev1=epochs['Deviant1p','Deviant1n'].average()
    evoked_dev2=epochs['Deviant2p','Deviant2n'].average()
    epochs.save(file_out + '_cABR_e.fif',overwrite=True)
    evoked_substd.save(file_out + '_evoked_substd_cabr.fif',overwrite=True)
    evoked_dev1.save(file_out + '_evoked_dev1_cabr.fif',overwrite=True)
    evoked_dev2.save(file_out + '_evoked_dev2_cabr.fif',overwrite=True)
    return evoked_substd,evoked_dev1,evoked_dev2,epochs

########################################
# root_path='/media/tzcheng/storage2/CBS/'
root_path='/media/tzcheng/storage2/SLD/MEG/'

os.chdir(root_path)

#%%## parameters 
runs = ['_01'] # ['_01','_02'] for the adults and ['_01'] for the infants
time = '_t1' # first time (6 mo) or second time (12 mo) or third time (14mo) coming back, or 0 for cbs
direction = "ba_to_pa"
do_cabr = False # True: use the cABR filter, cov and epoch setting; False: use the MMR filter, cov and epoch setting
st_correlation = 0.9 # 0.98 for adults and 0.9 for infants
int_order = 6 # 8 for adults and 6 for infants
lp = 50 
subjects = []

for file in os.listdir():
    if file.startswith('sld_127'): # cbs_b for the infants, sld for SLD infants
        subjects.append(file)

#%%###### do the jobs
for s in subjects:
    print(s)
    # do_otp(s,time)
    do_sss(s,st_correlation,int_order,time)
    for run in runs:
        if time == 0:
            time = ""
        filename = root_path + s + '/sss_fif/' + s + time + run + '_otp_raw_sss_proj.fif'

        if os.path.exists(filename):
            print ('ECG/EOG projection exists, loading...')
            raw = mne.io.read_raw_fif(filename, allow_maxshield=True,preload=True)
            raw_erm = mne.io.read_raw_fif(root_path + s + '/sss_fif/' + s + time + run + '_erm_raw_sss_proj.fif', allow_maxshield=True,preload=True)
        else:
            print ('Doing ECG/EOG projection...')
            [raw,raw_erm] = do_projection(s,run,time)
        print ('Doing filtering...')
        raw_filt = do_filtering(raw,lp, do_cabr)
        raw_erm_filt = do_filtering(raw_erm,lp, do_cabr)
        print ('calculate cov...')
        do_cov(s,raw_erm_filt,time,do_cabr)
        print ('Doing epoch...')
        if do_cabr == True:
            do_epoch_cabr(raw_filt, s, run, time)
        else:
            do_epoch_mmr(raw_filt, s, run, time, direction)
        del raw,raw_erm,raw_filt,raw_erm_filt

#%%###### get the cbsb_118 cov from the baseline
# epochs = mne.read_epochs(root_path + s + '/sss_fif/' + s + time + run + '_otp_raw_sss_proj_f_cABR_e.fif')
# noise_cov = mne.compute_covariance(epochs, tmin = None, tmax=0.0)
# fname_erm_out = root_path + '/' + s + '/sss_fif/' + s + time +run + '_erm_otp_raw_sss_proj_f_ffr-cov'
# mne.write_cov(fname_erm_out + '.fif', noise_cov,overwrite=True)