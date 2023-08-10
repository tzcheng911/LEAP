#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 12:01:23 2023
Modified by zcheng on Fri Aug 4 11:00:00 2023
@author: ashdrew

"""
import numpy as np  # noqa, analysis:ignore
import mnefun
import mne
import os

## Load the data
## MNEfun pipeline
params = mnefun.Params(n_jobs=6, n_jobs_mkl=1, proj_sfreq=200, n_jobs_fir='cuda',
                       n_jobs_resample='cuda', filter_length='auto')

params.subjects = [
    'cbs_b101']

params.work_dir = '/media/tzcheng/storage/CBS_test/'
params.run_names = ['%s_01_otp']
# params.runs_empty = ['%s_erm']
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
params.st_correlation = .9 # 0.98 for adults and 0.9 for infants
params.int_order = 6 # 8 for adults and 6 for infants
params.movecomp = 'inter'
params.mf_prebad['cbs_A101'] = ['MEG0122', 'MEG0333', 'MEG1612', 'MEG1643']
params.mf_prebad['cbs_A103'] = ['MEG0122', 'MEG0333', 'MEG0911', 'MEG1612', 'MEG1643']
params.mf_prebad['cbs_A104'] = ['MEG0122', 'MEG0333', 'MEG1612', 'MEG1643']
params.mf_prebad['cbs_b101'] = ['MEG0312', 'MEG1712', 'MEG1841', 'MEG1831','MEG2021', 'MEG2231']

params.mf_prebad['vMMR_901'] = ['MEG0312','MEG2241','MEG1712']

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

