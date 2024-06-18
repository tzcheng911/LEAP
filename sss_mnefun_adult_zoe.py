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

# params.subjects = ['cbs_b101']
params.subjects = ['sld_stim_240604']
# params.work_dir = '/media/tzcheng/storage/CBS_test/'
params.work_dir = '/media/tzcheng/storage2/SLD/stim_test/240604/'
params.run_names = ['%s_otp']
# params.runs_empty = ['%s_erm_otp']
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
params.st_correlation = .98 # 0.98 for adults and 0.9 for infants
params.int_order = 8 # 8 for adults and 6 for infants
params.movecomp = 'inter'
params.mf_prebad = {'cbs_A101': ['MEG0122', 'MEG0333', 'MEG1612', 'MEG1643'],
    'cbs_A103': ['MEG0122', 'MEG0333', 'MEG0911', 'MEG1612', 'MEG1643'],
    'cbs_A104': ['MEG0122', 'MEG0333', 'MEG1612', 'MEG1643'],
    'cbs_A105': ['MEG0122', 'MEG0333', 'MEG1612', 'MEG1643'],
    'cbs_A106': ['MEG0122', 'MEG0333', 'MEG1612', 'MEG1643'],
    'cbs_A107': ['MEG0122', 'MEG0333', 'MEG1612', 'MEG1643'],
    'cbs_A108': ['MEG0122', 'MEG0333', 'MEG0911', 'MEG1612', 'MEG1643'],
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
# params.mf_prebad = {'cbs_A101': ['MEG0122', 'MEG0333', 'MEG1612', 'MEG1643'],
# 'vMMR_901': ['MEG0312','MEG2241','MEG1712'],
# 'cbs_b101': ['MEG0312', 'MEG1712', 'MEG1841', 'MEG1831','MEG2021', 'MEG2231']}

params.mf_prebad = {'sld_stim_240604': ['MEG0312', 'MEG1712']}

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

