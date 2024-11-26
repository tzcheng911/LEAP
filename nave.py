#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 18:30:02 2024
Get how many epochs are averaged for the evoked object. 
@author: tzcheng
"""

import mne
import numpy as np
import os

#%%########################################
root_path='/media/tzcheng/storage/ME2_MEG/Zoe_analyses/7mo/' # project dir
# root_path = '/media/tzcheng/storage/BabyRhythm/'
subjects_dir = '/media/tzcheng/storage2/subjects/'

os.chdir(root_path)

## parameters 
subj = [] # A104 got some technical issue

for file in os.listdir():
    if file.startswith('me2_'): #  subject name
        subj.append(file)
runs = ['01','02','03','04']

#%% output the number of epochs averaged to obtain the evoked object
evoked_nave = np.zeros([len(subj),len(runs)])

for ns,s in enumerate(subj):
    for nrun,run in enumerate(runs): 
        print('Extracting ' + s + run + ' data')
        file_in = root_path + s + '/sss_fif/' + s +'_' + run 
        
        ## conventional MMR
        evoked = mne.read_evokeds(file_in+'_otp_raw_sss_ica_fil50_evoked.fif')[0] # evoked conditions
        evoked_nave[ns,nrun] = evoked.nave