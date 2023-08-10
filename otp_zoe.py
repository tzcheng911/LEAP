### Actually no need to do otp on mmr

import mne
import mnefun
import os
import numpy as np

root_path='/media/tzcheng/storage/vmmr/vMMR_901/raw_fif/'
    #find all the raw files
runs=['1','2','3','4','erm'] 
runs=['1'] 
subj='vMMR_901'

for run in runs:
    file_in=root_path+subj+'_'+run+'_raw.fif'
    file_out=root_path+subj+'_'+run+'_otp_raw.fif'
    raw=mne.io.Raw(file_in,allow_maxshield=True)
    picks=mne.pick_types(raw.info,meg=True,eeg=False,eog=False, ecg=False,exclude='bads')
    raw_otp=mne.preprocessing.oversampled_temporal_projection(raw,duration=1,picks=picks)
    raw_otp.save(file_out)

raw=mne.io.Raw(file_in,allow_maxshield=True)
picks=mne.pick_types(raw.info,meg=True,eeg=False,eog=False, ecg=False,exclude='bads')
raw_otp=mne.preprocessing.oversampled_temporal_projection(raw,duration=1,picks=picks)
raw_otp.save(file_out)
