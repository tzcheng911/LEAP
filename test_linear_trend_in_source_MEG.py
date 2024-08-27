#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 10:35:54 2024

Test the linear trend observed in cbs and I-LABS studies

@author: tzcheng
"""

###### Import library 
import mne
import matplotlib.pyplot as plt
from mne.preprocessing import maxwell_filter
import numpy as np
import os
import random

#%%##### Parameters
root_path='/media/tzcheng/storage2/CBS/'
subjects_dir = '/media/tzcheng/storage2/subjects/'

os.chdir(root_path)
run = '_01' # ['_01','_02'] for the adults and ['_01'] for the infants
hp = 0.5
baseline=(-0.1,0)

subj = [] # A104 got some technical issue
for file in os.listdir():
    if file.startswith('cbs_A'): # cbs_A for the adults and cbs_b for the infants
        subj.append(file)
s = subj[4]

#%%##### do the jobs for MEG
print(s)
filename = root_path + s + '/sss_fif/' + s + run + '_otp_raw_sss_proj.fif'
raw = mne.io.read_raw_fif(filename, allow_maxshield=True,preload=True)
raw_erm = mne.io.read_raw_fif(root_path + s + '/sss_fif/' + s + run + '_erm_raw_sss_proj.fif', allow_maxshield=True,preload=True)

print ('Doing filtering...')
raw_filt= raw.filter(l_freq=hp,h_freq=50,method='iir',iir_params=dict(order=4,ftype='butter'))
raw_erm_filt= raw_erm.filter(l_freq=hp,h_freq=50,method='iir',iir_params=dict(order=4,ftype='butter'))
        
print ('Calculate cov...')
cov = mne.compute_raw_covariance(raw_erm_filt, tmin=0, tmax=None)
        
print ('Doing epoch...')
reject=dict(grad=4000e-13,mag=4e-12)
picks = mne.pick_types(raw_filt.info,meg=True,eeg=False) 
    
mmr_events = mne.read_events(root_path + '/' + s + '/events/' + s + run + '_events_mmr-eve.fif')
event_id = {'Standard':1,'Deviant1':3,'Deviant2':6}
        
epochs_cortical = mne.Epochs(raw_filt, mmr_events, event_id,tmin =-0.1, tmax=0.6,baseline=None,preload=True,proj=True,reject=reject,picks=picks)
evoked_substd=epochs_cortical['Standard'].average()
evoked_dev1=epochs_cortical['Deviant1'].average()
evoked_dev2=epochs_cortical['Deviant2'].average()

print ('Doing source reconstruction...')
fwd = mne.read_forward_solution(root_path + s + '/sss_fif/'  + s +  '-fwd.fif')
src=mne.read_source_spaces(subjects_dir + s + '/bem/' + s + '-vol-5-src.fif')
inverse_operator = mne.minimum_norm.make_inverse_operator(epochs_cortical.info, fwd, cov,loose=1,depth=0.8)
standard = mne.minimum_norm.apply_inverse((evoked_substd), inverse_operator, pick_ori = None)

# standard.plot(src=src)
plt.figure()
plt.plot(standard.times,standard.data.mean(0))

plt.figure()
plt.plot(evoked_substd.times,evoked_substd.data.mean(0))

#%%##### Simulate the trend to see how it influences correlation  
from numpy.linalg import norm
from scipy.stats import pearsonr
from scipy import signal
y_min = 0
y_max = 3
n_points = 3501
x = np.random.normal(loc = 0.5*(y_min + y_max), scale = 1, size=n_points)
y = np.linspace(y_min, y_max, n_points) + np.random.normal(size=n_points)
y = np.linspace(y_min, y_max, n_points) + x

plt.figure()
plt.plot(x)
plt.plot(y)

print(pearsonr(x,y))

a = (x - np.mean(x))/np.std(x)
b = (y - np.mean(y))/np.std(y)
a = a / np.linalg.norm(a)
b = b / np.linalg.norm(b)

xcorr = signal.correlate(a,b)
print(max(xcorr))
lags = signal.correlation_lags(len(a),len(b))
lags_time = lags/5000
print(lags_time[np.argmax(abs(xcorr))]) # if negative then the b is shifted left, if positive b is shifted right
