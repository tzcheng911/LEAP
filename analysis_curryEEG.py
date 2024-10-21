#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 11:37:59 2024
Analyze the preprocessed EEG files.

@author: tzcheng
"""

import numpy as np
from numpy.linalg import norm
from scipy.stats import pearsonr
from scipy import signal
from scipy.io import wavfile
import pandas as pd
import copy
import random
import mne
from mne_connectivity import spectral_connectivity_epochs, spectral_connectivity_time
from mne_connectivity.viz import plot_connectivity_circle
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pandas

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar
#%%######################################## load EEG data and the wav file 
root_path = '/media/tzcheng/storage/RASP/'
epochs = mne.read_epochs(root_path + 'RASP_pilot6.fif')
data = epochs.get_data()
csvFile = pandas.read_csv(root_path + 'stimuli_all/Wav_Files_by_Block_zc.csv')
ntime = np.shape(data)[-1]
ntr = np.shape(data)[0]

#%% Extract the envelope (only need to be done once)
env_matrix = np.zeros((ntr,2,ntime))
Wn = 50
fs_new = 1000

for ntrial in np.arange(0,ntr,1):  
    print('Extracting envelope from trial ' + str(ntrial))
    if csvFile.loc[ntrial]['Condition'] == 'TA':
        fs, target_audio = wavfile.read(root_path + 'stimuli_all/' + 'alt_' + csvFile.loc[ntrial]['Target Wav'] + '.wav') 
        fs, bg_audio = wavfile.read(root_path + 'stimuli_all/' + csvFile.loc[ntrial]['Background Wav'] + '.wav') 
    elif csvFile.loc[ntrial]['Condition'] == 'BA':        
        fs, target_audio = wavfile.read(root_path + 'stimuli_all/' + csvFile.loc[ntrial]['Target Wav'] + '.wav') 
        fs, bg_audio = wavfile.read(root_path + 'stimuli_all/' + 'alt_' + csvFile.loc[ntrial]['Background Wav'] + '.wav') 
    elif csvFile.loc[ntrial]['Condition'] == 'CT':
        fs, target_audio = wavfile.read(root_path + 'stimuli_all/' + csvFile.loc[ntrial]['Target Wav'] + '.wav') 
        fs, bg_audio = wavfile.read(root_path + 'stimuli_all/' + csvFile.loc[ntrial]['Background Wav'] + '.wav') 
    else: print("something is wrong!")
    
    ## Get the envelope
    # Hilbert 
    target_env = np.abs(signal.hilbert(target_audio))
    bg_env = np.abs(signal.hilbert(bg_audio))
    
    ## Low-pass filtering 
    b, a = signal.butter(4, Wn, fs = fs, btype='lowpass')
    target_env_lp =1* signal.filtfilt(b, a, target_env)
    bg_env_lp =1* signal.filtfilt(b, a, bg_env)
    # plt.figure()
    # plt.plot(target_audio)
    # plt.plot(target_env_lp)
    
    ## Downsample
    num_audio = int((len(target_env_lp)*fs_new)/fs)
    target_env_lp_rs = signal.resample(target_env_lp, num_audio, t=None, axis=0, window=None)
    target_env_lp_rs_crop = target_env_lp_rs[:np.shape(data)[-1]]
    bg_env_lp_rs = signal.resample(bg_env_lp, num_audio, t=None, axis=0, window=None)
    bg_env_lp_rs_crop = bg_env_lp_rs[:ntime]
    env_matrix[ntrial,0,:] = target_env_lp_rs_crop
    env_matrix[ntrial,1,:] = bg_env_lp_rs_crop
np.save(root_path + 'stimuli_all/env_matrix.npy', np.array(env_matrix))

#%%######################################## Calculate cortical tracking
## load the env matrix and the eeg 
cond = [2, 6, 7, 5, 8, 1, 4, 3] # sort this matrix by the presentation order for each subject to match the eeg
root_path = '/media/tzcheng/storage/RASP/'

env_matrix = np.load(root_path + 'stimuli_all/env_matrix.npy')
ind = []
for ncond in cond:
    tmp = csvFile.index[csvFile['Block Number']== (ncond)].tolist()
    ind.extend(tmp)
env_matrix_sort = env_matrix[ind,:,:]

con_methods = ["pli", "plv", "coh"]

# over time
con = spectral_connectivity_time(  # Compute frequency- and time-frequency-domain connectivity measures
    np.tile(data_env,(1,1,1)), # need to be 3D
    method=con_methods,
    # if using cwt_morlet, add cwt_freqs = nfreq = np.array([1,2,3,4,5])
    mode="multitaper",
    n_cycles = 3,
    sfreq=1000,
    fmin=1,
    fmax=30,
    freqs = np.arange(3,30,1),
    faverage=False,
    n_jobs=1,
)

con_res = dict()
for method, c in zip(con_methods, con):
    con_res[method] = c.get_data(output="dense")
EEG_env_conn = np.squeeze(con_res["plv"][:,-2:,:-2,:]) # [tg/bg,channel,freq]

# get the ch index
ch_names = epochs.info['ch_names']

#%%######################################## Visualization
EEG_env_conn_contrast = EEG_env_conn[0,:,:]-EEG_env_conn[1,:,:] # positive: target > bg, negative: bg > target
fig, ax = plt.subplots()
im, cbar = heatmap(EEG_env_conn_contrast, ch_names, con[0].freqs, ax=ax,
                   cmap="jet_r", cbarlabel="EEG env Conn",aspect = 'auto',
                   vmin=-0.5,vmax=0.5)