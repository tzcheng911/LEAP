
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
epochs = mne.read_epochs(root_path + 'Data/RASP_pilot6.fif')
data = epochs.get_data()
csvFile = pandas.read_csv(root_path + 'stimuli_all/Wav_Files_by_Block_zc.csv')
ntime = np.shape(data)[-1]
ntr = np.shape(data)[0]
nch = np.shape(data)[1]

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
    plt.figure()
    plt.plot(np.linspace(0,len(target_audio)/fs, len(target_audio)),target_audio)
    plt.plot(np.linspace(0,len(target_env_lp)/fs, len(target_env_lp)),target_env_lp)
    plt.legend(['Raw waveform','Envelope'])
    
    ## Downsample
    num_audio = int((len(target_env_lp)*fs_new)/fs)
    target_env_lp_rs = signal.resample(target_env_lp, num_audio, t=None, axis=0, window=None)
    target_env_lp_rs_crop = target_env_lp_rs[:np.shape(data)[-1]]
    bg_env_lp_rs = signal.resample(bg_env_lp, num_audio, t=None, axis=0, window=None)
    bg_env_lp_rs_crop = bg_env_lp_rs[:ntime]
    env_matrix[ntrial,0,:] = target_env_lp_rs_crop
    env_matrix[ntrial,1,:] = bg_env_lp_rs_crop
# np.save(root_path + 'stimuli_all/env_matrix.npy', np.array(env_matrix))

#%%######################################## Calculate cortical tracking
## load the env matrix and the eeg 
block = [2, 6, 7, 5, 8, 1, 4, 3] # sort this matrix by the presentation order for each subject to match the eeg
env_matrix = np.load(root_path + 'stimuli_all/env_matrix.npy')
ind = []
for nblock in block:
    tmp = csvFile.index[csvFile['Block Number']== (nblock)].tolist()
    ind.extend(tmp)
env_matrix_sort = env_matrix[ind,:,:] # sort the env with the presenting order
BA_idx = csvFile.index[csvFile['Condition'] == 'BA'].tolist()
TA_idx = csvFile.index[csvFile['Condition'] == 'TA'].tolist()
NA_idx = csvFile.index[csvFile['Condition'] == 'CT'].tolist()
BA_idx_sort = np.where(np.isin(np.array(ind),np.array(BA_idx)))
TA_idx_sort = np.where(np.isin(np.array(ind),np.array(TA_idx)))
NA_idx_sort = np.where(np.isin(np.array(ind),np.array(NA_idx)))

EEG_env_conn_contrast = np.zeros((ntr,nch,27))
EEG_env_conn_target = np.zeros((ntr,nch,27))
EEG_env_conn_bg = np.zeros((ntr,nch,27))
con_methods = ["pli", "plv", "coh"]
for ntrial in np.arange(0,ntr,1): 
    print('Calculating Conn for trial ' + str(ntrial))
    data_env = np.concatenate((data[ntrial,:,:],env_matrix_sort[ntrial,:,:]),axis=0)
    
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
    temp_conn = np.squeeze(con_res["plv"][:,-2:,:-2,:]) # [tg/bg,channel,freq]
    temp_conn_contrast = temp_conn[0,:,:]-temp_conn[1,:,:] # positive: target > bg, negative: bg > target
    EEG_env_conn_target[ntrial,:,:] = np.squeeze(temp_conn[0,:,:])
    EEG_env_conn_bg[ntrial,:,:] = np.squeeze(temp_conn[1,:,:])
    EEG_env_conn_contrast[ntrial,:,:] = temp_conn_contrast
np.save(root_path + 'EEG_env_conn_target-bg.npy',EEG_env_conn_contrast)
np.save(root_path + 'EEG_env_conn_target.npy',EEG_env_conn_target)
np.save(root_path + 'EEG_env_conn_bg.npy',EEG_env_conn_bg)
# get the ch index
ch_names = epochs.info['ch_names']

#%%######################################## Visualization
fig, ax = plt.subplots()
conn_contrast = np.squeeze(EEG_env_conn_contrast[NA_idx_sort,:,:10].mean(1)) # BA_idx_sort, TA_idx_sort, NA_idx_sort
im, cbar = heatmap(conn_contrast, ch_names, con[0].freqs[:10], ax=ax,
                   cmap="jet_r", cbarlabel="EEG env Conn",aspect = 'auto',
                   vmin=-0.1,vmax=0.1)

fig, ax = plt.subplots()
conn = np.squeeze(EEG_env_conn_bg[BA_idx_sort,:,:].mean(1)) # BA_idx_sort, TA_idx_sort, NA_idx_sort
im, cbar = heatmap(conn, ch_names, con[0].freqs, ax=ax,
                   cmap="jet_r", cbarlabel="EEG env Conn",aspect = 'auto',
                   vmin=0,vmax=0.8)