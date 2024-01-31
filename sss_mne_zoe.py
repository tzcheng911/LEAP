import numpy as np  # noqa, analysis:ignore
import mnefun
import mne
import os
from mne.preprocessing import maxwell_filter
from mne.preprocessing import find_bad_channels_maxwell
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


## functions 
def get_t_window(raw):
    info = raw.info
    line_freq = info['line_freq'] or 60
    hpi_freqs = mne.chpi.get_chpi_info(info)[0]
    line_freqs = np.arange(line_freq, info['sfreq'] / 3., line_freq)
    all_freqs = np.concatenate((hpi_freqs, line_freqs))
    delta_freqs = np.diff(np.unique(all_freqs))
    t_window = max(5. / all_freqs.min(), 1. / delta_freqs.min())
    t_window = round(1000 * t_window) / 1000.  # round to ms
    t_window = float(t_window)
    return t_window

## Load the data
root_path='/media/tzcheng/storage/ME2_MEG/cHPI_issue/'
subj = 'me2_102_11m'
run = ['01']

raw = mne.io.read_raw_fif(root_path + subj + '/raw_fif/' + subj + '_01_otp_raw.fif',allow_maxshield=True,preload=True)
fine_cal_file = os.path.join(root_path, "sss_cal.dat")
crosstalk_file = os.path.join(root_path, "ct_sparse.fif")

## compute head position
chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(raw)
chpi_locs = mne.chpi.compute_chpi_locs(raw.info, chpi_amplitudes_212)
head_pos_mne = mne.chpi.compute_head_pos(raw.info, chpi_locs, verbose=True)
mne.chpi.write_head_pos(subj + '_01_otp_raw.pos',head_pos_mne)

## detect bad channels
# from the experiment notes
prebad = {'cbs_A101': ['MEG0122', 'MEG0333', 'MEG1612', 'MEG1643'],
'vMMR_901': ['MEG0312','MEG2241','MEG1712'],
'cbs_b101': ['MEG0312', 'MEG1712', 'MEG1841', 'MEG1831','MEG2021', 'MEG2231']}

# t_window can be calculated by def get_t_window(raw)
t_window = get_t_window(raw)
raw = mne.chpi.filter_chpi(raw, t_window=t_window, verbose=False)
raw.info["bads"] = []
raw_check = raw.copy()
auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(
    raw_check,
    cross_talk=crosstalk_file,
    calibration=fine_cal_file,
    h_freq=None,
    return_scores=True,
    verbose=True,
)
print(auto_noisy_chs)  # we should find them!
print(auto_flat_chs)  # none for this dataset

bads = raw.info["bads"] + auto_noisy_chs + auto_flat_chs
raw.info["bads"] = bads
raw.info["bads"] += prebad[subj]  # from recording notes

## Maxwell filtering
# note that destination = 'twa' is not implemented in MNE so there will be error msg? may need to use MNEfun _sss > calc_twa_hp to implement
raw_sss = mne.preprocessing.maxwell_filter(
    raw, head_pos=head_pos, cross_talk=crosstalk_file, calibration=fine_cal_file, bad_condition='warning',
    st_duration=60,destination=None, verbose=True) # check if need to do the head position file 

raw_sss.save(root_path + subj + '/sss_fif/' + subj + '_01_otp_raw_sss_mne.fif',overwrite=True)