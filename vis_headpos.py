###### Import library 
import mne
import matplotlib

###### Read files
head_pos_mnefun = mne.chpi.read_head_pos('/media/tzcheng/storage/vmmr/vMMR_901/raw_fif/vMMR_901_1_raw.pos')
head_pos_mne = mne.chpi.read_head_pos('/media/tzcheng/storage/vmmr/vMMR_901/raw_fif/vMMR_901_1_raw_mne.pos')

mne.viz.plot_head_positions(head_pos_mnefun, mode="traces")
%matplotlib qt

mne.viz.plot_head_positions(head_pos_mne, mode="traces")
%matplotlib qt