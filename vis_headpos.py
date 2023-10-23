# -*- coding: utf-8 -*-
"""
Created on Tue Aug 8 11:28:12 2023

For the purpose of comparing mne vs. mnefun

@author: tzcheng
"""

###### Import library 
import mne
import matplotlib

###### Read files
head_pos_mnefun = mne.chpi.read_head_pos('/media/tzcheng/storage/vmmr/vMMR_901/raw_fif/vMMR_901_1_raw.pos')
head_pos_mne = mne.chpi.read_head_pos('/media/tzcheng/storage/vmmr/vMMR_901/raw_fif/vMMR_901_1_raw_mne.pos')

mne.viz.plot_head_positions(head_pos_mnefun, mode="traces")
mne.viz.plot_head_positions(head_pos_mne, mode="traces")