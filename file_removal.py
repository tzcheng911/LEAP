#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 14:11:28 2026

Don't use this script if possible. 
Be very very careful when using this script because it can delete all the files very quickly. 

@author: tzcheng
"""

#%% get the files to delete
import os

root = "/media/tzcheng/storage2/CBS/"
keywords = ["test_morph-vl.stc"]

file_to_del = []
subj = [] 
for file in os.listdir():
    if file.startswith('cbs_A'): 
        subj.append(file)
for s in subj:
    print(s)
    folder = root + s + '/sss_fif'
    for fname in os.listdir(folder):
        if any(k in fname for k in keywords):
            full_path = os.path.join(folder, fname)
            print(full_path)
            if os.path.isfile(full_path):
                os.remove(full_path)


#%% Get the file names from certain folder for housekeeping
folder = '/media/tzcheng/storage/Brainstem/MEG/FFR/decoding'
keywords = ["svmacc_p10n40_pcffr802000_ntrial200_3"]

file_to_doc = []

for fname in os.listdir(folder):
    if any(k in fname for k in keywords):
        file_to_doc.append(fname)
file_to_doc.sort()
print(file_to_doc)
print(len(file_to_doc))