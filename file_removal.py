#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 14:11:28 2026

@author: tzcheng
"""

import os

root = "/media/tzcheng/storage2/CBS/"
keywords = ["_test_"]

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
