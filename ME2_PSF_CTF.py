#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 15:58:31 2026

@author: tzcheng
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import mne

def compare(BG_v, SM_v):
    BG, SM = set(BG_v), set(SM_v)
    return {
        "BG_unique": len(BG), "BG_duplicates": len(BG_v) - len(BG),
        "SM_unique": len(SM), "SM_duplicates": len(SM_v) - len(SM),
        "overlap": len(BG & SM),
        "BG_only": len(BG - SM),
        "SM_only": len(SM - BG),
        "total_unique": len(BG | SM)
    }

root_path = '/media/tzcheng/storage/ME2_MEG/Zoe_analyses/7mo/'
subjects_dir = '/media/tzcheng/storage2/subjects/'
os.chdir(root_path)
fname_aseg= subjects_dir + 'ANTS6-0Months3T/mri/aparc+aseg.mgz'
label_names = mne.get_volume_labels_from_aseg(fname_aseg)
 
#%% test the spread from the cortical SM to the BG
label_v_ind = np.load('/media/tzcheng/storage/scripts_zoe/ROI_lookup_ANTS6-0Months3T.npy', allow_pickle=True)
BG_list_idx = [6,7,20,21]
SM_list_idx = [48,53,55,82,87,89]
BG_list_idx = [6,7]
SM_list_idx = [55]
BG_v = np.concatenate([np.asarray(label_v_ind[i][0]).ravel() for i in BG_list_idx])
SM_v = np.concatenate([np.asarray(label_v_ind[i][0]).ravel() for i in SM_list_idx])
## check for how many v ind is overlapped 
print(compare(BG_v, SM_v))
print(label_names[SM_list_idx[0]])

#%% point-spread functions for the whole-brain
template_subject = 'ANTS6-0Months3T'
SM_spread_to_BG_all = []
BG_spread_to_BG_all = []

subj = [] 
for file in os.listdir():
    if file.startswith('me2_'):
        subj.append(file)
for subject in subj:      
    fname_raw = root_path + subject + '/raw_fif/' + subject + '_03_raw.fif'
    fname_evoked= root_path + subject + '/sss_fif/' + subject + '_03_otp_raw_sss_proj_fil50_mag6pT_evoked.fif'
    fname_cov = root_path + subject + '/sss_fif/' + subject + '_03_erm_otp_raw_sss_proj_fil50-cov.fif'
    fname_trans = root_path + subject + '/sss_fif/' + subject + '_trans.fif'
    fname_fwd = root_path + subject + '/sss_fif/' + subject + '-fwd.fif'
    fname_bem = subjects_dir + template_subject + '/bem/' + template_subject + '-5120-5120-5120-bem-sol.fif'
    fname_aseg= subjects_dir + template_subject + '/mri/aparc+aseg.mgz'
    label_names = mne.get_volume_labels_from_aseg(fname_aseg)
    
    raw = mne.io.Raw(fname_raw, allow_maxshield=True)
    raw.load_data().pick_types(meg=True, eeg=False, stim=True)
    fwd = mne.read_forward_solution(fname_fwd)
    noise_cov = mne.read_cov(fname_cov)  ## read the individual cov
    
    inverse_operator = mne.minimum_norm.make_inverse_operator(raw.info, fwd, noise_cov,loose=1,depth=0.8)
    
    # compute resolution matrix for dSPM
    rm_lor_vol = mne.minimum_norm.make_inverse_resolution_matrix(fwd, inverse_operator, method="dSPM")
    stc_psf_vol = mne.minimum_norm.get_point_spread(rm_lor_vol, fwd["src"], BG_v, mode = 'mean', norm=True)
    # stc_ctf_vol = mne.minimum_norm.get_cross_talk(rm_lor_vol, fwd["src"], BG_v, mode = 'mean', norm=True)
    stc_psf_vol.plot(src = fwd['src'])
    # stc_ctf_vol.plot(src = fwd['src'])
    
    ## metrics to evaluate PSF
    SM_psf = mne.minimum_norm.get_point_spread(rm_lor_vol, fwd["src"], SM_v, mode = 'mean', norm=True).data
    BG_psf = mne.minimum_norm.get_point_spread(rm_lor_vol, fwd["src"], BG_v, mode = 'mean', norm=True).data
    SM_spread_to_BG = SM_psf[BG_v].mean()
    BG_spread_to_BG = BG_psf[BG_v].mean()
    ratio = SM_spread_to_BG/BG_spread_to_BG
    print("PSF from SM to BG/PSF from BG to BG: " + str(ratio))
    SM_spread_to_BG_all.append(SM_spread_to_BG)
    BG_spread_to_BG_all.append(BG_spread_to_BG)
SM_spread_to_BG_all = np.asarray(SM_spread_to_BG_all)
BG_spread_to_BG_all = np.asarray(BG_spread_to_BG_all)

#%% More metrics if needed
import numpy as np
from mne.minimum_norm.spatial_resolution import _rectify_resolution_matrix
from mne import VolSourceEstimate


def _get_vol_locations(src):
    return np.vstack([
        s["rr"][s["inuse"].astype(bool)]
        for s in src
    ])


def resolution_metrics_vol(resmat, src, function,
                           metric="peak_err", threshold=0.5):

    # Full volume coordinates -> only active sources
    locations = np.vstack([
        s["rr"][s["vertno"]] for s in src
    ]) * 1000  # m -> mm

    # Collapse free orientations: (3*n_src, n_src) -> (n_src, n_src)
    resmat = _rectify_resolution_matrix(resmat)

    if function == "ctf":
        resmat = resmat.T

    if metric == "peak_err":
        peak = np.abs(resmat).argmax(axis=0)
        return np.linalg.norm(
            locations - locations[peak], axis=1
        )

    elif metric == "cog_err":
        out = np.empty(len(locations))
        for i in range(len(locations)):
            w = np.abs(resmat[:, i])
            cog = w @ locations / w.sum()
            out[i] = np.linalg.norm(locations[i] - cog)
        return out

    elif metric == "sd_ext":
        out = np.empty(len(locations))
        for i in range(len(locations)):
            d2 = np.sum((locations - locations[i]) ** 2, axis=1)
            w = np.abs(resmat[:, i]) ** 2
            out[i] = np.sqrt(np.sum(d2 * w) / np.sum(w))
        return out

    elif metric == "maxrad_ext":
        out = np.empty(len(locations))
        for i, w in enumerate(np.abs(resmat.T)):
            idx = np.where(w > threshold * w.max())[0]
            out[i] = np.linalg.norm(
                locations[idx] - locations[i], axis=1
            ).max()
        return out

    raise ValueError(f"Unknown metric: {metric}")

## main 
rm = np.linalg.norm(rm_lor_vol.reshape(5981, 3, 5981, 3), axis=(1, 3))
peak_err = resolution_metrics_vol(
    rm, inverse_operator["src"],  function="psf",metric="peak_err"
)

sd_ext = resolution_metrics_vol(
    rm, inverse_operator["src"],  function="psf",metric="sd_ext"
)

def summarize_resolution(peak_err, sd_ext):
    for name, x in [("PLE", peak_err), ("SD", sd_ext)]:
        print(
            f"{name}: "
            f"median={np.median(x):.2f} cm, "
            f"IQR={np.percentile(x, 25):.2f}–"
            f"{np.percentile(x, 75):.2f} cm, "
            f"mean={np.mean(x):.2f} cm"
        )

summarize_resolution(peak_err, sd_ext)
