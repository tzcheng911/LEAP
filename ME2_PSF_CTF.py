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

#%% test the spread from the cortical SM to the BG
label_v_ind = np.load('/media/tzcheng/storage/scripts_zoe/ROI_lookup_ANTS6-0Months3T.npy', allow_pickle=True)
BG_list_idx = [6,7,20,21]
SM_list_idx = [48,53,55,82,87,89]
ventrical_list_idx = [9,10]
BG_v = np.concatenate([np.asarray(label_v_ind[i][0]).ravel() for i in BG_list_idx])
SM_v = np.concatenate([np.asarray(label_v_ind[i][0]).ravel() for i in SM_list_idx])
V_v = np.concatenate([np.asarray(label_v_ind[i][0]).ravel() for i in ventrical_list_idx])
## check for how many v ind is overlapped 
print(compare(BG_v, SM_v))

#%% point-spread functions for the whole-brain
subject = 'me2_101_7m'
template_subject = 'ANTS6-0Months3T'

# compute for one subject for now: need the noise cov and fwd and raw so template brain does not work as well
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
rm_lor_vol = mne.minimum_norm.make_inverse_resolution_matrix(fwd, inverse_operator, pick_ori = None, method="dSPM", lambda2=lambda2)
stc_psf_vol = mne.minimum_norm.get_point_spread(rm_lor_vol, fwd["src"], sources_vol, mode = 'mean', norm=True)
stc_ctf_vol = mne.minimum_norm.get_cross_talk(rm_lor_vol, fwd["src"], sources_vol, mode = 'mean', norm=True)

n = 5981

rm_lor_vol_reduced = np.linalg.norm(
    rm_lor_vol.reshape(n, 3, n, 3),
    axis=(1, 3)
)

print(rm_lor_vol.shape)
print(rm_lor_vol_reduced.shape)

locations = np.linalg.norm(np.vstack([s['rr'][s['vertno']] for s in fwd["src"]])*1000,axis=1)  # -> mm
resmax = np.abs(rm_lor_vol_reduced).argmax(axis=0)
diffloc = locations - locations[resmax]
np.linalg.norm(diffloc, axis=1) 

def _get_all_src_locations(src):
"""Generalized replacement for MNE's internal _get_src_locations,
which is hardcoded to exactly two surface source spaces. This works
for surface, volume, or mixed source spaces by concatenating each
sub-source-space's used vertex coordinates in stacking order —
matching the row/column order of the resolution matrix."""
return np.vstack([s['rr'][s['vertno']] for s in src])  # (n_dipoles, 3), meters

def peak_localization_error(resmat, src, function='ctf'):
"""Same formula as mne.minimum_norm.resolution_metrics(metric='peak_err'),
generalized to mixed/volume source spaces."""
locations = 1000 * _get_all_src_locations(src)  # -> mm
m = resmat.T if function == 'ctf' else resmat
resmax = np.abs(m).argmax(axis=0)
diffloc = locations - locations[resmax]
return np.linalg.norm(diffloc, axis=1)  # per-vertex, mm

def spatial_deviation(resmat, src, function='ctf'):
"""Same formula as metric='sd_ext' (Molins et al., 2008, eq. 12),
generalized to mixed/volume source spaces."""
locations = 1000 * _get_all_src_locations(src)
m = resmat.T if function == 'ctf' else resmat
width = np.empty(m.shape[1])
for i in range(m.shape[1]):
sq_dist = np.sum((locations - locations[i]) ** 2, axis=1)
w = np.abs(m[:, i]) ** 2
width[i] = np.sqrt(np.sum(sq_dist * w) / np.sum(w))
return width





src_vol = fwd["src"]
verttrue_vol = src_vol[0]["vertno"][sources_vol]

# find vertex with maximum in PSF
max_vert_idx, _ = np.unravel_index(stc_psf_vol.data.argmax(), stc_psf_vol.data.shape)
vert_max_ctf_vol = src_vol[0]["vertno"][[max_vert_idx]]

# plot them
brain_psf_vol = stc_psf_vol.plot_3d(
    "me2_215_7m",
    src=fwd["src"],
    views="ven",
    subjects_dir=subjects_dir,
    volume_options=dict(alpha=0.5),
)
brain_psf_vol.add_text(0.1, 0.9, "Volumetric sLORETA PSF", "title", font_size=16)
brain_psf_vol.add_foci(
    verttrue_vol, coords_as_verts=True, scale_factor=1, hemi="vol", color="green"
)
brain_psf_vol.add_foci(
    vert_max_ctf_vol,
    coords_as_verts=True,
    scale_factor=1.25,
    hemi="vol",
    color="black",
    alpha=0.3,
)

## point-spread functions for the ROI label