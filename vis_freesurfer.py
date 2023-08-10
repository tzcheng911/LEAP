import mne
import os
import numpy

# could use freeview in the commend line with freesurfer activated as well 
sample_data_folder = '/media/tzcheng/storage/vmmr/vMMR_901/mri/'
subjects_dir = sample_data_folder 
Brain = mne.viz.get_brain_class()
brain = Brain("sample", hemi="lh", surf="pial", subjects_dir=subjects_dir, size=(800, 600))
brain.add_annottion("aparc.a2009s", borders=False)
