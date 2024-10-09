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
import pandas as pd
import copy
import random
import mne

#%%######################################## load EEG data and the wav file

#%%######################################## Extract the envelope

#%%######################################## Calculate cortical tracking
