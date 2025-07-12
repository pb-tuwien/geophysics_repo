# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 16:00:36 2023

@author: lukas
"""

# %% import modules
import os
import sys
from pathlib import Path
# path_to_libs = ("/shares/laigner/gp/Projects/TEM_modeling/PYTHON/")
path_to_libs = ("C:/Users/jakob/Documents/Meine Ordner/TU/Geophysik/GP_Lukas/em_modeling/")

if not path_to_libs in sys.path:
    sys.path.append(path_to_libs)

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from TEM.library.tem_tools.survey import Survey
from TEM.library.tem_tools.survey import Sounding

from TEM.library.utils.TEM_proc_tools import parse_TEMfastFile
from TEM.library.utils.TEM_proc_tools import plot_multiTEMlog
from TEM.library.utils.TEM_proc_tools import plot_TEMsingleFig
from TEM.library.utils.TEM_proc_tools import generate_TEMprotocol


# %% # plotting styles
plt.style.use('ggplot')

shift_sizes = 4
plt.rcParams['axes.labelsize'] = 18 - shift_sizes
plt.rcParams['axes.titlesize'] = 18 - shift_sizes
plt.rcParams['xtick.labelsize'] = 16 - shift_sizes
plt.rcParams['ytick.labelsize'] = 16 - shift_sizes
plt.rcParams['legend.fontsize'] = 15 - shift_sizes

# plt.rcParams['grid.linewidth'] = 5

#%% directions
std_dir = Path(__file__).parents[3]

proj_dir = '00_data'
path = std_dir / proj_dir

path = str(path)+'/'
path2coord = path

# filename = '20240709_sel.tem'
filename = '20250710_txt.tem'


# %% testing
snd_id = 0

data, nLogs, indices_hdr, indices_dat = parse_TEMfastFile(filename, path)
header = data.loc[indices_hdr.start[snd_id]:indices_hdr.end[snd_id]-1]

# path2coord = std_dir + r"01-data/GPS/"
# coordfile_name = r'GemessenePunkteTEMGK.txt'


#%% proc
# sep_TEMfiles(filename, path, matlab=True)
# filename_coord = addCoordinates(filename, coordfile_name,
#                                 path, path2coord,
#                                 show_map=False)
# sep_TEMfiles(filename_coord, path, matlab=True)

# filename_subset = selectSubset(filename, path, start_t=10, end_t=60)

# dmeas = plot_singleTEMlog(filename, path, snd_id=4,
#                           tmin=2, tmax=15500,
#                           Smin=10e-9, Smax=1.5,
#                           Rmin=0.001, Rmax=100,
#                           dpi=300, label_s=12,
#                           logY=False, errBars=True,
#                           errLine=True)
#
# filename_filtered = filterbyError(filename, path, delta=1e-5)

# dmeas = plot_singleTEMlog(filename_filtered, path, snd_id=7,
#                           tmin=2, tmax=15500,
#                           Smin=10e-9, Smax=1.5,
#                           Rmin=0.001, Rmax=100,
#                           dpi=300, label_s=12,
#                           logY=False, errBars=True,
#                           errLine=True)

# plot_multiTEMlog(filename, path, minmaxALL=False,
#                   tmin=1e0, tmax=1e4,
#                   Smin=1e-12, Smax=1e1,
#                   Rmin=1e1, Rmax=1e4,
#                   dpi=300, log_rhoa=True,
#                   errBars=True, errLine=True)

plot_multiTEMlog(filename_subset, path, minmaxALL=False,
                tmin=2, tmax=15500,
                Smin=10e-9, Smax=1.5,
                Rmin=1, Rmax=100,
                dpi=300, label_s=12,
                log_rhoa=False, errBars=True,
                errLine=True)

# plot_TEMsingleFig(filename, path, minmaxALL=False,
#                   tmin=1e0, tmax=1e4,
#                   Smin=1e-12, Smax=1e1,
#                   Rmin=1e1, Rmax=1e4,
#                   dpi=300, ms=4, lw=1.25,
#                   log_rhoa=True, errBars=False,
#                   errLine=True, mkLeg=True, lg_cols=1)

# generate_TEMprotocol(filename, path, sepFiles=False, matlab=True)

# %%
