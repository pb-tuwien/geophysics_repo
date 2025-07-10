# -*- coding: utf-8 -*-
"""
Created on Mon May  8 20:46:22 2023

@author: lukas
"""

# %% import modules
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# add relative path to folder that contains all custom modules

path_to_libs = ("C:/Users/jakob/Documents/Meine Ordner/TU/Geophysik/GP_Lukas/em_modeling/")

if not path_to_libs in sys.path:
    sys.path.append(path_to_libs)

from TEM.library.tem_tools.survey import Survey
from TEM.library.tem_tools.survey import Sounding

from TEM.library.utils.TEM_proc_tools import parse_TEMfastFile
from TEM.library.utils.TEM_proc_tools import plot_multiTEMlog
from TEM.library.utils.TEM_proc_tools import plot_TEMsingleFig
from TEM.library.utils.TEM_proc_tools import generate_TEMprotocol
from TEM.library.utils.TEM_proc_tools import sep_TEMfiles



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
#%% directions
std_dir = Path(__file__).parents[3]

proj_dir = '00_data/selected/'
path = std_dir / proj_dir

path = str(path)+'/'
path2coord = path
filename = '20250710_A.tem'

# %% testing
snd_id = 0

data, nLogs, indices_hdr, indices_dat = parse_TEMfastFile(filename, path)
header = data.loc[indices_hdr.start[snd_id]:indices_hdr.end[snd_id]-1]

# path2coord = std_dir + r"01-data/GPS/"
# coordfile_name = r'GemessenePunkteTEMGK.txt'


# %% proc
# sep_TEMfiles(filename, path, matlab=True)
# you can also add the correct coordinates to the file header if you like
# filename_coord = addCoordinates(filename, coordfile_name,
#                                 path, path2coord,
#                                 show_map=False)

# and split the files into individual sounding files apart (seperate TEM files)
# sep_TEMfiles(filename_coord, path, matlab=True)

# this was used in the beginning to filter the data and store them in a new file
# maybe useful if you want to filter prior to using the inversion script
# filename_subset = selectSubset(filename, path, start_t=10, end_t=60)

# this was used to filter by an error level, yet it was quite buggy and 
# didnt perform well for all different data sets, use with care
# filename_filtered = filterbyError(filename, path, delta=1e-5)


# %% plotting
# dmeas = plot_singleTEMlog(filename, path, snd_id=4,
#                           tmin=2, tmax=15500,
#                           Smin=10e-9, Smax=1.5,
#                           Rmin=0.001, Rmax=100,
#                           dpi=300, label_s=12,
#                           logY=False, errBars=True,
#                           errLine=True)

# dmeas = plot_singleTEMlog(filename_filtered, path, snd_id=7,
#                           tmin=2, tmax=15500,
#                           Smin=10e-9, Smax=1.5,
#                           Rmin=0.001, Rmax=100,
#                           dpi=300, label_s=12,
#                           logY=False, errBars=True,
#                           errLine=True)


# %% plot multiple soundings in separate figures (signal and apparent resistivity)
# plots will be closed and stored to disc automatically
plot_multiTEMlog(filename, path, minmaxALL=False,
                  tmin=1e0, tmax=8e3,                 # plotting limits for the time
                  Smin=1e-9, Smax=1e0,               # plotting limits for the signal
                  Rmin=1e1, Rmax=1e4,                 # plotting limits for the app. res
                  dpi=300, log_rhoa=True,
                  errBars=True, errLine=True)

# plot_multiTEMlog(filename_subset, path, minmaxALL=False,
#                 tmin=2, tmax=15500,
#                 Smin=10e-9, Smax=1.5,
#                 Rmin=1, Rmax=100,
#                 dpi=300, label_s=12,
#                 log_rhoa=False, errBars=True,
#                 errLine=True)

# plot all sounding from the file into one single figure
# (one for the signal and one for the apparent resistivity)
plot_TEMsingleFig(filename, path, minmaxALL=False,   # minmaxALL should automatically defien plot limits; doesnt great, use with care
                  tmin=1e0, tmax=8e3,                 # plotting limits for the time
                  Smin=1e-9, Smax=1e0,               # plotting limits for the signal
                  Rmin=1e1, Rmax=1e4,                 # plotting limits for the app. res
                  dpi=300, ms=4, lw=1.25,
                  log_rhoa=True, errBars=False,
                  errLine=True, mkLeg=True, lg_cols=1)  # lg_cols defines how many columns you want to have in the legend


# %% generate an automatic measurement protocol
generate_TEMprotocol(filename, path, sepFiles=False, matlab=True)

# %%
