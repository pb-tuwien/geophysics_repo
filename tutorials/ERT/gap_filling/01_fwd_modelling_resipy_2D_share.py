# -*- coding: utf-8 -*-
"""
Created in 2023
2D modelling of resistivity data
@author: Clemens Moser @ TU Wien, Research Unit Geophysics
"""

#%% Import Packages
import warnings
warnings.filterwarnings('ignore')
import os
import sys
from resipy import Project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Close all figures
plt.close('all')


#%% Survey
# Survey number
survey_nr = 1


#%% Plot properties
# Font size
fs = 12
ff = 'arial'

mpl.rcParams['figure.dpi'] = 300
mpl.rcParams.update({'font.size': fs,
                     'font.family': ff,
                     'axes.linewidth': 1})


#%% Create folders
folders = ['fwd', 'mesh', 'ohm', 'plots/survey' + str(survey_nr)]
for i in range(len(folders)):
    if not os.path.exists(folders[i]):
        os.makedirs(folders[i])

path_plots = 'plots/survey' + str(survey_nr)


#%% Create project and electrode positions
# Create project
k = Project(typ='R2', dirname='./fwd/syn_model' + '1')

# Design electrodes (x,z)
nelec = 64
elec_ar = np.vstack((np.arange(nelec)*1, np.ones(nelec)*100)).T

# Set electrodes
k.setElec(elec_ar)

    
#%% Create a mesh
# Create mesh
cl_factor = 1
cl = 0.25
k.createMesh(typ='trian', cl_factor=cl_factor, cl=cl, show_output=False)

# Get mesh
mesh_df = k.mesh.df

# Show mesh
fig, ax = plt.subplots(figsize=(7,2.5))
k.showMesh(ax=ax)
fig.tight_layout()
fig.savefig('./' + path_plots + '/mesh_fwd.png')


#%% Synthetic model
# Region 1
res1 = 100 # Resistivity (Ohmm)

# Assign values of region 1 to whole mesh as background values
k.mesh.df['res0'] = res1

# Region 2
res2 = 200 # Resistivity (Ohmm)

# Create coordinates of edge points of polygon 3
polygon_pts = np.array([[22.5,100],[22.5,97],[25.5,97],[25.5,100]])

# Create region 3
k.addRegion(polygon_pts, res0=res2, iplot=False)

# Show mesh with different regions
fig, ax = plt.subplots(figsize=(7,2.5))
k.showMesh(ax=ax, attr='res0', colorbar=True, color_map='jet', edge_color=None,
           clabel='Resistivity $\\rho$ ($\Omega$m)')
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Height (m)')
fig.tight_layout()
fig.savefig('./' + path_plots + '/mesh_fwd_res0.png')


#%% Create configuration
# Dipole-dipole (DD)
k.createSequence([('dpdp1', 1, 64)]) # Configuration, skip+1, number of electrodes (use 'wenner_alpha' for the Wenner configuration)
seq = k.sequence

# Print configuration
print(k.sequence)


#%% Forward Modeling
# Error for forward modeling: relative resistance error
relErr = 2 # %

# Forward modeling
k.forward(noise=relErr, iplot=False) 


#%% Pseudosections of synthetic data
# Plot rhoa pseudosection of synthetic data
fig, ax = plt.subplots(figsize=(7,4))
k.showPseudo(ax=ax)
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Pseudodepth (m)')
fig.tight_layout()
fig.savefig('./' + path_plots + '/pseudo_res.png')


#%% Write simulated data in ohm file
# Get data from resipy dataframe
data = k.surveys[0].df

# Coordinates
coord = elec_ar

# Number of electrodes
nelec = len(elec_ar)

# Open new file
filedir = './ohm/syn_data' + '1' + '.ohm'
f = open(filedir, 'w')

# Write number of electordes
f.write(str(nelec) + '\n')

# Write coord
f.write('#x\ty\n')

# Write topography information
for line in range(nelec):
    f.write('%.4f\t%.4f\n' % (coord[line,0], coord[line,1]))

# Write data header
f.write(str(len(data['resist'])) + '\n')
f.write('#a\tb\tm\tn\tr\n')

# Write quadrupoles and measurements
for line in range(len(data['resist'])):
    f.write('%s\t%s\t%s\t%s\t%.8f\n' %
            (data['a'][line],
             data['b'][line],
             data['m'][line],
             data['n'][line],
             data['resist'][line],
             ))

f.close()
