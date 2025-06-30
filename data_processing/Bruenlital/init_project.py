# -*- coding: utf-8 -*-
"""
Created on Thu May  6 17:15:32 2021

script to generate a consistent folder structure for TEM measuerement campaign
other campaigns (time or location) in folders outside

@author: lukas
"""
#%% 
import os
import sys
import numpy

# %%
for data_type in ['rawdata/bin/', 'selected/']:
    data_dir = f'./00_data/{data_type}/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

info_dir = './10_info/'
if not os.path.exists(info_dir):
    os.makedirs(info_dir)

for inv_type in ['pyGIMLi/', 'BEL1D']:
    res_dir = f'./30_inv-results/{inv_type}'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    vis_dir = f'./40_vis/{inv_type}'
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

crd_dir = './50_coord/'
if not os.path.exists(crd_dir):
    os.makedirs(crd_dir)

map_dir = './60_map/'
if not os.path.exists(map_dir):
    os.makedirs(map_dir)

for sub in ['inv/', 'proc/', 'vis/']:
    py_dir = f'./90_scripts/{sub}'
    if not os.path.exists(py_dir):
        os.makedirs(py_dir)


# %%
