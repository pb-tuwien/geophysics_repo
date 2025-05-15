# -*- coding: utf-8 -*-
"""
Created on Thu May 15 11:37:56 2025

Utility functions for working with ERT data.

@author: peter balogh @ TU Wien, Research Unit Geophysics
"""

#%% Imports
import matplotlib.axes
import matplotlib.figure
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from typing import Union

#%% Plot pseudosections

def plot_pseudosection(df: pd.DataFrame, column: str = 'rhoa', 
                       ax: Union[matplotlib.axes.Axes, None] = None, label: Union[str, None] = None, 
                       vmin: Union[int, float, None] = None, vmax: Union[int, float, None] = None) -> Union[matplotlib.figure.Figure, None]:
    """
    Plot a pseudosection.
    Needs a Dataframe with the "pseudo-coordinates" in the columns `x` and `y`,
    as well as a column with the data to be plotted. 

    Parameters
    ----------
    df: pd.DataFrame
        The Dataframe with the data
    column: str, optional
        The name of the column with the data itself. The default is 'rhoa'.
    ax: matplotlib.axes.Axes, optional
        The ax which should be used for the plotting. The default is None which creates a new one.
    label: str, optional
        Adds a label to the colorbar. The default is None which sets a standard label.
    vmin: float, optional
        The lower border of the colorbar. The default is None which uses the matplotlib default.
    vmax: float, optional
        The upper border of the colorbar. The default is None which uses the matplotlib default.
    
    Returns
    -------
    matplotlib.figure.Figure or None
    If an ax was provided, then nothing is returned. If the figure was created, it is returned.
    """

    values = df[column].values
    xpos = df['x'].values
    ypos = df['y'].values

    if label is None:
        if 'rhoa' in column:
            label = r'$\rho_a$ ($\Omega$m)'
        elif 'phia' in column:
            label = r'$\phi$ [mrad]'

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    im = ax.scatter(xpos, ypos, c=values, s=50, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)
    cbar.set_label(label)

    ax.invert_yaxis()
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Pseudo depth (m)')
    if ax is None:
        return fig

#%% Class for reading and writing OHM-files

class OHMfile:
    def __init__(self):
        self.data = None
        self.electrodes = None

    @property
    def n_electrodes(self) -> int:
        """
        Number of electrodes.
        """
        if self.electrodes is None:
            print('No electrodes found.')
        else:
            return len(self.electrodes)
    
    @property
    def n_data(self) -> int:
        """
        Number of measurements.
        """
        if self.data is None:
            print('No data found.')
        else:
            return len(self.data)

    @classmethod
    def read(cls, filepath: Union[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Read an OHM-file.
        
        Parameters
        ----------
        filepath: str or Path
            Path to the file which should be read.
        
        Returns
        -------
        pd.DataFrame, pd.DataFrame
        Two dataframes. First the one with the coordinates of the electrodes and then the one with the data.
        
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError('No file found at given path.')

        instance = cls()        
        with open(file=filepath, mode='r') as file:
            n_elec = int(file.readline().strip())
            elec_header = file.readline().strip().replace('#', '').split()
            elec_rows = []
            for _ in range(n_elec):
                line = file.readline()
                if line.strip():
                    elec_rows.append([float(x) for x in line.strip().split()])
            instance.electrodes = pd.DataFrame(elec_rows, columns=elec_header)

            n_data = int(file.readline().strip())
            data_header = file.readline().strip().replace('#', '').split()
            data_rows = []
            for _ in range(n_data):
                line = file.readline()
                if line.strip():
                    data_rows.append([float(x) for x in line.strip().split()])
            instance.data = pd.DataFrame(data_rows, columns=data_header)

        return instance
    
    @classmethod
    def write(cls, filepath: Union[str, Path], 
              data: Union[pd.DataFrame], 
              electrodes: Union[pd.DataFrame]):
        
        filepath = Path(filepath)
        if filepath.exists():
            print('File already exists. Overwritting.')

        instance = cls()
        
        instance.data = data
        instance.electrodes = electrodes
        
        with open(file=filepath, mode='w') as file:
            file.write(str(instance.n_electrodes) + '\n')
            file.write('#' + '\t'.join(electrodes.columns) + '\n')
            for line in range(instance.n_electrodes):
                file.write('\t'.join(map(str, electrodes.iloc[line].values)) + '\n')

            file.write(str(instance.n_data) + '\n')
            file.write('#' + '\t'.join(data.columns) + '\n')
            for line in range(instance.n_data):
                file.write('\t'.join(map(str, data.iloc[line].values)) + '\n')
            file.write('0')
        
        return instance

#%% Extract relevant data from Survey

def extract_data(project, index=0):
    electrodes = project.elec[['x', 'y', 'z']]

    xpos, _, ypos = project.surveys[index]._computePseudoDepth()
    all_data = project.surveys[index].df
    all_data['x'] = xpos
    all_data['y'] = ypos
    recip_data = all_data[all_data['irecip'] > 0]
    recip_data['rhoa'] = recip_data['recipMean'].values * recip_data['K'].values

    data = recip_data[['a', 'b', 'm', 'n', 'x', 'y', 'K']]

    pseudo_data = recip_data[['x', 'y', 'rhoa']]

    return pseudo_data, electrodes, data

def reconstruct_data(data, imputed_data):
    if len(data) != len(imputed_data):
        raise ValueError('Imputed data must have the same amount of entries as the original.')
    
    merged = data.merge(imputed_data, how='left', on=['x', 'y'])
    merged['resist'] = merged['rhoa'] / merged['K']

    return merged[['a', 'b', 'm', 'n', 'resist']]
