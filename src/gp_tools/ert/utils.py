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
import matplotlib
import matplotlib.pyplot as plt
from typing import Union

#%% Plot pseudosections

def plot_pseudosection(df: pd.DataFrame, column: str = 'rhoa', 
                       ax: Union[matplotlib.axes.Axes, None] = None, label: Union[str, None] = None, 
                       vmin: Union[int, float, None] = None, vmax: Union[int, float, None] = None,
                       cmap: str = 'viridis', **kwargs) -> Union[matplotlib.figure.Figure, None]:
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
    cmap: str, optional
        The colormap used for the plot. The default is 'viridis'.
    
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
        return_fig = True
        fig, ax = plt.subplots()
    else:
        return_fig = False
        fig = ax.get_figure()

    im = ax.scatter(xpos, ypos, c=values, s=50, vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)
    cbar.set_label(label)

    ax.invert_yaxis()
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Pseudo depth (m)')
    if return_fig:
        return fig

#%% Extract relevant data from Survey

default_cols: dict = {'resist': 'r'}

def extract_data(project, index=0):
    electrodes = project.elec[['x', 'y', 'z']]

    xpos, _, ypos = project.surveys[index]._computePseudoDepth()
    all_data = project.surveys[index].df
    all_data['x'] = xpos
    all_data['y'] = ypos
    recip_data = all_data[all_data['irecip'] >= 0].copy()
    recip_data['rhoa'] = recip_data['recipMean'] * recip_data['K']

    data = recip_data[['a', 'b', 'm', 'n', 'x', 'y', 'K']]

    pseudo_data = recip_data[['x', 'y', 'rhoa']]

    return pseudo_data, electrodes, data

def format_data(project, index=0, rename_col: Union[bool, dict] = False):
    pseudo_data, electrodes, data = extract_data(project=project, index=index)
    formated_data = data.copy()
    formated_data['resist'] = pseudo_data['rhoa'] / formated_data['K']

    return_df = formated_data[['a', 'b', 'm', 'n', 'resist']].copy()

    if isinstance(rename_col, dict):
        return_df.rename(columns=rename_col, inplace=True)
    elif isinstance(rename_col, bool) and rename_col:
        return_df.rename(columns=default_cols, inplace=True)

    return return_df, electrodes

def reconstruct_data(data, imputed_data, rename_col: Union[bool, dict] = False):
    if len(data) != len(imputed_data):
        raise ValueError('Imputed data must have the same amount of entries as the original.')
    
    merged = data.merge(imputed_data, how='left', on=['x', 'y'])
    merged['resist'] = merged['rhoa'] / merged['K']
    return_df = merged[['a', 'b', 'm', 'n', 'resist']].copy()

    if isinstance(rename_col, dict):
        return_df.rename(columns=rename_col, inplace=True)
    elif isinstance(rename_col, bool) and rename_col:
        return_df.rename(columns=default_cols, inplace=True)

    return return_df
