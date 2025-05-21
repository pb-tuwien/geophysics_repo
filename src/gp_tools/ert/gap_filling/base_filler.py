# -*- coding: utf-8 -*-
"""
Created on Wed May 21 11:16:13 2025

A basic class for gap filling of ERT pseudosections

@author: peter balogh @ TU Wien, Research Unit Geophysics
"""

#%% Imports

import matplotlib
import matplotlib.colors as mcolor
import matplotlib.pyplot as plt
from typing import Union
import shutil
import numpy as np
from pathlib import Path
from resipy import Project
from gp_tools.ert.utils import plot_pseudosection, extract_data, reconstruct_data
from gp_tools.core.file_types import OHMfile

#%% Base class for gap_filling

class GapFiller:
    def __init__(self, imputer: callable, base_dir: Union[str, Path] = Path.cwd()):
        self.basedir = Path(base_dir)
        self.imputer = imputer

        self.electrodes = None
        self.original_df = None
        self.filtered_df = None
        self.filled_df = None
        self.filled_data = None

        self.__original_data = None
        self.__merged_df = None
    
    @property
    def absrms(self):
        """
        The root mean square error.
        """
        if self.original_df is None or self.filled_df is None:
            print('No data found.')
            return None
        else:
            original_df = self.original_df.copy()
            imputed_df = self.filled_df.copy()
            summed_err = ((imputed_df['rhoa'].values - original_df['rhoa'].values)**2).sum()
            return np.sqrt(summed_err / len(original_df))

    @property
    def relrms(self):
        """
        The relative root mean square error.
        """
        if self.original_df is None or self.filled_df is None:
            print('No data found.')
            return None
        else:
            original_df = self.original_df.copy()
            imputed_df = self.filled_df.copy()
            summed_err = (((imputed_df['rhoa'].values - original_df['rhoa'].values) / original_df['rhoa'].values)**2).sum()
            return np.sqrt(summed_err / len(original_df))
                           
    def read_ohm(self, original: Union[str, Path], filtered: Union[str, Path], delete_temp: bool = True):
        path_orig = Path(original)
        path_orig = path_orig if path_orig.is_absolute() else self.basedir / path_orig
        path_filt = Path(filtered)
        path_filt = path_filt if path_filt.is_absolute() else self.basedir / path_filt

        proj_orig = Project(typ='R2', dirname=self.basedir / 'temp_dir')
        proj_orig.createSurvey(fname=path_orig, ftype='BERT')
        self.original_df, self.electrodes, self.__original_data = extract_data(proj_orig)

        proj_filt = Project(typ='R2', dirname=self.basedir / 'temp_dir')
        proj_filt.createSurvey(fname=path_filt, ftype='BERT')
        self.filtered_df, _, _ = extract_data(proj_filt)

        merged_df = self.original_df.merge(
            self.filtered_df,
            on=['x', 'y'],
            how="left",
            suffixes=("_drop", "")
        )
        merged_df.drop(columns='rhoa_drop', inplace=True)
        self.__merged_df = merged_df

        if delete_temp:
            shutil.rmtree(self.basedir / 'temp_dir', ignore_errors=True)

    def run(self, filepath: Union[str, Path], save: bool = True):
        filepath = Path(filepath)
        filepath = filepath if filepath.is_absolute() else self.basedir / filepath

        if self.__merged_df is not None:
            self.filled_df = self.imputer.fit_transform(data=self.__merged_df)
        else:
            raise ValueError('No data found. Must read it first with read_ohm().')
        
        reconstructed_data = reconstruct_data(self.__original_data, self.filled_df, rename_col=True)
        if save:
            _ = OHMfile.write(filepath=filepath, electrodes=self.electrodes, data=reconstructed_data)
        self.filled_data = reconstructed_data
    
    def plot_comparison(self, ax=None,
                        vmin: Union[int, float, None] = None, 
                        vmax: Union[int, float, None] = None,
                        cmap: str = 'viridis'):
        if self.filled_df is None:
            raise ValueError('Must run the gap filler first.')
        if ax is None:
            return_fig = True
            fig, ax = plt.subplots(1,3, figsize=(18,5))
            ax[0].set_title('Original')
            ax[1].set_title(f'Filtered')
            ax[2].set_title(f'Reconstructed')
        else:
            return_fig = False
            if len(ax) != 3:
                raise ValueError('Must provide an iterable with 3 axes.')
            fig = ax.get_figure()

        plot_pseudosection(self.original_df, ax=ax[0], vmin=vmin, vmax=vmax, cmap=cmap)
        plot_pseudosection(self.filtered_df, ax=ax[1], vmin=vmin, vmax=vmax, cmap=cmap)
        plot_pseudosection(self.filled_df, ax=ax[2], vmin=vmin, vmax=vmax, cmap=cmap)
        fig.tight_layout()

        if return_fig:
            return fig

    def plot_misfit(self, ax: Union[matplotlib.axes.Axes, None] = None,
                    relative_misfit: bool = False,
                    vmin: Union[int, float, None] = None, 
                    vmax: Union[int, float, None] = None,
                    cmap: str = 'seismic'):
        if self.filled_df is None:
            raise ValueError('Must run the gap filler first.')
        
        misfit_df = self.filled_df.copy()
        misfit_df['misfit'] = misfit_df['rhoa'] - self.original_df['rhoa']
        misfit_df['rel_misfit'] = (misfit_df['rhoa'] - self.original_df['rhoa']) / self.original_df['rhoa']
        
        if relative_misfit:
            title = 'Relative Misfit'
            label = '%'
            column = 'rel_misfit'
        else:
            title = 'Misfit'
            label = None
            column = 'misfit'

        if ax is None:
            return_fig = True
            fig, ax = plt.subplots()
            ax.set_title('Misfit')
        else:
            return_fig = False
            fig = ax.get_figure()

        offset = mcolor.TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax)
        plot_pseudosection(misfit_df, column=column, label=label, ax=ax, cmap=cmap, norm=offset)
        fig.tight_layout()

        if return_fig:
            return fig
