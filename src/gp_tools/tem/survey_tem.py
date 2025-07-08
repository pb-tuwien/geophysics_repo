# -*- coding: utf-8 -*-
"""
Created on Sun Nov 03 19:31:36 2024

A class for TEM surveys. Can do forward modelling, inversion and L-curve-analysis.

@author: peter balogh @ TU Wien, Research Unit Geophysics
"""

#%% Import modules

from pathlib import Path
import warnings
from tqdm import tqdm
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import pygimli.viewer.mpl
from scipy.interpolate import CubicSpline

from gp_tools.core import TEMfile, TIMfile, SurveyBase
from .inversion.empymod_inversion import tem_inv_smooth1D
from .forward.utils import plot_data, plot_model

warnings.filterwarnings('ignore')

#%% SurveyTEM class

class SurveyTEM(SurveyBase):
    def __init__(self, project_directory: Union[Path, str], 
                 dir_template: str = 'tem_default', 
                 save_log: bool = False, 
                 location: str = 'donauinsel') -> None:
        
        self._data_modelled = None #TODO
        self._mu = 4 * np.pi * 10 ** (-7)
        plt.close('all')
        plt.rcParams['figure.dpi'] = 300
        
        super().__init__(project_dir=project_directory, dir_structure=dir_template, save_log=save_log)

        self.__data_raw_path = None
        self.__color_dict = None

        self.data_raw = None
        self.data_preprocessed = None
        self.data_filtered = None
        self.data_inverted = None

        self.__ramp_location = location

    @property
    def data_raw(self) -> Optional[dict]:
        """
        Returns the raw data.

        Returns
        -------
        dict
            Raw data.
        """
        return self.__data_raw
    
    @data_raw.setter
    def data_raw(self, data: Optional[dict]):
        """
        Setter of data_raw property.
        """
        if data is None:
            self.__data_raw = None
        else:
            self._check_data(data_dictionary=data)
            self.__data_raw = data

    @property
    def data_preprocessed(self) -> Optional[dict]:
        """
        Returns the preprocessed data.

        Returns
        -------
        dict
            Preprocessed data.
        """
        return self.__data_preprocessed
    
    @data_preprocessed.setter
    def data_preprocessed(self, data: Optional[dict]):
        """
        Setter of data_preprocessed property.
        """
        if data is None:
            self.__data_preprocessed = None
        else:
            self._check_data(data_dictionary=data)
            self.__data_preprocessed = data

    @property
    def data_filtered(self) -> dict:
        """
        Returns the filtered data.

        Returns
        -------
        dict
            Filtered data.
        """
        return self.__data_filtered
    
    @data_filtered.setter
    def data_filtered(self, data: Optional[dict]):
        """
        Setter of data_filtered property.
        """
        self.__data_filtered = data

    @property
    def data_inverted(self) -> dict:
        """
        Returns the inverted data.

        Returns
        -------
        dict
            Inverted data.
        """
        return self.__data_inverted

    @data_inverted.setter
    def data_inverted(self, data: Optional[dict]):
        """
        Setter of data_inverted property.
        """
        self.__data_inverted = data
    
    @property
    def color_dict(self):
        """
        Dictionary mapping all soundings to a color.
        """
        return self.__color_dict

    def _check_data(self, data_dictionary: dict) -> None:
        """
        This function checks if the data dictionary contains the necessary keys and columns.
        The necessary keys are:
        - 'current', 'tloop', 'turn', 'timerange', 'filter', 'rloop', 'x', 'y', 'z'
        The necessary columns are:
        - 'Time', 'E/I[V/A]', 'Err[V/A]'

        Parameters
        ----------
        data_dictionary : dict
            Dictionary containing the data and metadata.

        Returns
        -------
        None.

        Raises
        ------
        KeyError
            If the data dictionary does not contain the necessary keys or columns.
        """
        metadata_key = ['current', 'tloop', 'turn', 'timerange', 'filter', 'rloop', 'x', 'y', 'z']
        data_columns = ['Time', 'E/I[V/A]', 'Err[V/A]']

        for value in tqdm(data_dictionary.values(), desc='Checking data', unit='sounding'):
            if value.get('data') is None:
                self.logger.error('Data not found in dictionary.')
                raise KeyError('Data not found in dictionary.')
            elif value.get('metadata') is None:
                self.logger.error('Metadata not found in dictionary.')
                raise KeyError('Metadata not found in dictionary.')
            elif not self._is_sublist(mainlist=list(value.get('metadata').keys()), sublist=metadata_key):
                self.logger.error('Metadata keys are missing.')
                raise KeyError('Metadata keys are missing.')
            elif not self._is_sublist(mainlist=list(value.get('data').columns), sublist=data_columns):
                self.logger.error('Data columns are missing.')
                raise KeyError('Data columns are missing.')

        self.logger.info('Data check successful.')

    def data_read(self, data: Union[Path, str] = None) -> None:
        """
        This function reads the raw data from the given file path.
        If no file path is given, it tries to read the raw data from the directory structure.

        Parameters
        ----------
        data : Path or str, optional
            File path to the raw data file. The default is None.

        Returns
        -------
        None.
        """
        data_raw = self.folder_structure.get('data_raw')
        
        if data is None:
            if data_raw.exists():
                raw_paths = [path for path in data_raw.iterdir() if path.is_file()]
                if len(raw_paths) > 1:
                    self.logger.warning('Multiple raw data files found. Using the first one.')
                if len(raw_paths) == 0:
                    self.logger.error('No raw data files found.')
                else:
                    data_path = raw_paths[0]
                    self.data_raw = TEMfile.read(data_path).data
                    self.__data_raw_path = data_path
        else:
            data = Path(data)
            if data.exists():
                new_data = data_raw / data.name
                self._gp_folder.move_files(from_path=data, to_path=data_raw)
                self.data_raw = TEMfile.read(new_data).data
                self.__data_raw_path = data
            else:
                self.logger.warning('Data file not found. Tries to read the data from the directory structure.')
                if data is not None:
                    self.data_read(data=None)

    def data_rename_columns(self, rename_dict: dict) -> Optional[dict]:
        """
        This function renames the columns of the data dictionary.

        Parameters
        ----------
        rename_dict : dict
            Dictionary containing the old and new column names.

        Returns
        -------
        dict
            Dictionary containing the data and metadata with the renamed columns.
        """
        new_data_dict = self.data_raw.copy()

        if new_data_dict is not None:
            for key, value in tqdm(new_data_dict.items(), desc='Renaming columns', unit='sounding'):
                df_data = value.pop('data')
                df_data.rename(columns=rename_dict, inplace=True)
                new_data_dict[key]['data'] = df_data
            return new_data_dict
        else:
            self.logger.error('data_rename_columns: No data found.')
            raise KeyError('data_rename_columns: No data found.')


    def _add_coords_to_data(self, data_dict: dict, parsing_dict: dict = None) -> Optional[dict]:
        """
        This function adds the coordinates to the metadata of the data dictionary.

        Parameters
        ----------
        data_dict : dict
            Dictionary containing the data and metadata.
        parsing_dict : dict, optional
            Dictionary to parse the keys of the data dictionary. The default is None.
            This parameter can be used if the keys of the data dictionary are not the same as the coordinates keys.
            Or if multiple measurements should have the same coordinates. Example:
            {'T001': 'Mtest', 'T002': 'Mtest'} -> T001 and T002 will both be assigned the coordinates of Mtest.

        Returns
        -------
        dict
            Dictionary containing the data and metadata with the coordinates added.
        """
        if self._coordinates_grouped is None:
            self.logger.warning('_add_coords_to_data: No coordinates found. Continued without.')
            return data_dict

        new_data_dict = data_dict.copy()

        if new_data_dict is not None:
            for key, value in tqdm(new_data_dict.items(), desc='Adding coordinates to data', unit='sounding'):
                if parsing_dict is not None:
                    coords_key = parsing_dict.get(key, key)
                else:
                    coords_key = key

                value['metadata']['x'] = self._coordinates_grouped.get(coords_key, {}).get('x', 0)
                value['metadata']['y'] = self._coordinates_grouped.get(coords_key, {}).get('y', 0)
                value['metadata']['z'] = self._coordinates_grouped.get(coords_key, {}).get('z', 0)
            return new_data_dict
        else:
            self.logger.error('_add_coords_to_data: No data found.')
            raise KeyError('_add_coords_to_data: No data found.')

    def _normalize_rhoa(self, data_dict: dict) -> Optional[dict]:
        """
        This function normalizes the data dictionary and calculates the apparent resistivity and conductivity.

        Parameters
        ----------
        data_dict : dict
            Dictionary containing the data and metadata.

        Returns
        -------
        dict
            Dictionary containing the normalized data and metadata.
        """
        norm_data_dict = data_dict.copy()

        if norm_data_dict is not None:
            for key, value in tqdm(norm_data_dict.items(), desc='Normalizing data', unit='sounding'):
                df_data = value.get('data')

                if 'rhoa' in df_data.columns:
                    norm_data_dict[key]['data'] = df_data
                else:
                    df_metadata = value.get('metadata')
                    current = df_metadata.get('current')
                    tloop = df_metadata.get('tloop')
                    turn = df_metadata.get('turn')

                    df_data['E/I[V/A]'] = current * df_data['E/I[V/A]'] / (tloop ** 2 * turn)
                    df_data['Err[V/A]'] = current * df_data['Err[V/A]'] / (tloop ** 2 * turn)
                    df_data['Time'] = df_data['Time'] / 1e6
                    mag_momen = turn * current * tloop ** 2
                    df_data['rhoa'] = 1 / np.pi * (mag_momen / (20 * np.abs(df_data['E/I[V/A]']))) ** (2 / 3) * (self._mu / df_data['Time']) ** (5 / 3)
                    df_data.loc[df_data['E/I[V/A]'] < 0, 'rhoa'] *= -1
                    df_data['sigma'] = 1 / (df_data['rhoa'])
                    norm_data_dict[key]['data'] = df_data

            return norm_data_dict
        else:
            self.logger.error('_normalize_rhoa: No data found.')
            raise KeyError('_normalize_rhoa: No data found.')

    def data_preprocess(self, data_dict: Optional[dict] = None, parsing_dict: dict = None) -> None:
        """
        This function preprocesses the data dictionary. If no dictionary is given, it uses the raw data.
        It adds the coordinates to the metadata, normalizes the data, and calculates the apparent resistivity and conductivity.
        It also saves the preprocessed data if not done already.

        Parameters
        ----------
        data_dict : dict, optional
            Dictionary containing the data and metadata. The default is None.
        parsing_dict : dict, optional
            Dictionary to parse the keys of the data dictionary. The default is None.
            This parameter can be used if the keys of the data dictionary are not the same as the coordinates keys.
            Or if multiple measurements should have the same coordinates. Example:
            {'T001': 'Mtest', 'T002': 'Mtest'} -> T001 and T002 will both be assigned the coordinates of Mtest.

        Returns
        -------
        None.
        """
        preproc_path = self.folder_structure.get('data_preproc')

        if data_dict is None:
            data_dict = self.data_raw
            if self.__data_raw_path is not None:
                data_path = preproc_path / f'{self.__data_raw_path.stem}_proc{self.__data_raw_path.suffix}'
            else:
                self.logger.error('data_preprocess: No raw data path found.')
                return
        else:
            data_path = preproc_path / f'preprocessed_data.tem'

        if data_path.exists():
            self.logger.info('data_preprocess: Preprocessed data already exists. Reading file.')
            self.data_preprocessed = TEMfile.read(filepath=data_path).data
        else:
            self.logger.info('data_preprocess: Preprocessing data.')
            self._check_data(data_dictionary=data_dict)
            data_dict = self._add_coords_to_data(data_dict=data_dict, parsing_dict=parsing_dict)
            data_dict = self._normalize_rhoa(data_dict=data_dict)

            self.data_preprocessed = data_dict
            TEMfile.write(data=data_dict, filepath=data_path)

    def _filter_sounding(self, data_dict: dict,
                         filter_times: Tuple[Union[float, int], Union[float, int]],
                         noise_floor: Union[float, int]) -> dict:
        """
        This function filters one sounding based on the given time range and noise floor.

        Parameters
        ----------
        data_dict : dict
            Dictionary containing the data and metadata.
        filter_times : Tuple[Union[float, int], Union[float, int]]
            Tuple containing the start and end time for the filter.
        noise_floor : [float, int]
            Noise floor for the relative error.

        Returns
        -------
        dict
            Dictionary containing the filtered data and metadata.
        """
        filtered_dict = data_dict.copy()
        df = data_dict.get('data')
        if df is None:
            self.logger.error('No data found.')
            raise KeyError('No data found.')

        mask_time = (df['Time'] > filter_times[0] / 1e6) & (df['Time'] < filter_times[1] / 1e6)

        df_f0 = df[mask_time]
        df_f0['rel_err'] = abs(df_f0['Err[V/A]'].values) / df_f0['E/I[V/A]'].values  # observed relative error

        df_f0.loc[df_f0['rel_err'] < noise_floor, 'rel_err'] = noise_floor

        # Find the index of the first negative value
        negative_indices = df_f0[df_f0['E/I[V/A]'] < 0].index
        while not negative_indices.empty:

            first_negative_index = int(negative_indices.min())

            if not pd.isnull(first_negative_index):
                # Find the index again to ensure its validity after potential DataFrame modifications
                negative_indices = df_f0[df_f0['E/I[V/A]'] < 0].index
                first_negative_index = negative_indices.min()
                # Determine if the negative value is in the first half or second half
                if first_negative_index < len(df_f0) / 2:
                    # Delete all rows before the first negative value
                    df_f0 = df_f0[df_f0.index >= first_negative_index + 1]
                    negative_indices = df_f0[df_f0['E/I[V/A]'] < 0].index
                elif first_negative_index > len(df_f0) / 2:
                    # Delete all rows after the first negative value
                    df_f0 = df_f0[df_f0.index < first_negative_index]
                    negative_indices = df_f0[df_f0['E/I[V/A]'] < 0].index

        df_f1 = df_f0.copy()

        # Find first lower index
        diff = df_f1['E/I[V/A]'] - df_f1['Err[V/A]']
        lower_indices = diff[diff <= 0].index

        # Truncate the DataFrame if such a condition exists
        if not lower_indices.empty:

            first_lower_index = int(lower_indices.min())
            if not pd.isnull(first_lower_index):
                df_f1 = df_f1.loc[:first_lower_index-1]

        df_f2 = df_f1.copy()

        df_f2.reset_index(drop=True, inplace=True)
        df_f2['rel_err'] = abs(df_f2['Err[V/A]'].values) / df_f2['E/I[V/A]'].values  # observed relative error

        df_f2.loc[df_f2['rel_err'] < noise_floor, 'rel_err'] = noise_floor

        filtered_dict['data'] = df_f2
        filtered_dict['metadata']['name'] = f'{filter_times[0]}_{filter_times[1]}_{noise_floor}'
        return filtered_dict

    def data_filter(self, filter_times: Tuple[Union[float, int], Union[float, int]] = (7, 700),
                    noise_floor: Union[float, int] = 0.025,
                    subset: list = None) -> None:
        """
        This function filters the data dictionary based on the given time range and noise floor.
        If no subset is given, it filters all soundings.

        Parameters
        ----------
        filter_times : Tuple[Union[float, int], Union[float, int]], optional
            Tuple containing the start and end time for the filter. The default is (7, 700).
        noise_floor : [float, int], optional
            Noise floor for the relative error. The default is 0.025.
        subset : list, optional
            List of keys to filter. The default is None.

        Returns
        -------
        None.
        """
        if subset is None:
            subset = list(self.data_preprocessed.keys())
        else:
            subset = [key for key in subset if key in self.data_preprocessed.keys()]
            invalid_subset = [key for key in subset if key not in self.data_preprocessed.keys()]
            if invalid_subset:
                self.logger.warning(f'Invalid subset keys: {invalid_subset}')

        filter_dir_path = self.folder_structure.get('data_filtered')
        filter_key = f'{filter_times[0]}_{filter_times[1]}_{noise_floor}'
        if self.data_filtered is None:
            self.data_filtered = {}

        for key in subset:
            file_path_filtered = filter_dir_path / f'{key}.tem'

            if file_path_filtered.exists():
                data_filtered = TEMfile.read(filepath=file_path_filtered).data

                if data_filtered.get(filter_key) is None:
                    data = self.data_preprocessed.get(key)
                    data_filtered[filter_key] = self._filter_sounding(data_dict=data, filter_times=filter_times,
                                                                      noise_floor=noise_floor)
                    TEMfile.write(data=data_filtered, filepath=file_path_filtered)

            else:
                data_filtered = {}
                data = self.data_preprocessed.get(key)
                data_filtered[filter_key] = self._filter_sounding(data_dict=data, filter_times=filter_times,
                                                                  noise_floor=noise_floor)
                TEMfile.write(data=data_filtered, filepath=file_path_filtered)

            if self.data_filtered.get(key) is None:
                self.data_filtered[key] = {}
            self.data_filtered[key][filter_key] = data_filtered[filter_key]

    def _inversion_sounding(self, data_dict: dict,
                            depth_vector: np.ndarray,
                            inversion_key: str,
                            start_model: np.ndarray = None,
                            verbose: bool = True) -> dict:
        filtered_data = data_dict.get('data')
        filtered_rhoa = filtered_data['rhoa'].values
        filtered_signal = filtered_data['E/I[V/A]'].values
        filtered_relerr = filtered_data['rel_err'].values

        filtered_time = filtered_data['Time'].values

        filtered_metadata = data_dict.get('metadata')
        tloop = filtered_metadata.get('tloop')
        rloop = filtered_metadata.get('rloop')
        current = filtered_metadata.get('current')
        turn = filtered_metadata.get('turn')
        timerange = filtered_metadata.get('timerange')
        filter_pl = filtered_metadata.get('filter')

        split_inversion_key = [float(i) for i in inversion_key.split('_')]
        if len(split_inversion_key) == 3:
            lam, filter_min, filter_max = split_inversion_key
        elif len(split_inversion_key) == 4:
            lam, filter_min, filter_max, _ = split_inversion_key
        else:
            raise ValueError('Invalid inversion key.')

        if start_model is None:
            rhoa_median = np.round(np.median(filtered_rhoa), 4)
            start_model = np.full_like(depth_vector, rhoa_median)

        setup_device = {
            "timekey": timerange,
            "currentkey": np.round(current),
            "txloop": tloop,
            "rxloop": rloop,
            "current_inj": current,
            "filter_powerline": filter_pl,
            "ramp_data": self.__ramp_location
        }

        # setup inv class and calculate response of homogeneous model
        tem_inv = tem_inv_smooth1D(setup_device=setup_device)

        self.test_resp = tem_inv.prepare_fwd(depth_vector=depth_vector,
                                start_model=start_model,
                                times_rx=filtered_time)
        tem_inv.prepare_inv(maxIter=20, verbose=True)  # prepare the inversion, keep the kwargs like this

        # start inversion
        res_mdld = tem_inv.run(dataVals=filtered_signal, errorVals=filtered_relerr,
                               startModel=start_model, lam=lam)

        resp_sgnl = tem_inv.response
        thks = np.diff(tem_inv.depth_fixed)  # convert depths to layer thicknesses
        chi2 = tem_inv.chi2()
        rrms = tem_inv.relrms()
        absrms = tem_inv.absrms()
        phi_model = tem_inv.phiModel()
        mag_momen = turn * current * tloop ** 2
        response_rhoa = 1 / np.pi * (mag_momen / (20 * resp_sgnl)) ** (2 / 3) * (self._mu / filtered_time) ** (
                5 / 3)

        inversion_df = pd.DataFrame()
        max_length = max(len(res_mdld), len(resp_sgnl), len(thks))
        inversion_df = inversion_df.reindex(range(max_length))

        inversion_df['depth_vector'] = pd.Series(depth_vector)
        inversion_df['start_model'] = pd.Series(start_model)
        inversion_df['resistivity_model'] = pd.Series(res_mdld)
        inversion_df['conductivity_model'] = pd.Series(1 / res_mdld)
        inversion_df['E/I[V/A]'] = pd.Series(resp_sgnl)
        inversion_df['modelled_thickness'] = pd.Series(thks)
        inversion_df['rhoa'] = pd.Series(response_rhoa)
        inversion_df['sigma'] = pd.Series(1 / response_rhoa)

        inversion_metadata = {
            'lambda': lam,
            'filtertime_min': filter_min,
            'filtertime_max': filter_max,
            'chi2': chi2,
            'relrms': rrms,
            'absrms': absrms,
            'phi_model': phi_model
        }

        return {'data': inversion_df, 'metadata': inversion_metadata}

    def data_inversion(self, lam: Union[int, float, list] = 600,
                       layer_type: str = 'linear',
                       layers: Union[int, float, dict, np.ndarray] = 4.5,
                       max_depth: Union[float, int] = None,
                       filter_times: Tuple[Union[float, int], Union[float, int]] = (7, 700),
                       start_model: np.ndarray = None,
                       noise_floor: Union[float, int] = 0.025,
                       subset: list = None,
                       verbose: bool = True) -> None:
        if self.data_preprocessed is None:
            self.logger.error('data_inversion: No preprocessed data found.')
            return
        if subset is None:
            subset = list(self.data_preprocessed.keys())
        else:
            subset = [key for key in subset if key in self.data_preprocessed.keys()]
            invalid_subset = [key for key in subset if key not in self.data_preprocessed.keys()]
            if invalid_subset:
                self.logger.warning(f'Invalid subset keys: {invalid_subset}')
        
        if isinstance(lam, list):
            lambda_list = lam
        elif isinstance(lam, (float, int)):
            lambda_list = [lam]

        self.data_filter(filter_times=filter_times, noise_floor=noise_floor, subset=subset)

        inversion_dir_path = self.folder_structure.get('data_inversion')
        filter_key = f'{filter_times[0]}_{filter_times[1]}_{noise_floor}'

        if self.data_inverted is None:
            self.data_inverted = {}

        if layer_type == 'linear':
            if verbose:
                self.logger.info(f'inversion: Inversion with linear layer thickness. Layer thickness: {layers}.')
            depth_vector = np.arange(0, max_depth, step=layers)

        elif layer_type == 'log':
            if verbose:
                self.logger.info(f'inversion: Inversion with logarithmic layer thickness. Number of layers: {layers}.')
            depth_vector = np.logspace(-1, np.log10(max_depth + 0.1), round(layers)) - 0.1

        elif layer_type == 'dict':
            if not isinstance(layers, dict):
                if verbose:
                    self.logger.error('inversion: layers must be a dictionary.')
                raise TypeError('Layers must be a dictionary.')
            if verbose:
                self.logger.info(f'inversion: Inversion with layer thicknesses extracted from the layers dict.')

            if all(key < max_depth for key in layers.keys()):
                lay_keys = sorted(list(layers.keys()))
            else:
                lay_keys = sorted([key for key in layers.keys() if key < max_depth])
            lay_keys.append(max_depth)

            layer_list = [lay_keys[0]]
            cur_depth = lay_keys[0]
            for i in range(len(lay_keys) - 1):
                while cur_depth <= lay_keys[i + 1]:
                    cur_depth += layers[lay_keys[i]]
                    if cur_depth <= max_depth:
                        layer_list.append(cur_depth)
            depth_vector = np.array(layer_list)

        elif layer_type == 'custom':
            if not isinstance(layers, np.ndarray):
                if verbose:
                    self.logger.error('inversion: layers must be an numpy array.')
                raise TypeError('Layers must be an numpy array.')
            if verbose:
                self.logger.info('inversion: Inversion with custom layer thicknesses. layers was read as the depth vector.')
            depth_vector = layers

        else:
            if verbose:
                self.logger.error(f'inversion: {layer_type} is an unknown keyword.')
            raise KeyError(f'{layer_type} is an unknown keyword.')

        for key in subset:
            file_path_inversion = inversion_dir_path / f'{key}.tim'
            data_filtered = self.data_filtered.get(key).get(filter_key)
            filtered_df = data_filtered.get('data')
            filtered_metadata = data_filtered.get('metadata')
            tloop = filtered_metadata.get('tloop')
            turn = filtered_metadata.get('turn')
            current = filtered_metadata.get('current')

            if max_depth is None:
                max_depth = np.round(np.sqrt(tloop ** 2 * turn * current), 2)

            if start_model is None:
                rhoa_median = np.round(np.median(filtered_df['rhoa']), 4)
                start_model = np.full_like(depth_vector, rhoa_median)

            data_inversion = {}
            if file_path_inversion.exists():
                data_inversion_raw = TIMfile.read(filepath=file_path_inversion).data
                for lambda_param in lambda_list:
                    old_inv_key = f'{lambda_param}_{filter_times[0]}_{filter_times[1]}'
                    inversion_key = f'{lambda_param}_{filter_times[0]}_{filter_times[1]}_{noise_floor}'

                    key_list = [key for key in data_inversion_raw.keys() if key.startswith(inversion_key)]

                    if not key_list:
                        key_list = [key for key in data_inversion_raw.keys() if key.startswith(old_inv_key) and not key.replace(old_inv_key, '').startswith('_0.')]
                        if key_list and verbose:
                            self.logger.warning('inversion: Using old format. Might not consider changing noise_floor inputs.')

                    data_missing = True
                    if key_list:
                        inversion_name = f'{inversion_key}_{len(key_list) + 1}'
                        for inv_key in key_list:
                            found_data = data_inversion_raw.get(inv_key)
                            found_df = found_data.get('data')
                            found_start_model = found_df['start_model'].dropna().values
                            found_depth_vector = found_df['depth_vector'].dropna().values

                            if depth_vector.size != found_depth_vector.size:
                                depth_same = False
                            else:
                                depth_same = np.allclose(depth_vector, found_depth_vector, atol=1e-5)

                            if start_model.size != found_start_model.size:
                                start_same = False
                            else:
                                start_same = np.allclose(start_model, found_start_model, atol=1e-5)

                            if start_same and depth_same:
                                if verbose:
                                    self.logger.info(f'inversion: Inversion already exists. Added {inversion_key} to {key}.')
                                data_inversion[inversion_key] = found_data
                                data_missing = False
                                break
                    else:
                        inversion_name = f'{inversion_key}'

                    if data_missing:
                        inversion_result = self._inversion_sounding(data_dict=data_filtered,
                                                                   depth_vector=depth_vector,
                                                                   inversion_key=inversion_key,
                                                                   start_model=start_model,
                                                                   verbose=verbose)
                        
                        inversion_result['metadata']['name'] = inversion_name
                        inversion_result['metadata']['max_depth'] = max_depth

                        data_inversion[inversion_key] = inversion_result
                        data_inversion_raw[inversion_name] = data_inversion[inversion_key]

                TIMfile.write(data=data_inversion_raw, filepath=file_path_inversion)        

            else:
                for lambda_param in lambda_list:
                    inversion_key = f'{lambda_param}_{filter_times[0]}_{filter_times[1]}_{noise_floor}'
                    data_inversion[inversion_key] = self._inversion_sounding(data_dict=data_filtered,
                                                                   depth_vector=depth_vector,
                                                                   inversion_key=inversion_key,
                                                                   start_model=start_model,
                                                                   verbose=verbose)
                    data_inversion[inversion_key]['metadata']['name'] = inversion_key
                    data_inversion[inversion_key]['metadata']['max_depth'] = max_depth

                TIMfile.write(data=data_inversion, filepath=file_path_inversion)
            
            if self.data_inverted.get(key) is None:
                self.data_inverted[key] = {}
            self.data_inverted[key].update(data_inversion)
    
    def _get_color(self, sounding: str, cmap = None):
        if self.color_dict is None:
            all_soundings = [key for key in self.data_preprocessed.keys()]

            if cmap is None:
                cmap = plt.get_cmap('viridis')
            elif isinstance(cmap, str):
                cmap = plt.get_cmap(cmap)
            
            color_dict = {key: cmap(col) for key, col in zip(all_soundings, np.linspace(0, 1, len(all_soundings)))}
            if color_dict:
                self.__color_dict = color_dict
            else:
                self.logger.error('Could not map the colors.')
                return None
            self.logger.info('Mapped all soundings to a color.')
        
        return self.color_dict.get(sounding)

    def plot_raw_filtered(self,
                          subset: list = None, 
                          unit: str = 'rhoa', 
                          scale: str = 'lin',
                          filter_times: Tuple[Union[int, float], Union[int, float]] = (7, 700),
                          noise_floor: Union[int, float] = 0.025, 
                          legend: bool = True,
                          limits_time: Optional[tuple] = None,
                          limits_signal: Optional[tuple] = None,
                          limits_rhoa: Optional[tuple] = None,
                          colormap = None,
                          fname: Union[str, bool] = None):
        plot_list = [point for point in self.data_preprocessed.keys() if subset is None or point in subset]
        self.data_filter(subset=subset, filter_times=filter_times, noise_floor=noise_floor)

        if scale == 'log':
            log_rhoa = True
        elif scale == 'lin':
            log_rhoa = False
        else:
            raise SyntaxError('input {} not valid for Argument scale'.format(scale))
        
        if unit == 'rhoa':
            rhoa_title = 'Apparent Resistivity'
            res2con = False
        elif unit == 'sigma':
            rhoa_title = 'Apparent Conductivity'
            res2con = True
        else:
            raise SyntaxError('input {} not valid for Argument unit'.format(unit))

        fig, ax = plt.subplots(2, 2, figsize=(13, 13))
        ax = ax.ravel()


        for key in plot_list:
            raw_data = self.data_preprocessed[key].get('data')
            filtered_data = self.data_filtered[key][f'{filter_times[0]}_{filter_times[1]}_{noise_floor}'].get('data')
            col = self._get_color(key, cmap=colormap)

            plot_data(
                time=raw_data['Time'], ax=ax[:2],
                signal=raw_data['E/I[V/A]'], rhoa=raw_data['rhoa'],
                signal_title='Raw Impulse Response', rhoa_title='Raw {}'.format(rhoa_title),
                noise_signal=raw_data['Err[V/A]'], log_rhoa=log_rhoa,
                res2con=res2con, legend=False, color=col,
                sub0_color_signal=col, sub0_color_rhoa=col,
                xlimits=limits_time, ylimits_signal=limits_signal,
                ylimits_rhoa=limits_rhoa, fontweight='bold'
            )

            plot_data(
                time=raw_data['Time'], ax=ax[2:],
                signal=raw_data['E/I[V/A]'], rhoa=raw_data['rhoa'],
                noise_signal=raw_data['Err[V/A]'], log_rhoa=log_rhoa,
                res2con=res2con, legend=False, color='#808080', alpha=0.4,
                sub0_color_signal='#808080', sub0_color_rhoa='#808080',
                xlimits=limits_time, ylimits_signal=limits_signal,
                ylimits_rhoa=limits_rhoa
            )

            plot_data(
                time=filtered_data['Time'], ax=ax[2:],
                signal=filtered_data['E/I[V/A]'], rhoa=filtered_data['rhoa'],
                signal_title='Filtered Impulse Response', rhoa_title='Filtered {}'.format(rhoa_title),
                signal_label=key, fontweight='bold',
                sub0_color_signal=col, sub0_color_rhoa=col,
                log_rhoa=log_rhoa, color=col,
                res2con=res2con, legend=legend,
                xlimits=limits_time, ylimits_signal=limits_signal,
                ylimits_rhoa=limits_rhoa
            )

        label_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
        for axis, label in zip(ax, label_list):
            axis.text(0.96, 0.96, f'({label})', transform=axis.transAxes, fontsize=18, zorder=5,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(facecolor='xkcd:light grey', boxstyle='round,pad=0.4', alpha=0.3))

        fig.tight_layout()
        fig.show()
        target_dir = self.folder_structure.get('data_first_look')
        time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if fname or fname is None:
            file_name = f'raw_filtered_{time}_{filter_times[0]}_{filter_times[1]}_{unit}.png' if fname is None else fname
            fig.savefig(target_dir / file_name)






    def _plot_one_inversion(self, sounding,
                           lam: Union[int, float] = 600,
                           filter_times: Tuple[Union[float, int], Union[float, int]] = (7, 700),
                           noise_floor: Union[int, float] = 0.025,
                           unit: str = 'rhoa',
                           scale: str = 'lin',
                           limits_depth: Optional[tuple] = None,
                           limits_rho: Optional[tuple] = None,
                           limits_time: Optional[tuple] = None,
                           limits_signal: Optional[tuple] = None,
                           limits_rhoa: Optional[tuple] = None,
                           colormap = None,
                           fname: Union[str, bool] = None) -> None:

        inv_name = f'{lam}_{filter_times[0]}_{filter_times[1]}_{noise_floor}'
        filter_name = f'{filter_times[0]}_{filter_times[1]}_{noise_floor}'
        inverted_data = self.data_inverted.get(sounding, {}).get(inv_name)
        filtered_data = self.data_filtered.get(sounding, {}).get(filter_name)

        if inverted_data is None:
            self.logger.error(f'No inversion data found for {sounding}.')
            return

        if filtered_data is None:
            self.logger.error(f'No filtered data found for {sounding}.')
            return

        filtered_data = filtered_data.get('data')
        inversion_data = inverted_data.get('data')
        inversion_metadata = inverted_data.get('metadata')

        thks = np.r_[inversion_data['modelled_thickness'].dropna(), 0]
        model_res = inversion_data['resistivity_model'].dropna()
        chi2 = inversion_metadata.get('chi2')
        rrms = inversion_metadata.get('relrms')

        if scale == 'log':
            log_rhoa = True
        elif scale == 'lin':
            log_rhoa = False
        else:
            raise SyntaxError('input {} not valid for Argument scale'.format(scale))
        
        if unit == 'rhoa':
            rhoa_title = 'Apparent Resistivity'
            model_title = 'Resistivity'
            res2con = False
        elif unit == 'sigma':
            rhoa_title = 'Apparent Conductivity'
            model_title = 'Conductivity'
            res2con = True
        else:
            raise SyntaxError('input {} not valid for Argument unit'.format(unit))
        
        fig = plt.figure(figsize=(14, 7) ,constrained_layout=True)

        ax_model = [fig.add_subplot(1,5,1)]
        ax_response = [fig.add_subplot(1,5,(2,3)), fig.add_subplot(1,5,(4,5))]

        col = self._get_color(sounding=sounding, cmap=colormap)

        plot_model(
            model=np.column_stack((thks, model_res)), ax=ax_model[0],
            res2con=res2con, xlimits=limits_rho, ylimits=limits_depth,
            title=f'Model of {model_title} at Depth', color=col, add_bottom=2
        )
        
        plot_data(
            time=filtered_data['Time'], ax=ax_response,
            signal=filtered_data['E/I[V/A]'], rhoa=filtered_data['rhoa'],
            signal_title='Impulse Response', rhoa_title=f'{rhoa_title}',
            signal_label='observed', rhoa_label='observed',
            sub0_color_signal=col, sub0_color_rhoa=col,
            log_rhoa=log_rhoa, color=col, noise_signal=filtered_data['Err[V/A]'],
            res2con=res2con, legend=True, marker='v',
            xlimits=limits_time, ylimits_signal=limits_signal,
            ylimits_rhoa=limits_rhoa
        )

        plot_data(
            time=filtered_data['Time'], ax=ax_response,
            signal=inversion_data['E/I[V/A]'].dropna(), rhoa=inversion_data['rhoa'].dropna(),
            signal_title='Impulse Response', rhoa_title=f'{rhoa_title}',
            signal_label='inversion', rhoa_label='inversion',
            log_rhoa=log_rhoa, color='k',
            res2con=res2con, legend=True,
            xlimits=limits_time, ylimits_signal=limits_signal,
            ylimits_rhoa=limits_rhoa
            )

        label_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
        for ax, label in zip(ax_model + ax_response, label_list):
            ax.text(0.96, 0.96, f'({label})', transform=ax.transAxes, fontsize=18, zorder=5,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(facecolor='xkcd:light grey', boxstyle='round,pad=0.4', alpha=0.3))

        fig.suptitle(f'$\lambda$ = {lam:<8.0f} Sounding = {sounding}\n$\chi^2$ = {chi2:<8.2f} Relative RMS = {rrms:<.2f}%', fontsize=22, fontweight='bold')
        fig.show()

        target_dir = self.folder_structure.get('data_inversion_plot')
        time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if fname or fname is None:
            file_name = f'{sounding}_{time}_{unit}.png' if fname is None else fname
            fig.savefig(target_dir / file_name)

    def plot_inversion(self, subset:list=None,
                       lam: Union[int, float] = 600,
                       layer_type: str = 'linear',
                       layers: Union[int, float, dict, np.ndarray] = 4.5,
                       max_depth: Union[float, int] = None,
                       start_model: np.ndarray = None,
                       noise_floor: Union[float, int] = 0.025,
                       unit: str = 'rhoa',
                       filter_times=(7, 700),
                       verbose: bool = True,
                       fname: Union[str, bool] = None) -> None:

        plot_list = [point for point in self.data_preprocessed.keys() if subset is None or point in subset]

        self.data_inversion(subset=plot_list, lam=lam, layer_type=layer_type, layers=layers,
                            max_depth=max_depth, filter_times=filter_times,
                            start_model=start_model, noise_floor=noise_floor,
                            verbose=verbose)

        for key in plot_list:
            self._plot_one_inversion(sounding=key,
                                     lam=lam,
                                     filter_times=filter_times,
                                     noise_floor=noise_floor,
                                     unit=unit,
                                     fname=fname)

    @staticmethod
    def _menger_curvature(p1, p2, p3):
        if any([len(p) != 2 for p in [p1, p2, p3]]):
            raise ValueError('Only implemented for 2D Points')
        p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
        matrix = np.array([p2 - p1, p3 - p2])

        dist_12 = np.linalg.norm(p2 - p1)
        dist_23 = np.linalg.norm(p3 - p2)
        dist_31 = np.linalg.norm(p1 - p3)

        det = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]

        curvature = (2 * np.abs(det)) / (dist_12 * dist_23 * dist_31) ** 0.5
        return curvature
    
    def _normalize_value(self, 
                         data: Union[int, float, np.ndarray], 
                         min_val: Union[int, float, np.ndarray], 
                         max_val: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
        return (data - min_val) / (max_val - min_val)

    def _l_curve_point(self, 
                       sounding: str, 
                       lam: Union[int, float], 
                       filter_times: tuple, 
                       layer_type: str, 
                       layers, 
                       max_depth: Union[int, float], 
                       noise_floor: float):

        if self.data_inverted is not None:
            inversion_dict = self.data_inverted.get(sounding, {}).get(f'{lam}_{filter_times[0]}_{filter_times[1]}_{noise_floor}')
            if inversion_dict is None:
                self.data_inversion(subset=[sounding], lam=lam, layer_type=layer_type, layers=layers,
                                    verbose=False, max_depth=max_depth, filter_times=filter_times,
                                    noise_floor=noise_floor, start_model=None)
                inversion_dict = self.data_inverted.get(sounding, {}).get(f'{lam}_{filter_times[0]}_{filter_times[1]}_{noise_floor}')
        else:
            self.data_inversion(subset=[sounding], lam=lam, layer_type=layer_type, layers=layers,
                                verbose=False, max_depth=max_depth, filter_times=filter_times,
                                noise_floor=noise_floor, start_model=None)
            inversion_dict = self.data_inverted.get(sounding, {}).get(f'{lam}_{filter_times[0]}_{filter_times[1]}_{noise_floor}')

        rms_value = inversion_dict.get('metadata').get('absrms')
        roughness_value = inversion_dict.get('metadata').get('phi_model')
        return np.array([roughness_value, rms_value])

    def analyse_inversion_cubic_spline_curvature(self, 
                                                sounding: str,
                                                layers,
                                                max_depth: float,
                                                test_range:tuple=(10, 10000, 30),
                                                layer_type:str = 'linear',
                                                filter_times: tuple = (7, 700),
                                                noise_floor: float = 0.025,
                                                normalize: bool = False,
                                                shared_mode: bool = False,
                                                fname: Union[str, bool] = None):

        test_tuple = test_range if len(test_range) == 3 else (test_range[0], test_range[1], 30)
        lambda_values = np.logspace(np.log10(test_tuple[0]), np.log10(test_tuple[1]), test_tuple[2])
        inv_points = []

        if not shared_mode:
            self.data_inversion(subset=[sounding], lam=list(lambda_values), layer_type=layer_type, layers=layers,
                                    verbose=False, max_depth=max_depth, filter_times=filter_times,
                                    noise_floor=noise_floor, start_model=None)
            self.logger.info('cubic-spline-search: All necessary inversions computed.')

        self.logger.info('cubic-spline-search: Starting search.')
        for lam in lambda_values:
            inv_point = self._l_curve_point(sounding=sounding, lam=lam, layer_type=layer_type, noise_floor=noise_floor,
                                            layers=layers,max_depth=max_depth, filter_times=filter_times)
            inv_points.append(inv_point)

        inv_points = np.array(inv_points)
        roughness_values = inv_points.T[0]
        rms_values = inv_points.T[1]
        lambda_values = np.array(lambda_values)

        sorted_indices = np.argsort(roughness_values)
        roughness_values = roughness_values[sorted_indices]
        rms_values = rms_values[sorted_indices]
        lambda_values = lambda_values[sorted_indices]

        if normalize:
            roughness_values = self._normalize_value(roughness_values, roughness_values.min(), roughness_values.max())
            rms_values = self._normalize_value(rms_values, rms_values.min(), rms_values.max())

        cs = CubicSpline(roughness_values, rms_values)
        cs_d = cs.derivative(nu=1)
        cs_dd = cs.derivative(nu=2)

        first_derivative = cs_d(roughness_values)
        second_derivative = cs_dd(roughness_values)

        curvature_values = second_derivative / (1 + first_derivative ** 2) ** (3 / 2)
        max_curvature_index = np.argmax(curvature_values)
        opt_lambda = lambda_values[max_curvature_index]
        self.logger.info(f'cubic-spline-search: Found optimal lambda ({opt_lambda}).')

        if not shared_mode:
            max_curvature_roughness = roughness_values[max_curvature_index]

            x_new = np.linspace(roughness_values.min(), roughness_values.max(), 500)
            y_new = cs(x_new)
            curvature_new = cs_dd(x_new) / (1 + cs_d(x_new) ** 2) ** (3 / 2)

            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].plot(roughness_values, rms_values, 'o', label='Original data')
            for i in np.arange(len(lambda_values)):
                ax[0].annotate(f'{lambda_values[i]:.0f}', (roughness_values[i], rms_values[i]), fontsize=8, ha='right',
                            textcoords="offset points", xytext=(10, 10))
            ax[0].plot(x_new, y_new, '-', label='Cubic spline fit')
            ax[0].vlines(max_curvature_roughness, rms_values.min(), rms_values.max(), colors='r', linestyles='--',
                        label='Optimum at lambda = {:.2f}'.format(opt_lambda))
            ax[0].set_xlabel('Roughness')
            ax[0].set_ylabel('RMS')
            ax[0].set_title(f'L-Curve (Sounding: {sounding})')
            ax[0].legend()
            ax[0].grid(True, which="both", alpha=.3)

            ax[1].plot(roughness_values, curvature_values, 'o', label='Curvature')
            for i in np.arange(len(lambda_values)):
                ax[1].annotate(f'{lambda_values[i]:.0f}', (roughness_values[i], curvature_values[i]), fontsize=8, ha='right',
                            textcoords="offset points", xytext=(10, 10))
            ax[1].plot(x_new, curvature_new, '-', label='Curvature fit')
            ax[1].vlines(max_curvature_roughness, curvature_values.min(), curvature_values.max(), colors='r', linestyles='--',
                        label='Max curvature at lambda = {:.2f}'.format(opt_lambda))
            ax[1].set_xlabel('Roughness')
            ax[1].set_ylabel('Curvature')
            ax[1].set_title(f'Curvature of the L-Curve')
            ax[1].grid(True, which="both", alpha=.3)

            time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            fig.tight_layout()
            fig.show()
            if fname or fname is None:
                file_name = f'lambda_analysis_cubic_spline_{time}_{sounding}_{filter_times[0]}_{filter_times[1]}.png' if fname is None else fname
                fig.savefig(self.folder_structure.get(
                    'data_inversion_analysis') / file_name)
            self.logger.info('cubic-spline-search: Plot created successfully.')

        return opt_lambda

    def analyse_inversion_gradient_curvature(self, 
                                            sounding: str,
                                            layers,
                                            max_depth: float,
                                            test_range:tuple=(10, 10000, 30),
                                            layer_type:str = 'linear',
                                            filter_times=(7, 700),
                                            noise_floor: float = 0.025,
                                            shared_mode: bool = False,
                                            normalize: bool = False,
                                            fname: Union[str, bool] = None):

        test_tuple = test_range if len(test_range) == 3 else (test_range[0], test_range[1], 30)
        lambda_values = np.logspace(np.log10(test_tuple[0]), np.log10(test_tuple[1]), test_tuple[2])
        inv_points = []

        if not shared_mode:
            self.data_inversion(subset=[sounding], lam=list(lambda_values), layer_type=layer_type, layers=layers,
                                    verbose=False, max_depth=max_depth, filter_times=filter_times,
                                    noise_floor=noise_floor, start_model=None)
            self.logger.info('gradient-search: All necessary inversions computed.')

        self.logger.info('gradient-search: Starting search.')
        for lam in lambda_values:
            inv_point = self._l_curve_point(sounding=sounding, lam=lam, layer_type=layer_type, noise_floor=noise_floor,
                                            layers=layers, max_depth=max_depth, filter_times=filter_times)
            inv_points.append(inv_point)
        inv_points = np.array(inv_points)
        roughness_values = inv_points.T[0]
        rms_values = inv_points.T[1]
        lambda_values = np.array(lambda_values)

        sorted_indices = np.argsort(roughness_values)
        roughness_values = roughness_values[sorted_indices]
        rms_values = rms_values[sorted_indices]
        lambda_values = lambda_values[sorted_indices]

        if normalize:
            roughness_values = self._normalize_value(roughness_values, roughness_values.min(), roughness_values.max())
            rms_values = self._normalize_value(rms_values, rms_values.min(), rms_values.max())

        first_derivative = np.gradient(rms_values, roughness_values)
        second_derivative = np.gradient(first_derivative, roughness_values)

        curvature_values = second_derivative / (1 + first_derivative ** 2) ** (3 / 2)
        max_curvature_index = np.argmax(curvature_values)
        opt_lambda = lambda_values[max_curvature_index]
        self.logger.info(f'gradient-search: Found optimal lambda ({opt_lambda}).')


        if not shared_mode:
            max_curvature_roughness = roughness_values[max_curvature_index]

            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].plot(roughness_values, rms_values, 'o', label='Original data')
            for i in np.arange(len(lambda_values)):
                ax[0].annotate(f'{lambda_values[i]:.0f}', (roughness_values[i], rms_values[i]), fontsize=8, ha='right',
                            textcoords="offset points", xytext=(10, 10))
            ax[0].vlines(max_curvature_roughness, rms_values.min(), rms_values.max(), colors='r', linestyles='--',
                        label='Optimum at lambda = {:.2f}'.format(opt_lambda))
            ax[0].set_xlabel('Roughness')
            ax[0].set_ylabel('RMS')
            ax[0].set_title(f'L-Curve (Sounding: {sounding})')
            ax[0].grid(True, which="both", alpha=.3)
            ax[0].legend()

            ax[1].plot(roughness_values, curvature_values, 'o', label='Curvature')
            for i in np.arange(len(lambda_values)):
                ax[1].annotate(f'{lambda_values[i]:.0f}', (roughness_values[i], curvature_values[i]), fontsize=8, ha='right',
                            textcoords="offset points", xytext=(10, 10))
            ax[1].vlines(max_curvature_roughness, curvature_values.min(), curvature_values.max(), colors='r', linestyles='--',
                        label='Max curvature at lambda = {:.2f}'.format(opt_lambda))
            ax[1].set_xlabel('Roughness')
            ax[1].set_ylabel('Curvature')
            ax[1].set_title('Curvature of the L-Curve')
            ax[1].grid(True, which="both", alpha=.3)

            time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            fig.tight_layout()
            fig.show()
            if fname or fname is None:
                file_name = f'lambda_analysis_gradient_{time}_{sounding}_{filter_times[0]}_{filter_times[1]}.png' if fname is None else fname
                fig.savefig(self.folder_structure.get(
                    'data_inversion_analysis') / file_name)
            self.logger.info('gradient-search: Plot created successfully.')

        return opt_lambda

    def analyse_inversion_golden_section(self,
                                        sounding: str,
                                        layers,
                                        max_depth: float,
                                        test_range: tuple = (10, 10000),
                                        layer_type: str = 'linear',
                                        filter_times = (7, 700),
                                        noise_floor: float = 0.025,
                                        max_iter: int = 20,
                                        tolerance: float = 0.01,
                                        shared_mode: bool = False,
                                        normalize: bool = False,
                                        fname: Union[str, bool] = None) -> float:
        phi = (1 + 5 ** 0.5) / 2

        lam1, lam4 = test_range
        lam2 = 10 ** ((np.log10(lam4) + phi * np.log10(lam1)) / (1 + phi))
        lam3 = 10 ** (np.log10(lam1) + np.log10(lam4) - np.log10(lam2))

        lambda_list = [lam1, lam2, lam3, lam4]

        self.logger.info('golden-section-search: Starting search.')
        p1 = self._l_curve_point(sounding=sounding, lam=lam1, layer_type=layer_type, noise_floor=noise_floor,
                                 layers=layers,max_depth=max_depth, filter_times=filter_times)
        p2 = self._l_curve_point(sounding=sounding, lam=lam2, layer_type=layer_type, noise_floor=noise_floor,
                                 layers=layers, max_depth=max_depth, filter_times=filter_times)
        p3 = self._l_curve_point(sounding=sounding, lam=lam3, layer_type=layer_type, noise_floor=noise_floor,
                                 layers=layers, max_depth=max_depth, filter_times=filter_times)
        p4 = self._l_curve_point(sounding=sounding, lam=lam4, layer_type=layer_type, noise_floor=noise_floor,
                                 layers=layers, max_depth=max_depth, filter_times=filter_times)
        
        boundry_points = np.stack((p1, p4))
        min_point, max_point = np.min(boundry_points, axis=0), np.max(boundry_points, axis=0)
        if normalize:
            p1 = self._normalize_value(p1, min_point, max_point)
            p2 = self._normalize_value(p2, min_point, max_point)
            p3 = self._normalize_value(p3, min_point, max_point)
            p4 = self._normalize_value(p4, min_point, max_point)

        points_list = [p1, p2, p3, p4]

        opt_lambda = None
        opt_point = None

        for i in range(max_iter):
            if (lam4 - lam1) / lam4 < tolerance:
                break
            else:
                curve2 = self._menger_curvature(p1, p2, p3)
                curve3 = self._menger_curvature(p2, p3, p4)

                if curve2 > curve3:
                    opt_lambda = lam2
                    opt_point = p2
                    lam4 = lam3
                    lam3 = lam2
                    p4 = p3
                    p3 = p2
                    lam2 = 10 ** ((np.log10(lam4) + phi * np.log10(lam1)) / (1 + phi))
                    p2 = self._l_curve_point(sounding=sounding, lam=lam2, layer_type=layer_type, noise_floor=noise_floor,
                                             layers=layers, max_depth=max_depth, filter_times=filter_times)
                    if normalize:
                        p2 = self._normalize_value(p2, min_point, max_point)
                    
                    lambda_list.append(lam2)
                    points_list.append(p2)
                else:
                    opt_lambda = lam3
                    opt_point = p3
                    lam1 = lam2
                    lam2 = lam3
                    p1 = p2
                    p2 = p3
                    lam3 = 10 ** (np.log10(lam1) + np.log10(lam4) - np.log10(lam2))
                    p3 = self._l_curve_point(sounding=sounding, lam=lam3, layer_type=layer_type, noise_floor=noise_floor,
                                             layers=layers, max_depth=max_depth, filter_times=filter_times)
                    if normalize:
                        p3 = self._normalize_value(p3, min_point, max_point)

                    lambda_list.append(lam3)
                    points_list.append(p3)
        self.logger.info(f'golden-section-search: Optimal Lambda ({opt_lambda}) after {i} iterations.')
        
        if not shared_mode:
            lambda_array = np.array(lambda_list)
            points_array = np.array(points_list)
            roughness_array = points_array.T[0]
            rms_array = points_array.T[1]

            sorted_indices = np.argsort(lambda_array)[::-1]
            roughness_values = roughness_array[sorted_indices]
            rms_values = rms_array[sorted_indices]
            lambda_values = lambda_array[sorted_indices]

            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.plot(roughness_values, rms_values, 'o', label='Original data')
            for i in np.arange(0, len(lambda_values), step=2):
                ax.annotate(f'{lambda_values[i]:.0f}', (roughness_values[i], rms_values[i]), fontsize=8, ha='right',
                            textcoords="offset points", xytext=(10, 10))
            ax.vlines(opt_point[0], rms_values.min(), rms_values.max(), colors='r', linestyles='--',
                        label='Optimum at lambda = {:.2f}'.format(opt_lambda))
            ax.set_xlabel('Roughness')
            ax.set_ylabel('RMS')
            ax.set_title(f'Sounding: {sounding}')
            ax.grid(True, which="both", alpha=.3)

            time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            fig.tight_layout()
            fig.show()
            if fname or fname is None:
                file_name = f'lambda_analysis_golden_section_{time}_{sounding}_{filter_times[0]}_{filter_times[1]}.png' if fname is None else fname
                fig.savefig(self.folder_structure.get(
                    'data_inversion_analysis') / file_name)
            self.logger.info('golden-section-search: Plott successfully created.')

        return opt_lambda

    def l_curve_plot(self, sounding: str,
                     layers,
                     max_depth: float,
                     test_range:tuple=(10, 10000, 30),
                     layer_type:str = 'linear',
                     filter_times=(7, 700),
                     noise_floor: float = 0.025,
                     fname: Union[str, bool] = None):

        test_tuple = test_range if len(test_range) == 3 else (test_range[0], test_range[1], 30)
        lambda_values = np.logspace(np.log10(test_tuple[0]), np.log10(test_tuple[1]), test_tuple[2])
        inv_points = []

        self.data_inversion(subset=[sounding], lam=list(lambda_values), layer_type=layer_type, layers=layers,
                                    verbose=False, max_depth=max_depth, filter_times=filter_times,
                                    noise_floor=noise_floor, start_model=None)
        self.logger.info('l_curve: All necessary inversions computed.')

        for lam in lambda_values:
            inv_point = self._l_curve_point(sounding=sounding, lam=lam, layer_type=layer_type, noise_floor=noise_floor,
                                            layers=layers, max_depth=max_depth, filter_times=filter_times)
            if inv_point is not None:
                inv_points.append(inv_point)
        if not inv_points:
            self.logger.error('l_curve_plot: Not inversion results found.')
            return
        inv_points = np.array(inv_points)
        roughness_values = inv_points.T[0]
        rms_values = inv_points.T[1]
        lambda_values = np.array(lambda_values)
        self.logger.info('l_curve: All points found.')

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        if roughness_values.size != rms_values.size:
            self.logger.error('l_curve_plot: Lengths of points not matched.')
            return
        ax.plot(roughness_values, rms_values, '.', label='L-Curve')
        for i in np.arange(len(lambda_values)):
            ax.annotate(f'{lambda_values[i]:.0f}', (roughness_values[i], rms_values[i]), fontsize=8, ha='right',
                           textcoords="offset points", xytext=(10, 10))
        ax.set_xlabel('Roughness')
        ax.set_ylabel('RMS')
        ax.set_title(f'Sounding: {sounding}')
        ax.legend()
        ax.grid(True, which="both", alpha=.3)

        time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        fig.tight_layout()
        fig.show()
        if fname or fname is None:
            file_name = f'lambda_analysis_l_curve_{time}_{sounding}_{filter_times[0]}_{filter_times[1]}.png' if fname is None else fname
            fig.savefig(self.folder_structure.get(
                'data_inversion_analysis') / file_name)
        self.logger.info('l-curve: L-curve plotted.')
        return fig

    def lambda_analysis_comparison(self, sounding: str,
                                   layers,
                                   max_depth: float,
                                   test_range:tuple=(10, 10000, 30),
                                   layer_type:str = 'linear',
                                   filter_times=(7, 700),
                                   noise_floor: float = 0.025,
                                   fname: Union[str, bool] = None):
        fig = self.l_curve_plot(sounding=sounding, layers=layers, max_depth=max_depth, noise_floor=noise_floor,
                                test_range=test_range, layer_type=layer_type, filter_times=filter_times,
                                fname=False)
        opt_grad = self.analyse_inversion_gradient_curvature(sounding=sounding, layers=layers, max_depth=max_depth, shared_mode=True,
                                test_range=test_range, layer_type=layer_type, filter_times=filter_times, noise_floor=noise_floor,
                                fname=False)
        opt_cubic = self.analyse_inversion_cubic_spline_curvature(sounding=sounding, layers=layers, max_depth=max_depth, shared_mode=True,
                                test_range=test_range, layer_type=layer_type, filter_times=filter_times, noise_floor=noise_floor,
                                fname=False)
        opt_golden = self.analyse_inversion_golden_section(sounding=sounding, layers=layers, max_depth=max_depth, shared_mode=True,
                                test_range=(test_range[0], test_range[1]), layer_type=layer_type, noise_floor=noise_floor,
                                filter_times=filter_times, fname=False)

        point_grad = self._l_curve_point(sounding=sounding, lam=opt_grad, layer_type=layer_type, noise_floor=noise_floor,
                                         layers=layers, max_depth=max_depth, filter_times=filter_times)
        point_cubic = self._l_curve_point(sounding=sounding, lam=opt_cubic, layer_type=layer_type, noise_floor=noise_floor,
                                         layers=layers, max_depth=max_depth, filter_times=filter_times)
        point_golden = self._l_curve_point(sounding=sounding, lam=opt_golden, layer_type=layer_type, noise_floor=noise_floor,
                                         layers=layers, max_depth=max_depth, filter_times=filter_times)

        ax = fig.get_axes()[0]
        ax.plot(*point_grad, 'd', label=f'Gradient-Based Curvature: {opt_grad:.2f}')
        ax.plot(*point_cubic, 's', label=f'Cubic-Spline-Based Curvature: {opt_cubic:.2f}')
        ax.plot(*point_golden, 'x', label=f'Golden Section Search: {opt_golden:.2f}')
        ax.set_title(f'Different Optimal $\lambda$-Values (Sounding: {sounding})', fontweight='bold')
        ax.set_xlabel('Roughness ($\Phi_{model}$)')
        ax.set_ylabel('Data-Misfit (RMS)')
        ax.legend()

        time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        fig.tight_layout()
        fig.show()
        if fname or fname is None:
            file_name = f'lambda_analysis_comparison_{time}_{sounding}_{filter_times[0]}_{filter_times[1]}.png' if fname is None else fname
            fig.savefig(self.folder_structure.get(
                'data_inversion_analysis') / file_name)
        self.logger.info('l-curve-comparison: Comparison plot succesfully created.')

    def analyse_inversion_plot(self, 
                            sounding: str,
                            layers,
                            max_depth: float,
                            test_range:tuple=(10, 10000, 30),
                            layer_type:str = 'linear',
                            filter_times=(7, 700),
                            noise_floor: Union[int, float] = 0.025,
                            unit: str = 'rhoa',
                            fname: Union[str, bool] = None) -> None:

        test_tuple = test_range if len(test_range) == 3 else (test_range[0], test_range[1], 30)
        lambda_values = np.logspace(np.log10(test_tuple[0]), np.log10(test_tuple[1]), test_tuple[2])

        nrows = np.ceil(len(lambda_values) / 5).astype(int)

        fig1, ax1 = plt.subplots(nrows, 5, figsize=(15, 3 * nrows))
        ax1 = ax1.ravel()

        fig2, ax2 = plt.subplots(nrows, 5, figsize=(15, 3 * nrows))
        ax2 = ax2.ravel()

        self.data_inversion(subset=[sounding], lam=list(lambda_values), layer_type=layer_type, layers=layers,
                            verbose=False, max_depth=max_depth, filter_times=filter_times,
                            noise_floor=noise_floor, start_model=None)
        for i, lam in enumerate(lambda_values):
            inv_name = f'{lam}_{filter_times[0]}_{filter_times[1]}_{noise_floor}'
            filter_name = f'{filter_times[0]}_{filter_times[1]}_{noise_floor}'
            inverted_data = self.data_inverted.get(sounding, {}).get(inv_name)
            filtered = self.data_filtered.get(sounding, {}).get(filter_name)

            filtered_data = filtered.get('data')
            inversion_data = inverted_data.get('data')

            obs_unit = filtered_data[unit]
            response_unit = inversion_data[unit].dropna()
            resp_sgnl = inversion_data['E/I[V/A]'].dropna()

            if unit == 'rhoa':
                unit_label_ax = r'$\rho_a$ ($\Omega$m)'
            else:
                unit_label_ax = r'$\sigma_a$ (S/m)'

            ax1[i].loglog(filtered_data['Time'], resp_sgnl, '-k', label='inversion', zorder=3)
            ax1[i].plot(filtered_data['Time'], filtered_data['E/I[V/A]'], marker='v', label='observed',
                       zorder=2)
            ax1[i].plot(filtered_data['Time'], filtered_data['Err[V/A]'], label='error', zorder=1, alpha=0.4,
                       linestyle='dashed')
            ax1[i].set_xlabel('time (s)', fontsize=16)
            ax1[i].set_ylabel(r'$\partial B_z/\partial t$ [V/m²]', fontsize=16)
            ax1[i].grid(True, which="both", alpha=.3)

            ax2[i].plot(filtered_data['Time'], response_unit, '-k', label='inversion', zorder=3)
            ax2[i].plot(filtered_data['Time'], obs_unit, marker='v', label='observed', zorder=2)
            ax2[i].set_title(f'{lam: .2f}', fontweight='bold', fontsize=16)
            ax2[i].set_xlabel('time (s)', fontsize=16)
            ax2[i].set_ylabel(unit_label_ax, fontsize=16)
            ax2[i].set_xscale('log')
            ax2[i].yaxis.tick_right()
            ax2[i].yaxis.set_label_position("right")
            ax2[i].grid(True, which="both", alpha=.3)

        time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        fig1.suptitle(f'Sounding: {sounding}', fontweight='bold', fontsize=18)
        fig1.tight_layout()
        fig1.show()
        fig2.tight_layout()
        fig2.show()
        
        if fname or fname is None:
            file_name_1 = f'lambda_analysis_signal_{time}_{sounding}_{filter_times[0]}_{filter_times[1]}.png' if fname is None else f'signal_{fname}'
            fig1.savefig(self.folder_structure.get(
                'data_inversion_analysis') / file_name_1)
            file_name_2 = f'lambda_analysis_{unit}_{time}_{sounding}_{filter_times[0]}_{filter_times[1]}.png' if fname is None else f'{unit}_{fname}'
            fig2.savefig(self.folder_structure.get(
                'data_inversion_analysis') / file_name_2)

    def optimised_inversion_plot(self, sounding: str = None,
                                 lam: Union[int, float] = 600,
                                 layer_type: str = 'linear',
                                 layers: Union[int, float, dict, np.ndarray] = 4.5,
                                 max_depth: Union[float, int] = None,
                                 start_model: np.ndarray = None,
                                 noise_floor: Union[float, int] = 0.025,
                                 test_range: tuple = (10, 10000, 30),
                                 unit: str = 'rhoa',
                                 scale: str = 'lin',
                                 filter_times=(7, 700),
                                 verbose: bool = True,
                                 limits_depth: Optional[tuple] = None,
                                 limits_rho: Optional[tuple] = None,
                                 limits_time: Optional[tuple] = None,
                                 limits_signal: Optional[tuple] = None,
                                 limits_rhoa: Optional[tuple] = None,
                                 colormap = None,
                                 fname: Union[str, bool] = None) -> None:


        test_tuple = test_range if len(test_range) == 3 else (test_range[0], test_range[1], 30)
        lambda_values = np.logspace(np.log10(test_tuple[0]), np.log10(test_tuple[1]), test_tuple[2])
        inv_points = []

        self.data_inversion(subset=[sounding], lam=list(lambda_values) + [lam], layer_type=layer_type, layers=layers,
                            max_depth=max_depth, filter_times=filter_times,
                            start_model=start_model, noise_floor=noise_floor,
                            verbose=False)
        
        for l in lambda_values:
            inv_point = self._l_curve_point(sounding=sounding, lam=l, layer_type=layer_type, noise_floor=noise_floor,
                                            layers=layers, max_depth=max_depth, filter_times=filter_times)
            if inv_point is not None:
                inv_points.append(inv_point)
        if not inv_points:
            self.logger.error('l_curve_plot: Not inversion results found.')
            return

        inv_points = np.array(inv_points)
        roughness_values = inv_points.T[0]
        rms_values = inv_points.T[1]
        lambda_values = np.array(lambda_values)

        inv_name = f'{lam}_{filter_times[0]}_{filter_times[1]}_{noise_floor}'
        filter_name = f'{filter_times[0]}_{filter_times[1]}_{noise_floor}'
        inverted_data = self.data_inverted.get(sounding, {}).get(inv_name)
        filtered_data = self.data_filtered.get(sounding, {}).get(filter_name)

        if inverted_data is None:
            self.logger.error(f'No inversion data found for {sounding}.')
            return

        if filtered_data is None:
            self.logger.error(f'No filtered data found for {sounding}.')
            return

        filtered_data = filtered_data.get('data')
        inversion_data = inverted_data.get('data')
        inversion_metadata = inverted_data.get('metadata')

        thks = np.r_[inversion_data['modelled_thickness'].dropna(), 0]
        model_res = inversion_data['resistivity_model'].dropna()
        chi2 = inversion_metadata.get('chi2')
        rrms = inversion_metadata.get('relrms')
        abs_rms = inversion_metadata.get('absrms')
        roughness = inversion_metadata.get('phi_model')

        if scale == 'log':
            log_rhoa = True
        elif scale == 'lin':
            log_rhoa = False
        else:
            raise SyntaxError('input {} not valid for Argument scale'.format(scale))
        
        if unit == 'rhoa':
            rhoa_title = 'Apparent Resistivity'
            model_title = 'Resistivity'
            res2con = False
        elif unit == 'sigma':
            rhoa_title = 'Apparent Conductivity'
            model_title = 'Conductivity'
            res2con = True
        else:
            raise SyntaxError('input {} not valid for Argument unit'.format(unit))
        
        fig, ax = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)
        ax = ax.ravel()

        col = self._get_color(sounding=sounding, cmap=colormap)

        plot_model(
            model=np.column_stack((thks, model_res)), ax=ax[2],
            res2con=res2con, xlimits=limits_rho, ylimits=limits_depth,
            title=f'Model of {model_title} at Depth', color=col, add_bottom=2
        )
        
        plot_data(
            time=filtered_data['Time'], ax=ax[:2],
            signal=filtered_data['E/I[V/A]'], rhoa=filtered_data['rhoa'],
            signal_title='Impulse Response', rhoa_title=f'{rhoa_title}',
            signal_label='observed', rhoa_label='observed',
            sub0_color_signal=col, sub0_color_rhoa=col,
            log_rhoa=log_rhoa, color=col, noise_signal=filtered_data['Err[V/A]'],
            res2con=res2con, legend=True, marker='v',
            xlimits=limits_time, ylimits_signal=limits_signal,
            ylimits_rhoa=limits_rhoa
        )

        plot_data(
            time=filtered_data['Time'], ax=ax[:2],
            signal=inversion_data['E/I[V/A]'].dropna(), rhoa=inversion_data['rhoa'].dropna(),
            signal_title='Impulse Response', rhoa_title=f'{rhoa_title}',
            signal_label='inversion', rhoa_label='inversion',
            log_rhoa=log_rhoa, color='k',
            res2con=res2con, legend=True,
            xlimits=limits_time, ylimits_signal=limits_signal,
            ylimits_rhoa=limits_rhoa
            )

        ax[3].plot(roughness_values, rms_values, 'o', label='L-Curve')
        ax[3].plot(roughness, abs_rms, 's', label=f'Optimal Lambda: {lam}')
        for i in np.arange(len(lambda_values)):
            ax[3].annotate(f'{lambda_values[i]:.0f}', (roughness_values[i], rms_values[i]), fontsize=8, ha='right',
                        textcoords="offset points", xytext=(10, 10))
        ax[3].set_xlabel('Roughness ($\Phi_{model}$)')
        ax[3].set_ylabel('Data-Misfit (RMS)')
        ax[3].yaxis.tick_right()
        ax[3].yaxis.set_label_position("right")
        ax[3].grid(True, which="both", alpha=.3)
        ax[3].set_title('L-Curve', fontsize=18, pad=12)
        ax[3].legend(loc='lower left')

        label_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
        for axis, label in zip(ax, label_list):
            axis.text(0.96, 0.96, f'({label})', transform=axis.transAxes, fontsize=18, zorder=5,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(facecolor='xkcd:light grey', boxstyle='round,pad=0.4', alpha=0.3))

        fig.suptitle(
            f'Sounding: {sounding:<6}\t$\lambda$: {lam:<4.0f}\n'
            f'Relative RMS: {rrms:<.2f}%\t$\chi^2$: {chi2:<8.2f}',
            fontsize=22,
            fontweight='bold'
        )
        fig.show()

        target_dir = self.folder_structure.get('data_inversion_plot')
        time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if fname or fname is None:
            file_name = f'opt_{sounding}_{time}_{unit}.png' if fname is None else fname
            fig.savefig(target_dir / file_name)