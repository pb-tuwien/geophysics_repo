# -*- coding: utf-8 -*-
"""
Created on Wed May 21 10:37:16 2025

Read and write classes for different file types

@author: peter balogh @ TU Wien, Research Unit Geophysics
"""

#%% Imports
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional
from .utils import BaseFunction

#%% OHM file
class OHMfile(BaseFunction):
    def __init__(self):
        self.data = None
        self.electrodes = None
        self.filepath = None

        self.__allowed_suffix = '.ohm'

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
    
    @property
    def filepath(self) -> Union[Path, None]:
        """
        Path to the file.
        """
        return self.__filepath
    
    @filepath.setter
    def filepath(self, path: Union[str, Path, None]):
        """
        Setter of the filepath property.
        """
        if path is None:
            self.__filepath = path
        elif isinstance(path, (str, Path)):
            self.__filepath = Path(path)
        else:
            raise TypeError(f'Invalid type for filepath: {type(path)}')
        
    @property
    def data(self) -> Optional[pd.DataFrame]:
        """
        A DataFrame containing the data.
        """
        return self.__data
    
    @data.setter
    def data(self, data: Optional[pd.DataFrame]):
        """
        Setter of the data property.
        """
        if isinstance(data, (pd.DataFrame, type(None))):
            self.__data = data
        else:
            raise TypeError(f'Invalid type for data: {type(pd.DataFrame)}')
        
    @property
    def electrodes(self) -> Optional[pd.DataFrame]:
        """
        A DataFrame containing the coordinates of the electrodes.
        """
        return self.__electrodes
    
    @electrodes.setter
    def electrodes(self, electrodes: Optional[pd.DataFrame]):
        """
        Setter of the electrodes property.
        """
        if isinstance(electrodes, (pd.DataFrame, type(None))):
            self.__electrodes = electrodes
        else:
            raise TypeError(f'Invalid type for electrodes: {type(electrodes)}')
    
    @property
    def suffix(self):
        """
        Suffix of the filepath. Must be '.ohm'.
        """
        if self.filepath is None:
            return None
        elif self.filepath.suffix == self.__allowed_suffix:
            return self.filepath.suffix
        else:
            raise TypeError(f'Must provide {self.__allowed_suffix}-file: {self.filepath.suffix} was given.')

    @classmethod
    def read(cls, filepath: Union[str, Path]):
        """
        Read an OHM-file.
        
        Parameters
        ----------
        filepath: str or Path
            Path to the file which should be read.
        """
        instance = cls()

        instance.filepath = filepath
        if not instance.filepath.exists():
            raise FileNotFoundError('No file found at given path.')
        _ = instance.suffix

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
        
        instance = cls()

        instance.filepath = filepath
        if instance.filepath.exists():
            print('Overwriting existing file.')
        _ = instance.suffix
        
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

#%% TEM file
class TEMfile(BaseFunction):
    def __init__(self):
        self.data = None
        self.electrodes = None
        self.filepath = None

        self.__allowed_suffix = '.tem'
        self.__data_delimiter = '\t'

        self.__reading_lines = [
            r'([\w\W]+)\s+Date:\s*([\w\W]+)',
            r'Place:\s*([\w\W]+)',
            r'#Set\s*([\w\W]+)',
            r'Time-Range\s+(\d+)\s+Stacks\s+(\d+)\s+deff=\s*(\d+)\s*us\s*I=(\d+(?:\.\d+)?)\s*A\s*FILTR=(\d+)\s*Hz\s*AMPLIFER=(\w+)',
            r'T-LOOP \(m\)\s+([\d\.]+)\s+R-LOOP \(m\)\s+([\d\.]+)\s+TURN=\s*(\d+)',
            r'Comments:\s*([\w\W]+)',
            r'Location:\s*x=\s*(None|[+-]?[\d\.]+)\s*y=\s*(None|[+-]?[\d\.]+)\s*z=\s*(None|[+-]?[\d\.]+)'
        ]

        self.__writing_lines = [
            '{device}\tDate:\t{date}',
            'Place:\t{place}',
            '#Set\t{name}',
            'Time-Range\t{timerange}\tStacks\t{stacks}\tdeff= {deff} us\tI={current} A\tFILTR={filter} Hz\tAMPLIFER={amplifier}',
            'T-LOOP (m)\t{tloop:.3f}\tR-LOOP (m)\t{rloop:.3f}\tTURN=\t{turn}',
            'Comments:\t{comments}',
            'Location:x=\t{x:+.3f}\ty=\t{y:+.3f}\tz=\t{z:+.3f}'
        ]

        self.__metadata_keys = [
            'device', 'date', 
            'place', 
            'name', 
            'timerange', 'stacks', 'deff', 'current', 'filter', 'amplifier', 
            'tloop', 'rloop', 'turn', 
            'comments', 
            'x', 'y', 'z'
        ]

        self.__header_len = len(self.__reading_lines)
    
    @property
    def filepath(self) -> Optional[Path]:
        """
        Path to the file.
        """
        return self.__filepath
    
    @filepath.setter
    def filepath(self, path: Union[str, Path, None]):
        """
        Setter of the filepath property.
        """
        if path is None:
            self.__filepath = path
        elif isinstance(path, (str, Path)):
            self.__filepath = Path(path)
        else:
            raise TypeError(f'Invalid type for filepath: {type(path)}')
        
    @property
    def data(self) -> Optional[dict]:
        """
        A dictionary containing the data.

        Structure
        ---------
        First layer: 
            Keys correspond to the names of the sounding.
            The values are dictionaries (-> Second layer).
        Second layer:
            Contains the keys 'metadata' and 'data'.
            'metadata' is a dictionary with the metadate and 'data' is a pd.DataFrame with the actual data.

        Example
        -------
        {'sounding1': {
        'metadata': {'name': 'Test', ... },
        'data': pandas.DataFrame()}, 'sounding2': ... }
        """
        return self.__data
    
    @data.setter
    def data(self, data: Optional[dict]):
        """
        Setter of the data property.
        """
        if isinstance(data, (dict, type(None))):
            self.__data = data
        else:
            raise TypeError(f'Invalid type for data: {type(data)}')
        
    @property
    def electrodes(self) -> Optional[dict]:
        """
        A dictionary containing the coordinates of the electrodes.

        Structure
        ---------
        First layer: 
            Keys correspond to the names of the sounding.
            The values are dictionaries (-> Second layer).
        Second layer:
            Contains the keys 'x', 'y', and 'z'.
            These are the coordinates of the sounding.

        Example
        -------
        {'sounding1': {'x': 5.0, 'y': 0.0, 'z': 0.0}, 
        'sounding2': ... }
        """
        return self.__electrodes
    
    @electrodes.setter
    def electrodes(self, electrodes: Optional[dict]):
        """
        Setter of the electrodes property.
        """
        if isinstance(electrodes, (dict, type(None))):
            self.__electrodes = electrodes
        else:
            raise TypeError(f'Invalid type for electrodes: {type(electrodes)}')
    
    @property
    def suffix(self):
        """
        Suffix of the filepath.
        """
        if self.filepath is None:
            return None
        elif self.filepath.suffix == self.__allowed_suffix:
            return self.filepath.suffix
        else:
            print(f'Error at: {self.filepath}')
            raise TypeError(f'Must provide an {self.__allowed_suffix}-file: {self.filepath.suffix} was given.')
    
    @classmethod
    def read(cls, filepath: Union[str, Path]):
        """
        Read an the file.
        
        Parameters
        ----------
        filepath: str or Path
            Path to the file which should be read.
        
        Returns
        -------
        dict
        A dictionary with the all the soundings. For each sounding there is a 'metadata' dictionary and 'data' DataFrame.
        
        """
        instance = cls()

        instance.filepath = filepath
        if not instance.filepath.exists():
            raise FileNotFoundError('No file found at given path.')
        _ = instance.suffix

        all_text = instance.filepath.read_text()
        separator = all_text.split(maxsplit=1)[0]
        soundings = [separator + sounding for sounding in all_text.split(separator) if sounding != '']

        dictionary = {}

        for sounding in soundings:
            lines = sounding.splitlines()
            columns = lines[instance.__header_len].strip().split(instance.__data_delimiter)
            raw_data = [line.strip().split(instance.__data_delimiter) for line in lines[instance.__header_len + 1:]]
            dataframe = instance.safe_to_numeric(pd.DataFrame(columns=columns, data=raw_data))
            dataframe.replace('nan', np.nan, inplace=True)

            pattern = r'\s*\n\s*'.join(instance.__reading_lines)
            matched = re.search(pattern, sounding)
            if matched:
                stripped = [entry.strip() for entry in matched.groups()]
                correct_type = BaseFunction.safe_to_numeric(stripped)
                metadata = dict(zip(instance.__metadata_keys, correct_type))
            else:
                raise ValueError('Metadata was not found')
            
            sounding_name = metadata.get('name', instance.filepath.stem)
            dictionary[sounding_name] = {'metadata': metadata, 'data': dataframe}

        instance.data = dictionary
        return instance
    
    @classmethod
    def write(cls, filepath: Union[str, Path], 
              data: Union[dict]):
        
        instance = cls()

        instance.filepath = filepath
        _ = instance.suffix
        
        instance.data = data

        with open(file=filepath, mode='w') as file:
            for name, soundings in instance.data.items():
                metadata = soundings.get('metadata')
                dataframe = soundings.get('data')
                if metadata is None or dataframe is None:
                    print('Skipped sounding {name}.')
                    continue
                metadata_str = '\n'.join(instance.__writing_lines).format(**metadata)
                columns_str = instance.__data_delimiter.join(dataframe.columns)
                data_str = '\n'.join([instance.__data_delimiter.join(map(str, row)) for row in dataframe.values])
                
                file.write(metadata_str + '\n')
                file.write(columns_str + '\n')
                file.write(data_str + '\n')
        
        return instance

#%% TIM file

class TIMfile(BaseFunction):
    def __init__(self):
        self.data = None
        self.electrodes = None
        self.filepath = None

        self.__allowed_suffix = '.tim'
        self.__data_delimiter = ','

        self.__reading_lines = [
            r'Name:\s*([\w\W]+)',
            r'Lambda:\s*(\d+(?:\.\d+)?)\s*Maximal Depth:\s+(\d+(?:\.\d+)?)\s*Filter time:\s*(\d+(?:\.\d+)?)\s*to\s*(\d+(?:\.\d+)?)\s*\[us\]',
            r'Chi2:\s*([\d\.eE+-]+)\s*RelRMS:\s*([\d\.eE+-]+)\s*AbsRMS:\s*([\d\.eE+-]+)\s*Phi model:\s*([\d\.eE+-]+)'
        ]

        self.__writing_lines = [
            'Name:\t{name}',
            'Lambda:\t{lambda}\tMaximal Depth:\t{max_depth}\tFilter time:\t{filtertime_min} to {filtertime_max} [us]',
            'Chi2:\t{chi2}\tRelRMS:\t{relrms}\tAbsRMS:\t{absrms}\tPhi model:\t{phi_model}'
        ]

        self.__metadata_keys = [
            'name',
            'lambda', 'max_depth', 'filtertime_min', 'filtertime_max',
            'chi2', 'relrms', 'absrms', 'phi_model'
        ]

        self.__header_len = len(self.__reading_lines)
    
    @property
    def filepath(self) -> Optional[Path]:
        """
        Path to the file.
        """
        return self.__filepath
    
    @filepath.setter
    def filepath(self, path: Union[str, Path, None]):
        """
        Setter of the filepath property.
        """
        if path is None:
            self.__filepath = path
        elif isinstance(path, (str, Path)):
            self.__filepath = Path(path)
        else:
            raise TypeError(f'Invalid type for filepath: {type(path)}')
        
    @property
    def data(self) -> Optional[dict]:
        """
        A dictionary containing the data.

        Structure
        ---------
        First layer: 
            Keys correspond to the names of the sounding.
            The values are dictionaries (-> Second layer).
        Second layer:
            Contains the keys 'metadata' and 'data'.
            'metadata' is a dictionary with the metadate and 'data' is a pd.DataFrame with the actual data.

        Example
        -------
        {'sounding1': {
        'metadata': {'name': 'Test', ... },
        'data': pandas.DataFrame()}, 'sounding2': ... }
        """
        return self.__data
    
    @data.setter
    def data(self, data: Optional[dict]):
        """
        Setter of the data property.
        """
        if isinstance(data, (dict, type(None))):
            self.__data = data
        else:
            raise TypeError(f'Invalid type for data: {type(data)}')
        
    @property
    def electrodes(self) -> Optional[dict]:
        """
        A dictionary containing the coordinates of the electrodes.

        Structure
        ---------
        First layer: 
            Keys correspond to the names of the sounding.
            The values are dictionaries (-> Second layer).
        Second layer:
            Contains the keys 'x', 'y', and 'z'.
            These are the coordinates of the sounding.

        Example
        -------
        {'sounding1': {'x': 5.0, 'y': 0.0, 'z': 0.0}, 
        'sounding2': ... }
        """
        return self.__electrodes
    
    @electrodes.setter
    def electrodes(self, electrodes: Optional[dict]):
        """
        Setter of the electrodes property.
        """
        if isinstance(electrodes, (dict, type(None))):
            self.__electrodes = electrodes
        else:
            raise TypeError(f'Invalid type for electrodes: {type(electrodes)}')
    
    @property
    def suffix(self):
        """
        Suffix of the filepath.
        """
        if self.filepath is None:
            return None
        elif self.filepath.suffix == self.__allowed_suffix:
            return self.filepath.suffix
        else:
            print(f'Error at: {self.filepath}')
            raise TypeError(f'Must provide an {self.__allowed_suffix}-file: {self.filepath.suffix} was given.')
    
    @classmethod
    def read(cls, filepath: Union[str, Path]):
        """
        Read an the file.
        
        Parameters
        ----------
        filepath: str or Path
            Path to the file which should be read.
        
        Returns
        -------
        dict
        A dictionary with the all the soundings. For each sounding there is a 'metadata' dictionary and 'data' DataFrame.
        
        """
        instance = cls()

        instance.filepath = filepath
        if not instance.filepath.exists():
            raise FileNotFoundError('No file found at given path.')
        _ = instance.suffix

        all_text = instance.filepath.read_text()
        separator = all_text.split(maxsplit=1)[0]
        soundings = [separator + sounding for sounding in all_text.split(separator) if sounding != '']

        dictionary = {}

        for sounding in soundings:
            lines = sounding.splitlines()
            columns = lines[instance.__header_len].strip().split(instance.__data_delimiter)
            raw_data = [line.strip().split(instance.__data_delimiter) for line in lines[instance.__header_len + 1:]]
            dataframe = instance.safe_to_numeric(pd.DataFrame(columns=columns, data=raw_data))
            dataframe.replace('nan', np.nan, inplace=True)

            pattern = r'\s*\n\s*'.join(instance.__reading_lines)
            matched = re.search(pattern, sounding)
            if matched:
                stripped = [entry.strip() for entry in matched.groups()]
                correct_type = BaseFunction.safe_to_numeric(stripped)
                metadata = dict(zip(instance.__metadata_keys, correct_type))
            else:
                raise ValueError('Metadata was not found')
            
            sounding_name = metadata.get('name', instance.filepath.stem)
            dictionary[sounding_name] = {'metadata': metadata, 'data': dataframe}

        instance.data = dictionary
        return instance
    
    @classmethod
    def write(cls, filepath: Union[str, Path], 
              data: Union[dict]):
        
        instance = cls()

        instance.filepath = filepath
        _ = instance.suffix
        
        instance.data = data

        with open(file=filepath, mode='w') as file:
            for name, soundings in instance.data.items():
                metadata = soundings.get('metadata')
                dataframe = soundings.get('data')
                if metadata is None or dataframe is None:
                    print('Skipped sounding {name}.')
                    continue
                metadata_str = '\n'.join(instance.__writing_lines).format(**metadata)
                columns_str = instance.__data_delimiter.join(dataframe.columns)
                data_str = '\n'.join([instance.__data_delimiter.join(map(str, row)) for row in dataframe.values])
                
                file.write(metadata_str + '\n')
                file.write(columns_str + '\n')
                file.write(data_str + '\n')
        
        return instance