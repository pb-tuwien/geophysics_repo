# -*- coding: utf-8 -*-
"""
Created on Wed May 21 10:37:16 2025

Read and write classes for different file types

@author: peter balogh @ TU Wien, Research Unit Geophysics
"""

#%% Imports
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union

#%% OHM file
class OHMfile:
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
        Path to the OHM-file.
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
            raise TypeError(f'Invalid type for filepath: {type(pathj)}')
    
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
            raise TypeError(f'Must provide an OHM file: {self.filepath.suffix} was given.')

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

class TEMfile:
    pass

#%% TIM file

class TIMfile:
    pass