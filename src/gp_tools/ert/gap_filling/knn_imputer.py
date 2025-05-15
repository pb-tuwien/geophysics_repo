# -*- coding: utf-8 -*-
"""
Created on Thu May 15 13:46:16 2025

Utility functions for working with ERT data.

@author: peter balogh @ TU Wien, Research Unit Geophysics
"""
#%% Imports
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from typing import Union, Literal
from sklearn.impute import KNNImputer

# %%

class KNNfiller:
    def __init__(self, missing_values: Union[float, int, None] = np.nan,
                 neighbors: int = 6,
                 weights: Union[Literal['distance', 'uniform'], callable] = 'distance',
                 metric: Union[Literal['weighted_euclidean', 'nan_euclidean'], callable] = 'weighted_euclidean',
                 horizontal_weight: Union[int, float] = 1
                 ):
        self.horizontal_weight = horizontal_weight
        self.neighbors = neighbors

        metric = self.__weighted_distance if metric == 'weighted_euclidean' else metric

        self.__imputer = KNNImputer(
            missing_values=missing_values,
            n_neighbors=self.neighbors,
            weights=weights,
            metric=metric)
    
    @property
    def horizontal_weight(self) -> int:
        """
        Factor to determine how much stronger horizontal distance should weighted compared to vertical distance.
        """
        return self.__horizontal_weight
    
    @horizontal_weight.setter
    def horizontal_weight(self, factor: Union[int, float]):
        if isinstance(factor, (int, float)):
            self.__horizontal_weight = factor
        else:
            raise TypeError(f'{type(factor)} is an invalid type for the horizantal_weight. Must be either a integer or float.')

    @property
    def neighbors(self) -> int:
        """
        Number of nearest neighbors considered.
        """
        return self.__neighbors
    
    @neighbors.setter
    def neighbors(self, neighbors: int):
        if isinstance(neighbors, int):
            self.__neighbors = neighbors
        else:
            raise TypeError(f'{type(neighbors)} is an invalid type for the neighbors. Must be a integer.')

    def __weighted_distance(self, x, y, missing_values=np.nan):
        x_distance = (x[0] - y[0])
        y_distance = (x[1] - y[1])
        return np.sqrt(x_distance**2 + (y_distance**2)*self.horizontal_weight)
    
    def fit_transform(self, data: Union[pd.DataFrame, np.ndarray]):
        if isinstance(data, pd.DataFrame):
            columns = data.columns
            data = data.values
            imputed = self.__imputer.fit_transform(data)
            return pd.DataFrame(data=imputed, columns=columns)
        else:
            return self.__imputer.fit_transform(data)