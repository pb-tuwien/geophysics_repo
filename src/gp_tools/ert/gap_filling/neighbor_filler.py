# -*- coding: utf-8 -*-
"""
Created on Wed May 21 11:16:13 2025

Read and write classes for different file types

@author: peter balogh @ TU Wien, Research Unit Geophysics
"""

#%% Imports

from .base_filler import GapFiller
from pathlib import Path
from typing import Union, Literal
import numpy as np
from .imputer import KNNfiller

class GapFillerNeighbor(GapFiller):
    def __init__(self, neighbors: int = 6,
                 horizontal_weight: Union[int, float] = 1,
                 missing_values: Union[float, int, None] = np.nan,
                 weights: Union[Literal['distance', 'distance_sqrt', 'uniform'], callable] = 'distance',
                 metric: Union[Literal['weighted_euclidean', 'nan_euclidean'], callable] = 'weighted_euclidean',
                 base_dir = Path.cwd()):
        
        imputer = KNNfiller(
            missing_values=missing_values,
            neighbors=neighbors,
            weights=weights,
            metric=metric,
            horizontal_weight=horizontal_weight
        )
        super().__init__(imputer=imputer, base_dir=base_dir)
