# -*- coding: utf-8 -*-
"""
Created on Wed May 21 19:32:29 2025

Basic utility functions and classes

@author: peter balogh @ TU Wien, Research Unit Geophysics
"""

#%% Imports

from pathlib import Path
import logging
from typing import Optional, Union
import numpy as np
import pandas as pd
import time
import datetime
from functools import wraps

#%% Timer decorator

def time_logger(func):
    """
    This is a decorator-type function, which measures the runtime of a function.

    Returns
    -------
    Function
        This is a decorator.
        It wraps around a function and measures its runtime.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        cls_instance = args[0] if args else None
        logger = getattr(cls_instance, 'logger', None)
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            if logger is not None:
                logger.error(f'An error occurred in {func.__name__}: {e}')
            raise
        finally:
            end_time = time.time()
            elapsed_time = end_time - start_time
            if logger is not None:
                logger.info(f'Time taken by {func.__name__}: {elapsed_time:.2f} seconds.')
            else:
                print(f'Time taken by {func.__name__}: {elapsed_time:.2f} seconds.')
    return wrapper

#%% Timer class
class TimerError(Exception):

    """A custom exception used to report errors in use of Timer class"""


class Timer:

    def __init__(self):
        self._start_time = None
        self._stop_time = None


    def start(self):
        """Start a new timer"""

        if self._start_time is not None:
            raise TimerError("Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()


    def stop(self, prefix='run-'):
        """Stop the timer, and report the elapsed time"""

        if self._start_time is None:
            raise TimerError("Timer is not running. Use .start() to start it")

        self._stop_time = time.perf_counter()
        elapsed_time = self._stop_time - self._start_time
        elapsed_time_string = str(datetime.timedelta(seconds=elapsed_time))[:-3]
        self._start_time = None

        print(f"Elapsed {prefix}time (hh:mm:ss.sss): {elapsed_time_string:s}")

        return elapsed_time, elapsed_time_string
#%% BaseFunction class
class BaseFunction:
    def _setup_logger(self, log_path=None):
        """
        This function sets a logger for the class.

        Returns
        -------
        logger : logging.Logger
            Returns created logger.

        """
        try:
            logger = logging.getLogger(self.__class__.__name__)
            if not logger.hasHandlers():

                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.INFO)
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)

                if log_path is not None:
                    log_file_path = Path(log_path)
                    log_file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_handler = logging.FileHandler(log_file_path)
                    file_handler.setLevel(logging.INFO)
                    file_handler.setFormatter(formatter)
                    logger.addHandler(file_handler)
            logger.setLevel(logging.INFO)
            return logger
        except Exception as e:
            print(f'Unexpected Error occurred in _setup_logger: {e}')
            raise

    def _output(self, message: str, level: str = 'info'):
        """
        This function logs the message with the specified level.
        It knows the levels 'info', 'warning', and 'error'.
        Unknown levels will be logged as 'info'.

        Parameters
        ----------
        message : str
            Message to be logged.
        level : str, optional
            Level of the message. The default is 'info'.

        Returns
        -------
        None.

        """
        if hasattr(self, 'logger'):
            if level == 'warning':
                self.logger.warning(message)
            elif level == 'error':
                self.logger.error(message)
            else:
                self.logger.info(message)
        else:
            print(f'{level.upper()}: {message}')

    @staticmethod
    def safe_to_numeric(data: Union[int, float, str, list, np.ndarray, pd.DataFrame]) -> Union[int, float, str, list, np.ndarray, pd.DataFrame]:
        """
        Converts individual values or arrays to numeric values if possible.
        Tries to convert to int first, then to float.

        Parameters
        ----------
        data : scalar, list, or pd.DataFrame

        Returns
        -------
        Converted data with numeric values where possible.
        """
        def convert_value(value):
            try:
                return pd.to_numeric(value, downcast='integer')
            except (ValueError, TypeError):
                try:
                    return pd.to_numeric(value, downcast='float')
                except (ValueError, TypeError):
                    return value

        if isinstance(data, pd.DataFrame):
            for col in data.columns:
                data[col] = data[col].apply(convert_value)
            return data
        elif isinstance(data, np.ndarray):
            return np.array([convert_value(val) for val in data])
        elif isinstance(data, (list, tuple)):
            return [convert_value(val) for val in data]
        else:
            return convert_value(data)

    def _type_changer(self, type_string: str, value: Union[int, float, bool, str]) -> Union[int, float, bool, str]:
        """
        Changes the type of the value.
        Parameters
        ----------
        type_string : str
            Type of the value.
        value : str
            Value to be changed.

        Returns
        -------
        changed : int, float, bool, str
            Changed value.
        """
        value = value.strip()
        if type_string == 'int':
            changed = int(value)
        elif type_string == 'float':
            changed = float(value)
        elif type_string == 'bool':
            changed = bool(value)
        elif type_string == 'str':
            changed = str(value)
        else:
            self._output(f'_type_changer: {type_string} is not a valid type.', level='error')
            raise TypeError(f'Invalid type string: {type_string}')
        return changed

    def _is_sublist(self, sublist: Optional[list], mainlist: Optional[list]) -> bool:
        """
        Checks if sublist is a sublist of mainlist using set operations.
        Should work if sublist and mainlist can be converted to sets.

        Parameters
        ----------
        sublist : list or tuple
            The list or tuple to check if it is a sublist.
        mainlist : list or tuple
            The list or tuple to check against.

        Returns
        -------
        bool
            True if sublist is a sublist of mainlist, False otherwise.
        """
        try:
            return set(sublist).issubset(set(mainlist))
        except TypeError as e:
            self._output(f'_is_sublist: Error occurred - {e}', level='error')
            return False

    def close_logger(self):
        """
        Closes the logger.
        """
        if hasattr(self, 'logger'):
            logger = getattr(self, 'logger')
            for handler in logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                    logger.removeHandler(handler)
        self._output(f'{self.__class__.__name__}: Logging to file stopped.')

#%% Some Useful Functions
# Sorting function for names like P2_1, P2_2,...
def sort_by_label(df: pd.DataFrame, column: str) -> None:
    
    num_groups = len(df[column][0].split('_'))

    df[[f'Part{i + 1}' for i in range(num_groups)]] = df[column].apply(
        lambda x: x.split('_')).apply(pd.Series)
    
    for i in range(num_groups):
        col_name = f'Part{i + 1}'
        df[col_name] = BaseFunction().safe_to_numeric(data=df[col_name])

    df.sort_values(by=[f'Part{i + 1}' for i in range(num_groups)], inplace=True)
    df.drop(columns=[f'Part{i + 1}' for i in range(num_groups)], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return

# Extrapolation function
def extrapolate_points(dataframe:pd.DataFrame,
                       name_column: str,
                       min_point: str, 
                       max_point: str,
                       distance: float,
                       new_points: list[str]) -> pd.DataFrame:
    
    df = dataframe.copy()
    start_point = df[df[name_column] == min_point][['x', 'y', 'z']].to_numpy()
    end_point = df[df[name_column] == max_point][['x', 'y', 'z']].to_numpy()
    if len(start_point) == 0 or len(end_point) == 0:
        raise KeyError('min_point or max_point not found')
    vector = end_point - start_point
    unit_vector = vector / np.linalg.norm(vector)

    add_distance = distance

    for name in new_points:
        new_point = (end_point + add_distance * unit_vector).tolist()[0]
        new_df = pd.DataFrame([[name] + new_point], columns=df.columns)
        df = pd.concat([df, new_df], ignore_index=True)
        add_distance += distance

    return df

# Interpolation function
def interpolate_points(dataframe:pd.DataFrame,
                       name_column: str,
                       min_point: str, 
                       max_point: str,
                       new_points: list[str]) -> pd.DataFrame:
    
    df = dataframe.copy()
    start_point = df[df[name_column] == min_point][['x', 'y', 'z']].to_numpy()
    end_point = df[df[name_column] == max_point][['x', 'y', 'z']].to_numpy()
    vector = end_point - start_point
    unit_vector = vector / np.linalg.norm(vector)
    distance = np.linalg.norm(vector) / (len(new_points) + 1)

    add_distance = distance

    for name in new_points:
        new_point = (start_point + add_distance * unit_vector).tolist()[0]
        new_df = pd.DataFrame([[name] + new_point], columns=df.columns)
        df = pd.concat([df, new_df], ignore_index=True)
        add_distance += distance

    return df

# Project onto line
def project_to_1D(df:pd.DataFrame, remove_overhang: bool = False) -> None:
    all_points = df[['x', 'y']].to_numpy()
    start_point = all_points[0]
    end_point = all_points[-1]

    vector = end_point - start_point
    unit_vector = vector / np.linalg.norm(vector)

    x_coords = np.dot(all_points-start_point, unit_vector)
    if remove_overhang:
        x_coords = pd.Series(x_coords)
        while (x_coords.diff() < 0).sum() > 0:
            diff_mask = x_coords.diff() < 0
            for idx in x_coords[diff_mask].index:
                x_coords[idx] = x_coords[idx-1]*1.0001
        x_coords = x_coords.to_numpy()
    y_coords = np.zeros_like(x_coords)
    proj_points = np.column_stack([x_coords, y_coords])
    df[['x', 'y']] = proj_points
    return