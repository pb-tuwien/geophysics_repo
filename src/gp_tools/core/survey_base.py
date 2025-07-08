# -*- coding: utf-8 -*-
"""
Created on Sun Nov 03 17:31:37 2024

A base class for geophysical surveys.

@author: peter balogh @ TU Wien, Research Unit Geophysics
"""
#%% Import modules

from pathlib import Path
from typing import Optional, Union
import pandas as pd
from .utils import BaseFunction
from .coordinate_handler import CoordinateHandler
from .folder_handler import FolderHandler

#%% SurveyBase class

class SurveyBase(BaseFunction):
    """
    The SurveyBase class is a base class for geophysical surveys.
    It builds upon functionality implemented in the core module of the gp_package.
    It gives the user a structured way to handle survey coordinates.
    For handling survey data, the user should go to the daughter classes.
    """
    def __init__(self, project_dir: Union[Path, str], dir_structure: Union[str, dict], save_log: bool = False) -> None:
        """
        Initializes the SurveyBase class.

        Parameters
        ----------
        project_dir : Path or str
            The project directory.
        dir_structure : str or dict
            The directory structure of the project directory.
        """
        self.coordinates_raw = None
        self.coordinates_proc = None
        self.coordinates_grouped = None

        self.__project_dir = Path(project_dir)
        self._gp_folder = FolderHandler(root_path=self.project_dir, template=dir_structure, save_log=save_log)
        self.logger = self._setup_logger(log_path=self.log_path)
        self._gp_coords = CoordinateHandler(log_path=self.log_path)

    @property
    def project_dir(self) -> Path:
        """
        Returns the project directory.

        Returns
        -------
        Path
            The project directory.
        """
        return self.__project_dir

    @property
    def folder_structure(self) -> dict:
        """
        Returns the folder structure of the project directory.

        Returns
        -------
        dict
            The folder structure of the project directory.
        """
        return self._gp_folder.folder_structure

    @property
    def log_path(self) -> Optional[Path]:
        """
        Returns the log path of the project directory.

        Returns
        -------
        Path
            The log path of the project directory.
        """
        return self._gp_folder.log_path

    @property
    def coordinates_raw(self) -> Optional[pd.DataFrame]:
        """
        Returns the raw coordinates.

        Returns
        -------
        pd.DataFrame
            The raw coordinates.
        """
        return self.__coordinates_raw
    
    @coordinates_raw.setter
    def coordinates_raw(self, coordinates: Optional[pd.DataFrame]):
        """
        Setter of coordinates_raw property.
        """
        if coordinates is None or isinstance(coordinates, pd.DataFrame):
            self.__coordinates_raw = coordinates
        else:
            self.logger.error(f'Coordinates must be None or a DataFrame: {type(coordinates)} was given.')

    @property
    def coordinates_proc(self) -> Optional[pd.DataFrame]:
        """
        Returns the processed coordinates.

        Returns
        -------
        pd.DataFrame
            The processed coordinates.
        """
        return self.__coordinates_proc
    
    @coordinates_proc.setter
    def coordinates_proc(self, coordinates: Optional[pd.DataFrame]):
        """
        Setter of coordinates_proc property.
        """
        if coordinates is None or isinstance(coordinates, pd.DataFrame):
            self.__coordinates_proc = coordinates
        else:
            self.logger.error(f'Coordinates must be None or a DataFrame: {type(coordinates)} was given.')

    @property
    def coordinates_grouped(self) -> Optional[dict]:
        """
        Returns the grouped coordinates.

        Returns
        -------
        dict
            The grouped coordinates.
        """
        return self.__coordinates_grouped
    
    @coordinates_grouped.setter
    def coordinates_grouped(self, coordinates: Optional[dict]):
        """
        Setter of coordinates_grouped property.
        """
        if coordinates is None or isinstance(coordinates, dict):
            self.__coordinates_grouped = coordinates
        else:
            self.logger.error(f'Grouped coordinates must be None or a Dictionary: {type(coordinates)} was given.')

    def coords_read(self, coords: Union[Path, str] = None, sep: str = ','):
        """
        Reads the coordinates from a file.
        If no file is given, the function will look for raw and processed coordinate files in the directory structure.

        Parameters
        ----------
        coords : Path or str, optional
            The path to the coordinate file. The default is None.
        sep : str, optional
            The separator of the coordinate file. The default is ','.

        Returns
        -------
        None
        """
        raw_coords = self.folder_structure.get('coordinates_raw')
        proc_coords = self.folder_structure.get('coordinates_proc')

        if coords is None:
            if raw_coords is not None and raw_coords.exists():
                raw_paths = [path for path in raw_coords.iterdir() if path.is_file()]
                if len(raw_paths) > 1:
                    self.logger.warning('coords_read: Multiple raw coordinate files found. Using the first one.')
                if len(raw_paths) == 0:
                    self.logger.warning('coords_read: No raw coordinate files found in the directory structure.')
                else:
                    self._gp_coords.read(file_path=raw_paths[0], sep=sep)
                    self.coordinates_raw = self._gp_coords.coordinates

            if proc_coords is not None and proc_coords.exists():
                proc_paths = [path for path in proc_coords.iterdir() if path.is_file()]
                if len(proc_paths) > 1:
                    self.logger.warning('coords_read: Multiple processed coordinate files found. Using the first one.')
                if len(proc_paths) == 0:
                    self.logger.warning('coords_read: No processed coordinate files found in the directory structure.')
                else:
                    self._gp_coords.read(file_path=proc_paths[0], sep=sep)
                    self.coordinates_proc = self._gp_coords.coordinates
                    self.coordinates_grouped = self._gp_coords.extract_coords()

        else:
            coords = Path(coords)
            if coords.exists():
                file_name = coords.name
                new_coords = raw_coords / file_name
                self._gp_folder.copy_files(from_path=coords, to_path=raw_coords)
                self._gp_coords.read(file_path=new_coords, sep=sep)
                self.coordinates_raw = self._gp_coords.coordinates
            else:
                self.logger.warning('coords_read: No file found. Tries reading from the directory structure.')
                if coords is not None:
                    self.coords_read(coords=None)

    def coords_rename_columns(self, rename_dict: dict) -> None:
        """
        Renames the columns of the coordinates.
        (using the GPcoords class)

        Parameters
        ----------
        rename_dict : dict
            The dictionary containing the old and new column names.

        Returns
        -------
        None
        """
        self._gp_coords.rename_columns(rename_dict)

    def coords_rename_points(self, rename_dict: dict) -> None:
        """
        Renames the points of the coordinates.
        (using the GPcoords class)

        Parameters
        ----------
        rename_dict : dict
            The dictionary containing the old and new point names. (For more information see the GPcoords class)

        Returns
        -------
        None
        """
        self._gp_coords.rename_points(rename_dict)

    def coords_sort_points(self, ascending: bool = True) -> None:
        """
        Sorts the points of the coordinates.
        (using the GPcoords class)

        Parameters
        ----------
        ascending : bool, optional
            The sorting order. The default is True.

        Returns
        -------
        None
        """
        self._gp_coords.sort_points(ascending=ascending)

    def coords_reproject(self, reproj_key: str = 'wgs84_utm33n', correct_height: bool = True) -> None:
        """
        Transforms the coordinates.
        (using the GPcoords class)

        Parameters
        ----------
        reproj_key : str, optional
            The key of the projection. The default is 'wgs84_utm33n'.
        correct_height : bool, optional
            If the coordinates should be corrected for antenna height. The default is True.

        Returns
        -------
        None
        """
        self._gp_coords.reproject(reproj_key=reproj_key, correct_height=correct_height)

    def coords_custom_reproject(self, from_key: str, to_key: str, correct_height: bool = True) -> None:
        """
        A more open transformation of the coordinates. (keys should work with pyproj.Transformer)
        (using the GPcoords class)

        Parameters
        ----------
        from_key : str
            The key of the projection to transform from.
        to_key : str
            The key of the projection to transform to.
        correct_height : bool, optional
            If the coordinates should be corrected for antenna height. The default is True.

        Returns
        -------
        None
        """
        self._gp_coords.custom_reprojection(from_crs=from_key, to_crs=to_key, correct_height=correct_height)

    def coords_interpolate_points(self) -> None:
        """
        Interpolates the points of the coordinates.
        (using the GPcoords class)

        Returns
        -------
        None
        """
        self._gp_coords.interpolate_points()

    def coords_extract_save(self) -> None:
        """
        Groups coordinates and saves it to self.coordinates_grouped.
        If a processed coordinates folder is found in the directory structure, the processed coordinates will be saved there.
        (using the GPcoords class)

        Returns
        -------
        None
        """
        self.coordinates_grouped = self._gp_coords.extract_coords()

        proc_coords = self.folder_structure.get('coordinates_proc')
        if proc_coords is None:
            self.logger.warning('coords_extract_save: No processed coordinates folder found in the directory structure.')
        else:
            coords_file = self._gp_coords.coordinate_path
            if coords_file.parent != proc_coords:
                proc_file = proc_coords / f'{coords_file.stem}_proc{coords_file.suffix}'
                self._gp_coords.write(file_path=proc_file)
                self.logger.info(f'coords_extract_save: Processed coordinates saved to {proc_file}')

    def coords_create_dummy(self, number_electrodes: int, separation: Union[int, float]) -> pd.DataFrame:
        """
        Creates dummy coordinates DataFrame.
        (using the GPcoords class)

        Parameters
        ----------
        number_electrodes : int
            The number of electrodes.
        separation : int or float
            The separation between the electrodes.

        Returns
        -------
        pd.DataFrame
            The dummy coordinates DataFrame.
        """
        return self._gp_coords.create_dummy(elec_num=number_electrodes, elec_sep=separation)

    def close(self):
        """
        Closes the SurveyBase class.
        Close the logger and all used GP classes.
        Resets class variables.
        Returns
        -------

        """
        self.close_logger()
        self._gp_folder.close()
        self._gp_coords.close()

        self.coordinates_raw = None
        self.coordinates_proc = None
        self.coordinates_grouped = None
