# -*- coding: utf-8 -*-
"""
Created on Sat Nov 02 17:15:02 2024

Class for creating folder structures.

@author: peter balogh @ TU Wien, Research Unit Geophysics
"""

#%% Imports
import json
from pathlib import Path
import shutil
from typing import Union, Optional
from .utils import BaseFunction

# %% GPfolder class

class FolderHandler(BaseFunction):
    """
    This class creates a directory structure based on a template.
    """
    def __init__(self, root_path: Union[Path, str], template: Union[str, dict], save_log: bool = False) -> None:
        """
        This method initializes the GPfolder class.
        It creates the main folders of the directory structure.

        Parameters
        ----------
        root_path : Path or str
            The root path of the directory structure.
        template : str or dict
            The template for the directory structure.
            If a string is given, it must be the name of a template in the "templates/dir_structure" directory.
            If a dictionary is given, it must be the template itself.
        save_log: bool, optional
            If the log should be saved to a log file. The default is False.
        """
        self.root_path = root_path
        self._load_templates()
        self.__dir_template = self._choose_template(template)
        self._create_folders(save_log=save_log)

        self.logger = self._setup_logger(log_path=self.log_path)

    @property
    def root_path(self) -> Path:
        """
        Returns the root path of the directory structure.

        Returns
        -------
        Path
            The root path of the directory structure.
        """
        return self.__root_path
    
    @root_path.setter
    def root_path(self, path: Union[str, Path]):
        """
        Setter of root_path property
        """
        self.__root_path = Path(path)

    @property
    def dir_template(self) -> dict:
        """
        Returns the folder structure of the directory.

        Returns
        -------
        dict
            The folder structure of the directory.
        """
        return self.__dir_template

    @property
    def log_path(self) -> Optional[Path]:
        """
        Returns the path of the log file.

        Returns
        -------
        Path
            The path of the log file.
        """
        return self.__log_path
    
    @log_path.setter
    def log_path(self, filepath: Union[str, Path, None]):
        """
        Setter for log_path property.
        """
        if filepath is None:
            self.__log_path = None
        else:
            self.__log_path = Path(filepath)

    @property
    def folder_structure(self):
        """
        Returns the folders of the directory structure.

        Returns
        -------
        dict
            The folders of the directory structure.
        """
        return self.__folder_structure

    def _load_templates(self):
        """
        Loads the templates for the directory structure.

        Returns
        -------
        None
        """
        template_folder = Path(__file__).parent / 'templates'

        if template_folder.exists() and template_folder.is_dir():
            templates_dir = [(file.stem, json.loads(file.read_text())) for file in
                             template_folder.iterdir() if file.is_file() and file.suffix == '.json']
            if len(templates_dir) == 0:
                raise FileNotFoundError('No templates for directory structure found.')
            self.__templates_dir = {temp.get('name', name): temp for name, temp in
                                    templates_dir if temp.get('type') == 'dir_structure'}
        else:
            raise FileNotFoundError('Make sure "templates/dir_structure" directory exists.')

    def _choose_template(self, template: Union[str, dict]):
        """
        Chooses the template for the directory structure.

        Returns
        -------
        dict
            The template for the directory structure.
        """
        if isinstance(template, dict):
            return template
        elif isinstance(template, str):
            if template in self.__templates_dir.keys():
                return self.__templates_dir[template]
            else:
                raise KeyError(f'Template "{template}" not found in the templates directory.')
        else:
            raise TypeError('Template must be a string or a dictionary.')

    def _create_folders(self, save_log: bool):
        """
        Creates the main folders of the directory structure.

        Parameters
        ----------
        save_log: bool
            If the log should be saved in a log file.

        Returns
        -------
        dict
            The main folders of the directory structure
        """
        basedir = self.root_path            
        main_structure = self.dir_template.get('main_template')
        if main_structure is None:
            raise KeyError('create_folders: Main template was not found.')

        main_structure = {key: basedir / value for key, value in main_structure.items()}

        if main_structure.get('log') is not None:
            log_path = main_structure.pop('log')
            log_path.parent.mkdir(parents=True, exist_ok=True)
            if not save_log:
                log_path = None
        else:
            log_path = None
        self.log_path = log_path

        for value in main_structure.values():
            value.mkdir(parents=True, exist_ok=True)

        self.__folder_structure = main_structure

    def create_subdir(self, directory: str) -> None:
        """
        Creates a subdirectory of the directory structure.

        Parameters
        ----------
        directory : str
            The name of the directory to be created.

        Returns
        -------
        None
        """
        sub_structure = self.dir_template.get('sub_template')
        if sub_structure is None:
            self.logger.error('create_subdirs: Sub template was not found.')
            raise KeyError('create_subdirs: Sub template was not found.')

        if sub_structure.get('root') is not None:
            sub_root_name = sub_structure.pop('root')
            sub_root = self.folder_structure.get(sub_root_name)
        else:
            sub_root = None

        sub_root = self.root_path if sub_root is None else sub_root

        sub_structure = {key: sub_root / directory / value for key, value in sub_structure.items()}

        for value in sub_structure.values():
            value.mkdir(parents=True, exist_ok=True)

        self.__folder_structure[directory] = sub_structure

    def move_files(self, from_path: Union[Path, str], to_path: Union[Path, str]) -> None:
        """
        Moves files from one directory to another.

        Parameters
        ----------
        from_path : Path or str
            The path of the files to be moved.
        to_path : Path or str
            The path to which the files will be moved.

        Returns
        -------
        None
        """
        from_path = Path(from_path)
        to_path = Path(to_path)

        if not to_path.is_dir():
            self.logger.error('move_files: To path was not a directory.')
            raise ValueError('move_files: Invalid to_path. Must be a directory.')
        if not to_path.exists():
            to_path.mkdir(parents=True, exist_ok=True) #create the directory if it does not exist

        if from_path.exists():
            if from_path.is_file():
                from_path.rename(to_path / from_path.name)
            elif from_path.is_dir():
                for file in from_path.iterdir():
                    file.rename(to_path / file.name)
                from_path.rmdir()
            else:
                self.logger.error('move_files: From path was neither a file nor a directory.')
                raise ValueError('move_files: Invalid from_path. Must be a file or directory.')
        else:
            self.logger.warning('move_files: No files were found at the from_path.')

    def copy_files(self, from_path: Union[Path, str], to_path: Union[Path, str]) -> None:
        """
        Copies files from one directory to another.

        Parameters
        ----------
        from_path : Path or str
            The path of the files to be copied.
        to_path : Path or str
            The path to which the files will be copied.

        Returns
        -------
        None
        """
        from_path = Path(from_path)
        to_path = Path(to_path)

        if not to_path.is_dir():
            self.logger.error('move_files: To path was not a directory.')
            raise ValueError('move_files: Invalid to_path. Must be a directory.')
        if not to_path.exists():
            to_path.mkdir(parents=True, exist_ok=True)

        if from_path.exists():
            if from_path.is_file():
                shutil.copy(from_path, to_path / from_path.name)
            elif from_path.is_dir():
                for file in from_path.iterdir():
                    shutil.copy(file, to_path / file.name)
            else:
                self.logger.error('copy_files: From path was neither a file nor a directory.')
                raise ValueError('copy_files: Invalid from_path. Must be a file or directory.')
        else:
            self.logger.warning('copy_files: No files were found at the from_path.')

    def close(self):
        """
        Closes the logger and resets the class variables.

        Returns
        -------
        None
        """
        self.close_logger()
