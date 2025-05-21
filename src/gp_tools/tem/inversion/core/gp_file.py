# -*- coding: utf-8 -*-
"""
Created on Fri Nov 01 10:38:41 2024

@author: peter
"""

#%% Import modules

import re
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, TextIO, Union

from TEM_tools.core.base import BaseFunction

# %% GPfile class

class GPfile(BaseFunction):
    """
    This class is designed to read and write files containing geophysical data.

    It can read and write files with the following structures:

    - dict: A dictionary block in the file.
    - dataframe: A dataframe block in the file. (a 'simple' table: DAS-files are not implemented yet)
    - line: A line block in the file.

    The class is designed to be used with templates for the file structure.
    Up to now (03.11.2024) following templates are available:

    - syscal: For files given by Syscal devices. Suffix is '.dat'.
    - ohm: For ERT-files, they contain coordinates and the data. Suffix is '.ohm'.
    - tem: For files given by the TEMfast device. Suffix is '.tem'.
    - tim: TEM Inversion Model files, a custom filetype for inversion results. Suffix is '.tim'.

    All templates are stored in the 'templates/file_type' directory.
    """
    def __init__(self, name: str = None, data: dict = None, log_path: Optional[str] = None, verbose: bool = True):
        """
        Initializes the GPfile class.
        Automatically loads the templates for the file structure.

        Parameters
        ----------
        name : str
            Name of the file.
        data : dict
            Data dictionary that holds the data of the file.
        log_path : str
            Path to the log file.
        verbose : bool
            If True, the logger will print to console.
        """
        self._name = name
        self._templates_file = None
        self._load_templates()
        self._suffix = None
        self._data_path = None
        self._data = data
        self.verbose = verbose
        self.logger = self._setup_logger(log_path=log_path)

    def name(self) -> str:
        """
        Returns the name of the file.

        Returns
        -------
        str
            Name of the file.
        """
        return self._name

    def suffix(self) -> str:
        """
        Returns the suffix of the file.

        Returns
        -------
        str
            Suffix of the file.
        """
        return self._suffix

    def data(self) -> dict:
        """
        Returns the data dictionary of the file.

        Returns
        -------
        dict
            Data dictionary of the file.
        """
        return self._data

    def data_path(self) -> Path:
        """
        Returns the path of the data.

        Returns
        -------
        Path
            Path of the data.
        """
        return self._data_path

    def _load_templates(self) -> None:
        """
        Loads the templates for the file structure.

        Returns
        -------
        None
        """
        template_folder = Path(__file__).parents[1] / 'templates' / 'file_type'

        if template_folder.exists() and template_folder.is_dir():
            templates_file = [(file.stem, yaml.safe_load(file.read_text())) for file in
                              template_folder.iterdir() if file.is_file() and file.suffix == '.yml']
            if len(templates_file) == 0:
                raise FileNotFoundError('No templates for file structure found.')
            self._templates_file = {temp.get('config', {}).get('name', name): temp for name, temp in
                                    templates_file}
        else:
            raise FileNotFoundError('Make sure "templates/file_type" directory exists.')

    def _check_extension(self, file_path: Union[str, Path]) -> str:
        """
        Checks the extension of the file.

        Parameters
        ----------
        file_path : str or Path
            Path to the file.

        Returns
        -------
        str
            Extension of the file.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            if self.verbose:
                self.logger.error(f'_check_extension: {file_path} does not exist.')
            raise FileNotFoundError(f'_check_extension: {file_path} does not exist.')
        else:
            if file_path.is_file():
                extensions = file_path.suffix
            elif file_path.is_dir():
                extensions = {file.suffix for file in file_path.iterdir() if file.is_file()}
                if len(extensions) != 1:
                    if self.verbose:
                        self.logger.error(f'_load_filepaths: Files have different extensions: {extensions}')
                    raise ValueError('All files must have the same extension.')
                else:
                    extensions = next(iter(extensions))
            else:
                if self.verbose:
                    self.logger.error(f'_check_extension: {file_path} is neither a file nor a directory.')
                raise ValueError(f'_check_extension: {file_path} is neither a file nor a directory.')
        return extensions

    def _choose_file_type(self, file_path: Union[str, Path], chosen_template: str = None) -> dict:
        """
        Chooses the file type based on the file extension.

        Parameters
        ----------
        file_path : str or Path
            Path to the file.
        chosen_template : str
            Name of the template to choose.

        Returns
        -------
        dict
            File type dictionary
        """
        suffix = self._check_extension(file_path=file_path)

        if chosen_template is not None:
            file_type = self._templates_file.get(chosen_template)
            if file_type is None:
                if self.verbose:
                    self.logger.error('choose_file_template: Chosen file type was not found')
                raise FileNotFoundError('choose_file_template: Chosen file type was not found')
            if file_type.get('config').get('file_suffix') != suffix:
                if self.verbose:
                    self.logger.error('choose_file_template: File suffix does not match')
                raise ValueError('choose_file_template: File suffix does not match')
            else:
                return file_type
        else:
            parsing_dict = {value.get('config').get('file_suffix'): key for key, value in self._templates_file.items()}
            type_name = parsing_dict.get(suffix)
            file_type = self._templates_file.get(type_name)
            if file_type is None:
                if self.verbose:
                    self.logger.error(f'choose_file_template: File type was not found.')
                raise FileNotFoundError(f'choose_file_template: File type was not found.')
            return file_type

    @staticmethod
    def _find_measurement_blocks(template_dict: dict, lines: list) -> list:
        """
        Finds the indices of the measurement blocks in the file.

        Parameters
        ----------
        template_dict : dict
            Template dictionary for the file.
        lines : list
            List of lines in the file.

        Returns
        -------
        list
            List of indices of the measurement blocks.
        """
        block_indices = []
        keyword = template_dict.get('config', {}).get('block_start_pattern')
        buffer = template_dict.get('config', {}).get('start_buffer', 0)
        if keyword is None:
            block_indices.append(0)
        else:
            key_lower = keyword.lower()
            for i, line in enumerate(lines):
                if key_lower in line.lower():
                    block_indices.append(i + buffer)
        return block_indices

    @staticmethod
    def _get_block(template_dict: dict, lines: list, start_index: int) -> Optional[list]:
        """
        Gets the block from the file.

        Parameters
        ----------
        template_dict : dict
            Template dictionary for the block.
        lines : list
            List of lines in the file.
        start_index : int
            Starting index of the block.

        Returns
        -------
        list
            Block of lines from the file.
        """
        block_start = start_index
        start_with = template_dict.get('start_with')
        if start_with is not None:
            start_lower = start_with.lower()
            for i, line in enumerate(lines):
                if i >= start_index:
                    block_start = i
                    if start_lower in line.lower():
                        break

        start_after = template_dict.get('start_after', 0)
        block_start += start_after

        block_end = None
        end_found = False
        end_with = template_dict.get('end_with')
        for i, line in enumerate(lines):
            if i >= block_start:
                block_end = i
                if end_with is not None:
                    end_lower = end_with.lower()
                    if end_lower in line.lower():
                        end_found = True
                        break

        end_after = template_dict.get('end_after', 0)
        if end_found:
            block_end += end_after
        if block_end is None:
            block_lines = lines[block_start:]
        else:
            block_lines = lines[block_start:block_end + 1]
        return block_lines

    def _parse_dict(self, template_dict: dict, lines: list, start_index: int) -> dict:
        """
        Parses a dictionary block from the file.

        Parameters
        ----------
        template_dict : dict
            Template dictionary for the block.
        lines : list
            List of lines in the file.
        start_index : int
            Starting index of the block.

        Returns
        -------
        dict
            Parsed dictionary block.
        """
        if not template_dict.get('type') == 'dict':
            if self.verbose:
                self.logger.error('parse_dict: Invalid type for parse_dict.')
            raise TypeError('parse_dict: Invalid type for parse_dict.')
        dictionary = {}

        raw_block_lines = self._get_block(template_dict=template_dict, lines=lines, start_index=start_index)
        template_lines = template_dict.get('lines', [])
        if not template_lines:
            if self.verbose:
                self.logger.error('parse_dict: Template is missing "lines". It is needed for parsing.')
            raise KeyError('parse_dict: Template is missing "lines". It is needed for parsing.')
        block_lines = raw_block_lines[:len(template_lines)]

        for line in block_lines:
            for entry in template_lines:
                match = re.search(entry.get('pattern'), line)
                if match:
                    for i, key in enumerate(entry.get('key', [])):
                        dictionary[key] = match.group(i + 1)

        for key, value in dictionary.items():
            dictionary[key] = self.safe_to_numeric(value)

        return dictionary

    def _parse_dataframe(self, template_dict: dict, lines: list, start_index: int) -> pd.DataFrame:
        """
        Parses a dataframe block from the file.

        Parameters
        ----------
        template_dict : dict
            Template dictionary for the block.
        lines : list
            List of lines in the file.
        start_index : int
            Starting index of the block

        Returns
        -------
        pd.DataFrame
            Parsed dataframe block.
        """
        if not template_dict.get('type') == 'dataframe':
            if self.verbose:
                self.logger.error('parse_dataframe: Invalid type for parse_dataframe.')
            raise TypeError('parse_dataframe: Invalid type for parse_dataframe.')

        block_lines = self._get_block(template_dict=template_dict, lines=lines, start_index=start_index)
        raw_columns = list(block_lines)[0]
        raw_measurements = block_lines[1:]
        delimiter = template_dict.get('delimiter', '\t').encode().decode('unicode_escape')
        column_delimiter = template_dict.get('column_delimiter', delimiter).encode().decode('unicode_escape')
        column_prefix = template_dict.get('column_prefix', '')
        columns = raw_columns.replace(column_prefix, '').strip().split(column_delimiter)
        columns = [col.strip() for col in columns]
        measurements = [line.strip().split(delimiter) for line in raw_measurements]
        measurements = [[m.strip() for m in line] for line in measurements]

        df = pd.DataFrame(measurements, columns=columns)
        df.replace('nan', np.nan, inplace=True)
        df = self.safe_to_numeric(df)
        return df

    def _parse_line(self, template_dict: dict, lines: list, start_index: int) -> Union[str, list]:
        """
        Parses a line block from the file.

        Parameters
        ----------
        template_dict : dict
            Template dictionary for the block.
        lines : list
            List of lines in the file.
        start_index : int
            Starting index of the block

        Returns
        -------
        str or list
            Parsed line block.
        """
        if not template_dict.get('type', None) == 'line':
            if self.verbose:
                self.logger.error('parse_line: Invalid type for parse_line.')
            raise TypeError('parse_line: Invalid type for parse_line.')

        block_lines = self._get_block(template_dict=template_dict, lines=lines, start_index=start_index)

        line_number = template_dict.get('lines', 1)
        subtype = template_dict.get('subtype')
        if line_number == 1:
            line = list(block_lines)[0]
            if subtype:
                line = self._type_changer(subtype, line)
            return line
        else:
            line_block = block_lines[:line_number]
            if subtype:
                line_block = [self._type_changer(subtype, x) for x in line_block]
            return line_block

    def _find_parsing_type(self, template_dict: dict, lines: list, start_index: int) -> Union[dict, pd.DataFrame, str, list]:
        """
        Finds the parsing type of the block and applies it.

        Parameters
        ----------
        template_dict : dict
            Template dictionary for the block.
        lines : list
            List of lines in the file.
        start_index : int
            Starting index of the block.

        Returns
        -------
        dict, pd.DataFrame, str or list
            Parsed block
        """
        template_type = template_dict.get('type', None)
        if template_type is None:
            if self.verbose:
                self.logger.error('find_parsing_type: Template is missing "type". It is needed for parsing.')
            raise KeyError('find_parsing_type: Template is missing "type". It is needed for parsing.')
        if template_type == 'dict':
            return self._parse_dict(template_dict, lines, start_index)
        elif template_type == 'dataframe':
            return self._parse_dataframe(template_dict, lines, start_index)
        elif template_type == 'line':
            return self._parse_line(template_dict, lines, start_index)
        else:
            if self.verbose:
                self.logger.error('find_parsing_type: Template type was not recognized.')
            raise KeyError('find_parsing_type: Template type was not recognized.')

    def read(self, file_path: Union[str, Path], file_type: str = None, verbose: bool = None) -> Optional[dict]:
        """
        Reads the file and parses the data.
        If no file type is chosen, the file type is chosen based on the file extension.

        Parameters
        ----------
        file_path : str or Path
            Path to the file.
        file_type : str
            Name of the template to choose.
        verbose : bool
            If True, the logger will print to console. If None, the class attribute is used.

        Returns
        -------
        dict
            Parsed data dictionary.
        """
        file_path = Path(file_path)
        if verbose is not None:
            self.verbose = verbose
        if self.verbose:
            self.logger.info(f'read: Started reading {file_path.name}')

        type_file = self._choose_file_type(file_path=file_path, chosen_template=file_type)
        if type_file is None:
            if self.verbose:
                self.logger.error('read: No file type was found.')
            raise KeyError('read: No file type was found.')

        self._suffix = type_file.get('config', {}).get('file_suffix')
        end_zero = type_file.get('config', {}).get('end_zero', False)
        file_template = type_file.get('template')
        if file_template is None:
            if self.verbose:
                self.logger.error('read: Template is missing "template". It is needed for parsing.')
            raise KeyError('read: "template" not found in YAML-file.')

        name_line = file_template.get('name', {}).get('type', None)
        name_line = True if name_line == 'line' else False
        dict_blocks = [block for block, template in file_template.items() if template.get('type', None) == 'dict']

        name_block = False
        for block in dict_blocks:
            for line in file_template.get(block, {}).get('lines', []):
                for key in line.get('key', []):
                    if key == 'name':
                        name_block = block
                        break
                if name_block:
                    break

        if name_line and name_block:
            self.logger.error('read: Found two names in the template.')
            return None
        if name_line or name_block:
            name_exists = True
        else:
            name_exists = False

        measurements = None
        if file_path is not None:
            if file_path.is_dir():
                self.logger.error('read: Input is a directory. Must be a file.')
                return None
            else:
                file_name = file_path.stem
                with open(file_path, 'r') as file_:
                    lines = file_.readlines()

                if end_zero:
                    lines = lines[:-1]

                block_start_indices = self._find_measurement_blocks(template_dict=type_file, lines=lines)
                multiple_blocks = len(list(block_start_indices)) > 1

                if multiple_blocks and not name_exists:
                    if self.verbose:
                        self.logger.error('read: Multiple blocks found but no name found in template.')
                    raise KeyError('read: Multiple blocks found but no name found in template.')

                measurements = {}
                for start_index in block_start_indices:
                    block_dict = {}
                    for temp_name, temp in file_template.items():
                        block_dict[temp_name] = self._find_parsing_type(temp, lines, start_index)

                    block_name = block_dict.get('name', file_name)
                    measurement_name = block_dict.get(name_block, {}).get('name', block_name)

                    if measurement_name in measurements:
                        self.logger.warning(f'read: {measurement_name} already exists in measurements.')
                        measurement_name = f'{measurement_name}_{len(measurements)}'
                    measurements[measurement_name] = block_dict

        if self.verbose:
            self.logger.info(f'read: Added {file_path.name} to data.')
        self._data = measurements
        self._data_path = file_path
        return measurements

    @staticmethod
    def _write_dict(template_dict: dict, data: dict, file_: TextIO) -> None:
        """
        Writes a dictionary to the file.

        Parameters
        ----------
        template_dict : dict
            Template dictionary for the block.
        data : dict
            Data dictionary to be written.
        file_ : TextIO
            File to write the data to.

        Returns
        -------
        None
        """
        template_lines = template_dict.get('lines', [])
        for line in template_lines:
            line_keys = line.get('key', [])
            line_values = {key: data.get(key) for key in line_keys}
            write_line = line.get('output_format', '').encode().decode('unicode_escape').format(**line_values)
            file_.write(write_line + '\n')

    @staticmethod
    def _write_dataframe(template_dict: dict, data: pd.DataFrame, file_: TextIO) -> None:
        """
        Writes a dataframe to the file.

        Parameters
        ----------
        template_dict : dict
            Template dictionary for the block.
        data : pd.DataFrame
            Dataframe to be written.
        file_ : TextIO
            File to write the data to.

        Returns
        -------
        None
        """
        delimiter = template_dict.get('delimiter', '\t').encode().decode('unicode_escape')
        column_delimiter = template_dict.get('column_delimiter', delimiter).encode().decode('unicode_escape')
        column_prefix = template_dict.get('column_prefix', '')
        columns = column_delimiter.join(data.columns)
        file_.write(column_prefix + columns + '\n')

        for index, row in data.iterrows():
            file_.write(f'{delimiter.join(map(str, row))}\n')

    def _write_line(self, template_dict: dict, data: Union[str, list], file_: TextIO) -> None:
        """
        Writes a line to the file.

        Parameters
        ----------
        template_dict : dict
            Template dictionary for the block.
        data : str or list
            Data to be written.
        file_ : TextIO
            File to write the data to.

        Returns
        -------
        None
        """
        line_amount = template_dict.get('lines', 1)
        if line_amount == 1:
            file_.write(f'{data}\n')
        else:
            if not len(data) == line_amount:
                if self.verbose:
                    self.logger.error(
                    f'write_line: Input dictionary has {len(data)} lines while template allows {line_amount} lines.')
                raise ValueError(f'Input dictionary has {len(data)} lines ({line_amount} needed).')
            for line in data:
                file_.write(f'{line}\n')

    def _find_writing_type(self, template_dict: dict, data: Union[str, list, dict, pd.DataFrame], file_: TextIO) -> None:
        """
        Finds the writing type of the block and applies it.

        Parameters
        ----------
        template_dict : dict
            Template dictionary for the block.
        data : str, list, dict, or pd.DataFrame
            Data to be written.
        file_ : TextIO
            File to write the data to.

        Returns
        -------
        None
        """
        template_type = template_dict.get('type', None)
        if template_type is None:
            if self.verbose:
                self.logger.error('find_writing_type: Template is missing "type". It is needed for parsing.')
            raise KeyError('find_writing_type: Template is missing "type". It is needed for parsing.')
        if template_type == 'dict':
            self._write_dict(template_dict, data, file_)
        elif template_type == 'dataframe':
            self._write_dataframe(template_dict, data, file_)
        elif template_type == 'line':
            self._write_line(template_dict, data, file_)
        else:
            if self.verbose:
                self.logger.error('find_writing_type: Template type was not recognized.')
            raise KeyError('find_writing_type: Template type was not recognized.')

    def write(self, file_path: Union[str, Path], template: str = None, data: dict = None, verbose: bool = None) -> None:
        """
        Writes the data to the file.

        If no template is chosen, the template is chosen based on the file extension of the file_path provided.

        Uses data attribute if no data is provided.
        For this data must be provided during class initialization or by using the read function.

        Parameters
        ----------
        file_path : str or Path
            Path to the file.
        template : str
            Name of the template to choose.
        data : dict
            Data dictionary to be written.
        verbose : bool
            If True, the logger will print to console. If None, the class attribute is used.

        Returns
        -------
        None
        """
        file_path = Path(file_path)
        if verbose is not None:
            self.verbose = verbose

        if template is None:
            suffix = file_path.suffix
            if suffix is None:
                self.logger.error('write: No file type was found.')
                return

            parsing_dict = {value.get('config').get('file_suffix'): key for key, value in self._templates_file.items()}
            type_name = parsing_dict.get(suffix)
        else:
            type_name = template

        template_dict = self._templates_file.get(type_name)
        if template_dict is None:
            self.logger.error('write: No template was found.')
            return


        self._suffix = template_dict.get('config', {}).get('file_suffix')
        end_zero = template_dict.get('config', {}).get('end_zero', False)

        data_dict = data if data is not None else self._data
        if data_dict is None:
            self.logger.error('write: No data was found.')
            return

        if file_path.exists():
            if self.verbose:
                self.logger.warning('write: File already exists. Overwriting.')

        file_template = template_dict.get('template')

        if data_dict.get(next(iter(file_template))) is not None:
            data_list = [data_dict]
        elif data_dict.get(next(iter(data_dict)), {}).get(next(iter(file_template))) is not None:
            data_list = list(data_dict.values())
        else:
            self.logger.error('write: Data does not match the template.')
            return

        for dat in data_list:
            for key, value in file_template.items():
                if dat.get(key) is None:
                    self.logger.error(f'write: Data does not match the template. Missing key: {key}')
                    return

        with open(file_path, 'w') as file_:
            for dat in data_list:
                for key, value in file_template.items():
                    data_block = dat.get(key)
                    self._find_writing_type(value, data_block, file_)
            if end_zero:
                file_.write('0\n')
        if self.verbose:
            self.logger.info(f'write: Wrote data to {file_path.name}')

    def close(self):
        """
        Closes the logger and resets the class variables.

        Returns
        -------
        None
        """
        self.close_logger()
        self._data = None
        self._suffix = None
        self._name = None
        self._templates_file = None