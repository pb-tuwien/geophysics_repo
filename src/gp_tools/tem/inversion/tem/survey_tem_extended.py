# -*- coding: utf-8 -*-
"""
Created on Sun Nov 03 19:31:36 2024

@author: peter & jakob
"""

#%% Import modules

from scipy.interpolate import griddata
from scipy.optimize import curve_fit #todo: schauen ob man das sinnvoll einbauen kann?? @jakob
from pathlib import Path
import warnings
from typing import Any, Union
import numpy as np
import pandas as pd
# from datetime import datetime
import matplotlib.pyplot as plt
import TEM_tools.tem.survey_tem as st

warnings.filterwarnings('ignore')

#%% ExtendedSurveyTEM class

class ExtendedSurveyTEM(st.SurveyTEM):
    def __init__(self, project_directory: Union[Path, str], dir_template: str = 'tem_default') -> None:
        super().__init__(project_directory=project_directory, dir_template=dir_template)

    def choose_from_csv(self, filepath: Path, chosen_points: tuple = (), line_name: str = '') -> list[Any]:
        filepath = Path(filepath)
        target_folder = self._folder_structure.get('coordinates_choose')  #
        if target_folder is None:
            raise ValueError('No target folder for coordinates found.')
        new_file = target_folder / filepath.name  # saving new filepath based on current filepath and folder from class structure
        self._gp_folder.move_files(from_path=filepath, to_path=target_folder)

        # todo: was passiert ab hier?
        tem_line_table = pd.read_csv(new_file, sep=';')  # reading the csv file from the tem line
        if not chosen_points:
            chosen_points = list(tem_line_table[
                                     'Name'])  # if points are given as chosen, they are used, else the points from the line are used
        cur_all = self.chosen_data.get('all', [])  # what has previously been chosen is loaded
        cur_all.extend(chosen_points)  # the newly chosen points are added to the previously chosen points
        self.chosen_data['all'] = cur_all  # updating the chosen list of all points
        self.chosen_data[line_name] = chosen_points  # updating the chosen list of points for the current line
        # todo: log entry for chosen data - ja, hab log noch nicht verstanden
        return chosen_points

    # def inversion_chosen(self, from_csv:Path=None,
    #                      chosen_points:tuple=(),
    #                      subset:list=None, unit='rhoa',
    #                      lam=600,
    #                      layer_type='linear',
    #                      layers=4.5,
    #                      max_depth=None,
    #                      filter_times=(7, 700)):
    #
    #     if from_csv is not None:
    #         inversion_list = self.choose_from_csv(from_csv, chosen_points)
    #     else:
    #         inversion_list = self.chosen_data.get('all')
    #
    #     if subset is not None:
    #         self.chosen_data['subset'] = subset
    #         inversion_list = subset
    #     #todo: log entry for inverted data
    #     self.plot_inversion(subset=inversion_list, lam=lam, layer_type=layer_type, unit=unit, layers=layers, max_depth=max_depth, filter_times=filter_times)

    @staticmethod
    def _find_inflection_index(xvalues: Union[np.ndarray, list], yvalues: Union[np.ndarray, list]) -> list:
        x_array = np.array(xvalues)
        y_array = np.array(yvalues)
        first_diff = np.diff(y_array) / np.diff(x_array)
        second_diff = np.diff(first_diff) / np.diff(x_array[:-1])
        inflection_index = np.argmin(np.abs(second_diff[:-1] + second_diff[1:]))
        # inflection_value_index = xvalues[inflection_indices]
        return [inflection_index, first_diff, second_diff]

    def _find_inflection_point(self, xvalues: Union[np.ndarray, list], yvalues: Union[np.ndarray, list]) -> tuple:
        inflection_index, first_diff, second_diff = self._find_inflection_index(xvalues, yvalues)
        start_x = xvalues[inflection_index]
        start_y = second_diff[inflection_index]
        end_x = xvalues[inflection_index + 1]
        end_y = second_diff[inflection_index + 1]
        y = abs(start_y) + abs(end_y)
        x = end_x - start_x  # x goes along the x-Axis, here only the difference between start and end is needed
        inflection_x = start_y / y * x + start_x
        return inflection_x, 0

    def analyse_inversion(self, sounding: str, layers, max_depth: float, plot_id: str = None,
                          test_range: tuple = (10, 10000, 30), layer_type: str = 'linear', filter_times=(7, 700)):
        # todo: make plotting more efficient and neat ... (@jakob)
        # computing relevant values for plotting
        test_tuple = test_range if len(test_range) == 3 else (test_range[0], test_range[1], 30)
        lambda_values = np.logspace(np.log10(test_tuple[0]), np.log10(test_tuple[1]), test_tuple[2])
        roughness_values = []
        rms_values = []
        for lam in lambda_values:
            self.data_inversion(subset=[sounding], lam=lam, layer_type=layer_type, layers=layers, verbose=False,
                                max_depth=max_depth, filter_times=filter_times, noise_floor=0.025, start_model=None)

            inversion_dict = self._data_inverted.get(sounding, {}).get(f'{lam}_{filter_times[0]}_{filter_times[1]}')
            rms_values.append(inversion_dict.get('metadata').get('absrms'))  # todo: change labels
            roughness_values.append(inversion_dict.get('metadata').get('phi_model'))

        # computing inflection
        inflection_lambda_index, first_diff_lam, second_diff_lam = self._find_inflection_index(lambda_values,
                                                                                               rms_values)
        inflection_roughness_index, first_diff_phi, second_diff_phi = self._find_inflection_index(roughness_values,
                                                                                                  rms_values)
        inflection_lam_phi_index, first_diff_lam_phi, second_diff_lam_phi = self._find_inflection_index(lambda_values,
                                                                                                        roughness_values)

        def plot_analysis(sounding: str, plot_id: str, lambda_values: np.array, roughness_values: list,
                          rms_values: list, test_range: tuple = (10, 10000, 30), filter_times=(7, 700)):
            fig, axs = plt.subplots(1, 3, figsize=(24, 8))
            ax1, ax2, ax3 = axs[0], axs[1], axs[2]
            fig.suptitle(
                f'Analysis of Inversion Results\n Lambda varied between {test_range[0]} and {test_range[1]} with a spacing of {test_range[2]} for sounding {sounding}',
                fontsize=14)
            ax1.set_title(f'Data plotting, Inflection at {lambda_values[inflection_lambda_index]}')
            ax1.set_ylabel('relRMS (%)')
            ax1.set_xlabel('Lambda')
            ax1.scatter(lambda_values, rms_values)
            ax1.axvline(x=lambda_values[inflection_lambda_index], color='r', linestyle='--')
            ax1.set_ylim(0, max(rms_values) * 1.1)

            ax2.set_title('First Derivative')
            ax2.set_ylabel('first derivative')
            ax2.set_xlabel('Lambda')
            ax2.scatter(lambda_values[:-1], first_diff_lam)
            ax2.axvline(x=lambda_values[inflection_lambda_index], color='r', linestyle='--')
            ax2.axvline(x=self._find_inflection_point(lambda_values, rms_values)[0], color='b', linestyle='--')
            ax2.axhline(y=first_diff_lam[inflection_lambda_index], color='r', linestyle='--')
            ax2.plot(lambda_values[inflection_lambda_index], first_diff_lam[inflection_lambda_index], marker='o',
                     color='g')

            ax3.set_title('Second Derivative')
            ax3.set_ylabel('second derivative')
            ax3.set_xlabel('Lambda')
            ax3.axhline(y=0, color='r', linestyle='-')
            ax3.scatter(lambda_values[:-2], second_diff_lam)
            ax3.axvline(x=lambda_values[inflection_lambda_index], color='r', linestyle='--')
            ax3.axvline(x=self._find_inflection_point(lambda_values, rms_values)[0], color='b', linestyle='--')

            fig.tight_layout()
            fig.show()
            fig.savefig(self._folder_structure.get(
                'data_inversion_analysis') / f'{sounding}_{filter_times[0]}_{filter_times[1]}_lambda_rrms_analysis.png')

            fig2, axs2 = plt.subplots(1, 3, figsize=(24, 8))
            ax4, ax5, ax6 = axs2[0], axs2[1], axs2[2]
            fig2.suptitle(
                f'Analysis of Inversion Results\nLambda varied between {test_range[0]} and {test_range[1]} with a spacing of {test_range[2]} for sounding {sounding}, roughness and relRMS plotted',
                fontsize=14)
            ax4.set_title('Data plotting')
            ax4.set_xlabel('roughness')
            ax4.set_ylabel('relRMS (%)')
            ax4.scatter(roughness_values, rms_values)
            ax4.axvline(x=roughness_values[inflection_lambda_index], color='r', linestyle='-')
            ax4.axvline(x=roughness_values[inflection_roughness_index], color='g', linestyle='--')
            ax4.axvline(x=self._find_inflection_point(roughness_values, rms_values)[0], color='b', linestyle='-')
            ax4.set_ylim(0, max(rms_values) * 1.1)

            ax5.set_title('Data plotting')
            ax5.set_xlabel('roughness')
            ax5.set_ylabel('first diff')
            ax5.scatter(roughness_values[:-1], first_diff_phi)
            ax5.axvline(x=roughness_values[inflection_lambda_index], color='r', linestyle='-')
            ax5.axvline(x=roughness_values[inflection_roughness_index], color='g', linestyle='--')
            ax5.axvline(x=self._find_inflection_point(roughness_values, rms_values)[0], color='b', linestyle='-')

            ax6.set_title('Data plotting')
            ax6.set_xlabel('roughness')
            ax6.set_ylabel('second diff')
            ax6.scatter(roughness_values[:-2], second_diff_phi)
            ax6.axvline(x=roughness_values[inflection_lambda_index], color='r', linestyle='-')
            ax6.axvline(x=roughness_values[inflection_roughness_index], color='g', linestyle='--')
            ax6.axvline(x=self._find_inflection_point(roughness_values, rms_values)[0], color='b', linestyle='-')
            ax6.axhline(y=0, color='r', linestyle='-')

            fig2.tight_layout()
            fig2.show()
            fig2.savefig(self._folder_structure.get(
                'data_inversion_analysis') / f'{sounding}_{filter_times[0]}_{filter_times[1]}_phi_rrms_analysis.png')

            fig3, axs3 = plt.subplots(1, 3, figsize=(24, 8))
            ax7, ax8, ax9 = axs3[0], axs3[1], axs3[2]
            fig3.suptitle(
                f'Analysis of Inversion Results\nLambda varied between {test_range[0]} and {test_range[1]} with a spacing of {test_range[2]} for sounding {sounding}, lambda and roughness plotted',
                fontsize=14)
            ax7.set_title('Data plotting')
            ax7.set_ylabel('Roughness')
            ax7.set_xlabel('Lambda')
            ax7.scatter(lambda_values, roughness_values)
            ax7.axvline(x=lambda_values[inflection_lam_phi_index], color='r', linestyle='--')
            ax7.set_ylim(0, max(roughness_values) * 1.1)

            ax8.set_title('First Derivative')
            ax8.set_ylabel('first derivative')
            ax8.set_xlabel('Lambda')
            ax8.scatter(lambda_values[:-1], first_diff_lam_phi)
            ax8.axvline(x=lambda_values[inflection_lam_phi_index], color='r', linestyle='--')
            ax8.axvline(x=self._find_inflection_point(lambda_values, roughness_values)[0], color='b', linestyle='--')
            ax8.axhline(y=first_diff_lam[inflection_lam_phi_index], color='r', linestyle='--')
            ax8.plot(lambda_values[inflection_lam_phi_index], first_diff_lam[inflection_lam_phi_index], marker='o',
                     color='g')

            ax9.set_title('Second Derivative')
            ax9.set_ylabel('second derivative')
            ax9.set_xlabel('Lambda')
            ax9.axhline(y=0, color='r', linestyle='-')
            ax9.scatter(lambda_values[:-2], second_diff_lam_phi)
            ax9.axvline(x=lambda_values[inflection_lam_phi_index], color='r', linestyle='--')
            ax9.axvline(x=self._find_inflection_point(lambda_values, roughness_values)[0], color='b', linestyle='--')

            fig3.tight_layout()
            fig3.show()
            fig3.savefig(self._folder_structure.get(
                'data_inversion_analysis') / f'{sounding}_{filter_times[0]}_{filter_times[1]}_lambda_phi_analysis.png')

            fig4, axs4 = plt.subplots(1, 3, figsize=(24, 8))
            ax10, ax11, ax12 = axs4[0], axs4[1], axs4[2]
            fig4.suptitle(
                f'Analysis of Inversion Results\nLambda varied between {test_range[0]} and {test_range[1]} with a spacing of {test_range[2]} for sounding {sounding}, roughness and relRMS plotted',
                fontsize=14)
            ax10.set_title('Data plotting')
            ax10.set_xlabel('roughness')
            ax10.set_ylabel('relRMS (%)')
            ax10.scatter(roughness_values, rms_values, c='b')
            ax10.scatter(roughness_values, np.negative(rms_values), c='r')
            ax10.axvline(x=roughness_values[inflection_lambda_index], color='r', linestyle='-')
            ax10.axvline(x=roughness_values[inflection_roughness_index], color='g', linestyle='--')
            ax10.axvline(x=self._find_inflection_point(roughness_values, rms_values)[0], color='b', linestyle='-')
            ax10.loglog()

            ax11.set_title('Data plotting')
            ax11.set_xlabel('roughness')
            ax11.set_ylabel('first diff')
            ax11.scatter(roughness_values[:-1], first_diff_phi, c='b')
            ax11.scatter(roughness_values[:-1], np.negative(first_diff_phi), c='r')
            ax11.axvline(x=roughness_values[inflection_lambda_index], color='r', linestyle='-')
            ax11.axvline(x=roughness_values[inflection_roughness_index], color='g', linestyle='--')
            ax11.axvline(x=self._find_inflection_point(roughness_values, rms_values)[0], color='b', linestyle='-')
            ax11.loglog()

            ax12.set_title('Data plotting')
            ax12.set_xlabel('roughness')
            ax12.set_ylabel('second diff')
            ax12.scatter(roughness_values[:-2], second_diff_phi, c='b')
            ax12.scatter(roughness_values[:-2], np.negative(second_diff_phi), c='r')
            ax12.axvline(x=roughness_values[inflection_lambda_index], color='r', linestyle='-')
            ax12.axvline(x=roughness_values[inflection_roughness_index], color='g', linestyle='--')
            ax12.axvline(x=self._find_inflection_point(roughness_values, rms_values)[0], color='b', linestyle='-')
            ax12.axhline(y=0, color='r', linestyle='-')
            ax12.loglog()

            fig4.tight_layout()
            fig4.show()
            fig4.savefig(self._folder_structure.get(
                'data_inversion_analysis') / f'{sounding}_{filter_times[0]}_{filter_times[1]}_phi_rms_log_analysis.png')

        plot_analysis(sounding, plot_id, lambda_values, roughness_values, rms_values, test_range=test_range,
                      filter_times=filter_times)

    # todo: implement the 2D-parts (@jakob)
    def inversion_plot_2D_section(self, inversion_soundings, unit='rhoa', lam=600, lay_thk=3, save=True, max_depth=50):
        fig, ax = plt.subplots(figsize=(10, 4))
        if unit == 'rhoa':
            unit_label = r'$\rho$ ($\Omega$m)'
        elif unit == 'sigma_a':
            unit_label = r'$\sigma$ (mS/m)'
        else:
            raise SyntaxError('input {} not valid for Argument unit'.format(unit))

        all_xcoords = []
        all_ycoords = []
        all_values = []
        for lmnt in inversion_soundings:
            self.data_inversion(subset = [lmnt], lam=lam, lay_thk=lay_thk)
            xcoord = np.full(len(res_mdld), lmnt.posX)
            ycoord = np.arange(thks[0], thks[0] + len(res_mdld) * thks[0], thks[0])
            all_xcoords.append(xcoord)
            all_ycoords.append(ycoord)
            all_values.append(res_mdld)

        # Concatenate all x, y, and value arrays
        xcoords = np.concatenate(all_xcoords)
        ycoords = np.concatenate(all_ycoords)
        values = np.concatenate(all_values)
        unit_values = values if unit == 'rhoa' else 1000 / values

        sc = ax.scatter(xcoords, ycoords, c=unit_values, cmap='viridis', marker='o',
                        s=60)  # You can choose any colormap you like
        ax.invert_yaxis()
        ax.set_xlabel('Distance along profile (m)', fontsize=16)
        ax.set_ylabel('Depth (m)', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_ylim(max_depth + 2)

        # Customize colorbar to match the data values
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_ticks(np.linspace(np.min(unit_values), np.max(unit_values), num=6),
                       fontsize=14)  # Set ticks according to data range
        cbar.set_label(unit_label, fontsize=16)  # Set colorbar label
        fig.suptitle('Lambda = {:<8.0f} Layer Thickness = {:<.2f}m'.format(lam, lay_thk), fontsize=20,
                     fontweight='bold')
        plt.tight_layout()

        if save:
            fig.savefig(self.path_plot_2D + 'inversion_2Dsection_{}.png'.format(unit))

        self.section_coords = (xcoords, ycoords, values)
        return (xcoords, ycoords, values)

    def interpolation(self, lam=600, lay_thk=3):
        if self.section_coords is not None:
            xcoords_concat, ycoords_concat, values_concat = self.section_coords
        else:
            xcoords_concat, ycoords_concat, values_concat = self.inversion_plot_2D_section(lam=lam, lay_thk=lay_thk,
                                                                                           save=False)

        # Define grid for interpolation
        xi = np.linspace(xcoords_concat.min(), xcoords_concat.max(), 140)
        yi = np.linspace(ycoords_concat.min(), ycoords_concat.max(), 75)
        xi, yi = np.meshgrid(xi, yi)

        # Interpolate values
        zi = griddata((xcoords_concat, ycoords_concat), values_concat, (xi, yi),
                      method='linear')  # linear, nearest, cubic

        self.interpolation_res = (xi, yi, zi, xcoords_concat, ycoords_concat, values_concat)
        return (xi, yi, zi, xcoords_concat, ycoords_concat, values_concat)

    def plot_inversion_interpolated(self, unit='rhoa', lam=600, lay_thk=4.5, save=True, max_depth=50):
        fig, ax = plt.subplots(figsize=(10, 4))
        if unit == 'rhoa':
            unit_label = r'$\rho$ ($\Omega$m)'
        elif unit == 'sigma_a':
            unit_label = r'$\sigma$ (mS/m)'
        else:
            raise SyntaxError('input {} not valid for Argument unit'.format(unit))

        if self.interpolation_res is not None:
            xi, yi, zi, _, _, _ = self.interpolation_res
            value_i = zi if unit == 'rhoa' else 1000 / zi
        else:
            xi, yi, zi, _, _, _ = self.interpolation(lam=lam, lay_thk=lay_thk)
            value_i = zi if unit == 'rhoa' else 1000 / zi

        sc = ax.scatter(xi, yi, c=value_i, cmap='viridis', label='Interpolated Points', marker='s', s=7)
        ax.invert_yaxis()
        ax.set_xlabel('Distance along profile (m)', fontsize=16)
        ax.set_ylabel('Depth (m)', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_ylim(max_depth)

        # Customize colorbar to match the data values
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_ticks(np.linspace(np.min(value_i), np.max(value_i), num=6),
                       fontsize=14)  # Set ticks according to data range
        cbar.set_label(unit_label, fontsize=14)  # Set colorbar label
        fig.suptitle('Lambda = {:<8.0f} Layer Thickness = {:<.2f}m'.format(lam, lay_thk), fontsize=20,
                     fontweight='bold')
        plt.tight_layout()

        if save:
            fig.savefig(self.path_plot_2D + 'inversion_2Dsection_interpolated_{}.png'.format(unit))
        return fig