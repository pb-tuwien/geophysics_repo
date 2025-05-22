# -*- coding: utf-8 -*-
"""
Created on Wed Mar 5 18:33:00 2025

A class for the forward modelling of TEM data.

@author: jakob welkens & peter balogh @ TU Wien, Research Unit Geophysics
"""
#%% Importing modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import IPython.display as display
import matplotlib.gridspec as gridspec
from typing import Union
import itertools
from .empymod_forward import empymod_frwrd
from .utils import plot_model, plot_data

#%% ForwardTEM class

class ForwardTEM:
    
    def __init__(self):
        """
        Class to perform forward modelling of TEM data.

        - Settings can be changed through class attributes
        - A model can be added through add_...() methods
        - The modelling can be performed through the run() method
        """
        # Default plotting settings
        self.plot_title = None
        self.res2con = False
        self.time_limits = None
        self.signal_limits = None
        self.rho_limits = None
        self.rhoa_limits = None
        self.depth_limits = None

        # Default device settings
        self.device = 'TEMfast'
        self.timekey = 9
        self.currentkey = 1
        self.loop = 12.5
        self.current_inj = None
        self.filter_powerline = 50
        self.ramp_data = 'donauinsel'

        # Default modelling settings
        self.setup_solver = None
        self.model_type = None
        self.absolute_error = 1e-28
        self.relative_error = 1e-6

        # Internal defaults
        self.model = None
        self.response_signal = None
        self.noise_signal = None
        self.response_times = None
        self.response_rhoa = None
        self.noisy_response = None
        self.noisy_rhoa = None
        self.track = False
        self.fig = None
        self.ax_model = None
        self.ax_response = None
        self.__last_model_type = None
        self.__color_cycle = itertools.cycle(plt.cm.tab10.colors)
    
    @property
    def track(self) -> bool:
        '''
        Setting if plots should be tracked.
        If True multiple responses of different models can plotted in the same figure.
        '''
        return self.__track
    
    @track.setter
    def track(self, track: bool):
        '''
        Setter of track attribute
        '''
        if isinstance(track, bool):
            self.__track = track
        else:
            raise ValueError('Must be a boolean.')

    @property
    def setup_solver(self) -> Union[dict, None]:
        """
        The setup_solver attribute is a dictionary that contains the setup for the empymod_frwrd class.
        Look at the empymod_frwrd class for more information.
        """
        return self.__setup_solver
    
    @setup_solver.setter
    def setup_solver(self, setup_solver: Union[dict, None]):
        """
        Setter for the setup_solver attribute.
        """
        self.__setup_solver = setup_solver

    @property
    def device(self) -> str:
        """
        This attribute represents the device which is simulated for the forward modelling.
        Currently only 'TEMfast' is supported.
        """
        return self.__device
    
    @device.setter
    def device(self, device: str):
        """
        Setter for the device attribute.
        """
        if device != 'TEMfast':
            print('Currently only TEMfast is supported.')
        else:
            self.__device = device
    
    @property
    def timekey(self) -> int:
        """
        The timekey attribute represents the time key for the TEM-FAST device.
        Needs to be between 1 and 9.
        """
        return self.__timekey
    
    @timekey.setter
    def timekey(self, timekey: int):
        """
        Setter for the timekey attribute.
        """
        if timekey not in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
            print('Not supported. Used the default value instead.')
        else:
            self.__timekey = timekey

    @property
    def currentkey(self) -> int:
        """
        The currentkey attribute represents the current which the device will inject into the ground.
        Needs to be 1 or 4.
        """
        return self.__currentkey
    
    @currentkey.setter
    def currentkey(self, currentkey: int):
        """
        Setter for the currentkey attribute.
        """
        if currentkey not in [1, 4]:
            print('Not supported. Used the default value instead.')
        else:
            self.__currentkey = currentkey
        
    @property
    def loop(self) -> Union[int, float]:
        """
        The loop attribute represents the loop size of the TEM-FAST device using a single-loop configuration.
        It is given in Meters (side length of the square loop).
        """
        return self.__loop
    
    @loop.setter
    def loop(self, loop: Union[int, float]):
        """
        Setter for the loop attribute.
        """
        if loop < 0:
            print('Not supported. Used the default value instead.')
        else:
            self.__loop = loop

    @property
    def current_inj(self) -> Union[float, None]:
        """
        The current_inj attribute represents the current which the device actually injected into the ground.
        It is given in Amperes.
        If this information is not available, the theoretical current (== currentkey) is used.
        """
        return self.__current_inj
    
    @current_inj.setter
    def current_inj(self, current_inj: Union[int, float, None]):
        """
        Setter for the current_inj attribute.
        """
        try:
            self.__current_inj = float(current_inj)
        except TypeError:
            self.__current_inj = None
        
    @property
    def filter_powerline(self) -> int:
        """
        The filter_powerline attribute represents the powerline filter which is used for the data.
        It is given in Hz and can be 50 or 60.
        """
        return self.__filter_powerline
    
    @filter_powerline.setter
    def filter_powerline(self, filter_powerline: int):
        """
        Setter for the filter_powerline attribute.
        """
        if filter_powerline not in [50, 60]:
            print('Not supported. Used the default value instead.')
        else:
            self.__filter_powerline = filter_powerline
    
    @property
    def ramp_data(self) -> str:
        """
        The ramp_data attribute represents a location where the actual ramp times were measured.
        It can be 'donauinsel', 'salzlacken', 'hengstberg', or 'sonnblick'.
        If the location is not supported, the ramp data for 'donauinsel' is used as a approximation.
        """
        return self.__ramp_data
    
    @ramp_data.setter
    def ramp_data(self, ramp_data: str):
        """
        Setter for the ramp_data attribute.
        """
        if ramp_data not in ['donauinsel', 'salzlacken', 'hengstberg', 'sonnblick']:
            print('Not supported. Used the default value instead.')
        else:
            self.__ramp_data = ramp_data
    
    @property
    def model(self) -> np.ndarray:
        """
        The model attribute represents the 1D subsurface-model which is used for the forward modelling.
        The model is a numpy array with n rows and at least 2 columns. 

        The first columns represents the layer thicknesses and the other columns the different properies of each layer.
        """
        return self.__model
    
    @model.setter
    def model(self, model: Union[tuple, list, np.ndarray]):
        """
        Setter for the model attribute.
        """
        if model is None:
            self.__model = None
        else:
            try:
                self.__model = np.asarray(model)
            except ValueError:
                print('Model is not supported.')
    
    @property
    def res2con(self) -> bool:
        """
        The res2con attribute determines if the resistivity is converted to conductivity for the plotting.
        """
        return self.__res2con
    
    @res2con.setter
    def res2con(self, res2con: bool):
        """
        Setter for the res2con attribute.
        """
        if isinstance(res2con, bool):
            self.__res2con = res2con
        else:
            print('Not supported. Use boolean instead.')
    
    @property
    def time_limits(self) -> Union[tuple, None]:
        """
        The time_limits attribute represents the limits for the time axis in the plot.
        """
        return self.__time_limits
    
    @time_limits.setter
    def time_limits(self, time_limits: Union[tuple, None]):
        """
        Setter for the time_limits attribute.
        """
        if time_limits is None:
            self.__time_limits = None
        elif self.__check_limit(time_limits):
            self.__time_limits = time_limits
        else:
            print('Not supported. Used None instead.')
            self.__time_limits = None
    
    @property
    def signal_limits(self) -> Union[tuple, None]:
        """
        The signal_limits attribute represents the limits for the signal axis in the plot.
        """
        return self.__signal_limits
    
    @signal_limits.setter
    def signal_limits(self, signal_limits: Union[tuple, None]):
        """
        Setter for the signal_limits attribute.
        """
        if signal_limits is None:
            self.__signal_limits = None
        elif self.__check_limit(signal_limits):
            self.__signal_limits = signal_limits
        else:
            print('Not supported. Used None instead.')
            self.__signal_limits = None
    
    @property
    def rho_limits(self) -> Union[tuple, None]:
        """
        The rho_limits attribute represents the limits for the resistivity axis in the model plot.
        """
        return self.__rho_limits
    
    @rho_limits.setter
    def rho_limits(self, rho_limits: Union[tuple, None]):
        """
        Setter for the rho_limits attribute.
        """
        if rho_limits is None:
            self.__rho_limits = None
        elif self.__check_limit(rho_limits):
            self.__rho_limits = rho_limits
        else:
            print('Not supported. Used None instead.')
            self.__rho_limits = None
    
    @property
    def rhoa_limits(self) -> Union[tuple, None]:
        """
        The rhoa_limits attribute represents the limits for the apparent resistivity axis in the plot.
        """
        return self.__rhoa_limits
    
    @rhoa_limits.setter
    def rhoa_limits(self, rhoa_limits: Union[tuple, None]):
        """
        Setter for the rhoa_limits attribute.
        """
        if rhoa_limits is None:
            self.__rhoa_limits = None
        elif self.__check_limit(rhoa_limits):
            self.__rhoa_limits = rhoa_limits
        else:
            print('Not supported. Used None instead.')
            self.__rhoa_limits = None
    
    @property
    def depth_limits(self) -> Union[tuple, None]:
        """
        The depth_limits attribute represents the limits for the depth axis in the plot.
        """
        return self.__depth_limits
    
    @depth_limits.setter
    def depth_limits(self, depth_limits: Union[tuple, None]):
        """
        Setter for the depth_limits attribute.
        """
        if depth_limits is None:
            self.__depth_limits = None
        elif self.__check_limit(limits=depth_limits):
            self.__depth_limits = (depth_limits[1], depth_limits[0])
        else:
            print('Not supported. Used None instead.')
            self.__depth_limits = None

    def __check_limit(self, limits: tuple) -> bool:
        """
        Helper function to check if the limits are given as a tuple.
        """
        if isinstance(limits, tuple) and len(limits) == 2:
            if limits[0] < limits[1] and all(isinstance(i, (int, float)) for i in limits):
                return True
        return False
    
    @property
    def plot_title(self) -> Union[str, None]:
        '''
        Add a title to the figure using the suptitle method.
        '''
        return self.__plot_title

    @plot_title.setter
    def plot_title(self, title: Union[str, None]):
        '''
        Setter for plot_title attribute.
        '''
        if isinstance(title, (str, type(None))):
            self.__plot_title = title
        else:
            raise ValueError('Invalid input. Must be either a string or `None`')

    @property
    def model_type(self) -> Union[str, None]:
        """
        The type of subsurface model used for the forward modelling.
        If set to `None` a thickness/resistivity model is expected.
        For the modelling of IP-effects `pelton` and `mpa` (max phase angle) are available.
        """
        return self.__model_type
    
    @model_type.setter
    def model_type(self, model_type: Union[str, None]):
        """
        Setter for model_type attribute.
        """
        allowed = [None, 'pelton', 'mpa']
        not_tested = ['cole_cole', 'cc_kozhe', 'dielperm']

        if model_type in allowed:
            self.__model_type = model_type
        elif model_type in not_tested:
            print('Model type was not tested.')
            self.__model_type = model_type
        else:
            raise ValueError('Model type not available.')
    
    @property
    def absolute_error(self) -> float:
        """
        The absolute error used for simulation of the data error.
        """
        return self.__absolute_error
    
    @absolute_error.setter
    def absolute_error(self, absolute_error: Union[int, float]):
        """
        Setter for absolute_error attribute.
        """
        try:
            self.__absolute_error = float(absolute_error)
            if absolute_error < 0:
                raise ValueError('Must be larger than 0.')
        except TypeError:
            raise TypeError('Must be convertible to float.')
    
    @property
    def relative_error(self) -> float:
        """
        The relative error used for simulation of the data error.
        """
        return self.__relative_error
    
    @relative_error.setter
    def relative_error(self, relative_error: Union[int, float]):
        """
        Setter for relative_error attribute.
        """
        try:
            relative_error = float(relative_error)
            if relative_error < 0:
                raise ValueError('Must be larger than 0.')
            self.__relative_error = relative_error
        except TypeError:
            raise TypeError('Must be convertible to float.')
    
    def add_resistivity_model(self, thickness: np.ndarray, resistivity: np.ndarray):
        """
        Add a simple resistivity model.
        It takes **two** arrays with the same length.

        The lenght of the array represents the number of layers.

        Parameters
        ----------
        thickness: array-like object
            The thicknesses of each layer.
        resistivity: array-like object
            The resistivities of each layer.
        """
        thk = np.asarray(thickness)
        res = np.asarray(resistivity)
        if thk.shape != res.shape:
            raise ValueError('Thickness and resistivity must have the same lenght.')
        model = np.column_stack((thk, res))
        if (model <= 0).sum() != 0:
            raise ValueError('Thicknesses and resistivities must be greater than 0.')
        self.model = model
        self.model_type = None

    def add_pelton_model(self, thickness: np.ndarray, resistivity: np.ndarray, 
                         chargeability: np.ndarray, relaxation_time: np.ndarray, 
                         dispersion_coefficient: np.ndarray):
        """
        Add a **pelton** complex resistivity model.
        It takes **five** arrays with the same length.

        The lenght of the array represents the number of layers.
        
        - **m: chargeability**
        - **tau: relaxation_time**
        - **c: dispersion_coefficient**

        Parameters
        ----------
        thickness: array-like object
            The thicknesses of each layer.
        resistivity: array-like object
            The resistivities of each layer.
        chargeability: array-like object
            The chargeability of each layer. The values must be between 0 and 1.
        relaxation_time: array-like object
            The relaxation time of each layer in seconds. The values should be between 0 and 1.
        dispersion_coefficient: array-like object
            The dispersion coefficient of each layer. The values must be between 0 and 1.
        """
        thk = np.asarray(thickness)
        res = np.asarray(resistivity)
        m = np.asarray(chargeability)
        tau = np.asarray(relaxation_time)
        c = np.asarray(dispersion_coefficient)

        try:
            model = np.column_stack((thk, res, m, tau, c))
        except ValueError:
            raise ValueError('All inputs mus have the same lenght.')
        if (model[:,:2] <= 0).sum() != 0:
            raise ValueError('Thicknesses and resistivities must be greater than 0.')
        if (model[:,2:] < 0).sum() + (model[:,2:] > 1).sum():
            raise ValueError('The values of m, tau and c must be between 0 and 1.')
        
        self.model_type = 'pelton'
        self.model = model

    def add_mpa_model(self, thickness: np.ndarray, resistivity: np.ndarray, 
                      max_phase_angle: np.ndarray, relaxation_time: np.ndarray, 
                      dispersion_coefficient: np.ndarray):
        """
        Add a **max phase angle** complex resistivity model.
        It takes **five** arrays with the same length.

        The lenght of the array represents the number of layers.
        
        - **phi: max_phase_angle**
        - **tau_phi: relaxation_time**
        - **c: dispersion_coefficient**

        Parameters
        ----------
        thickness: array-like object
            The thicknesses of each layer.
        resistivity: array-like object
            The resistivities of each layer.
        max_phase_angle: array-like object
            The max phase angle of each layer. The values must be between 0 and 1.
        relaxation_time: array-like object
            The relaxation time of each layer in seconds. The values should be between 0 and 1.
        dispersion_coefficient: array-like object
            The dispersion coefficient of each layer. The values must be between 0 and 1.
        """
        thk = np.asarray(thickness)
        res = np.asarray(resistivity)
        phi_max = np.asarray(max_phase_angle)
        tau = np.asarray(relaxation_time)
        c = np.asarray(dispersion_coefficient)

        try:
            model = np.column_stack((thk, res, phi_max, tau, c))
        except ValueError:
            raise ValueError('All inputs mus have the same lenght.')
        if (model[:,:2] <= 0).sum() != 0:
            raise ValueError('Thicknesses and resistivities must be greater than 0.')
        if (model[:,2] < 0).sum() != 0:
            raise ValueError('Phi_max must be greater than or equal to 0.')
        if (model[:,3] < 0).sum() + (model[:,3] > 1).sum():
            raise ValueError('The values of tau_phi and c must be between 0 and 1.')
        
        self.model_type = 'mpa'
        self.model = model

    def prepare_setup_device(self):
        """
        Creates a dictionary with all the necessary keys for forward modeller (empymod_frwrd).
        """
        if self.__current_inj is None:
            current_inj = float(self.__currentkey)
            current_key = self.__currentkey
        else:
            current_inj = self.__current_inj
            current_key = np.round(self.__currentkey)

        setup_device = {
            "timekey": self.__timekey,
            "currentkey": current_key,
            "txloop": self.__loop,
            "rxloop": self.__loop,
            "current_inj": current_inj,
            "filter_powerline": self.__filter_powerline,
            "ramp_data": self.__ramp_data
        }
        return setup_device
    
    def plot_data(self, ax: Union[matplotlib.axes.Axes, None] = None, show_noise: bool = False, 
                  label: Union[str, None] = None, xlimits: Union[tuple, None] = None, 
                  ylimits_signal: Union[tuple, None] = None, ylimits_rhoa: Union[tuple, None] = None, 
                  log_rhoa: bool = False, res2con: bool = None,
                  plot_noisy: bool = False, legend=True, **kwargs):
        """
        Method to plot the last modelled signal response and apparent resistivity with a predefined style.

        Parameters
        ----------
        ax : array-like object, optional
            Use this if you want to plot to existing axes.
            Must contain two matplotlib.axes.Axes objects.
            If set to None a new figure and ax is created.
            The default is None.
        show_noise : boolean, optional
            If the simulated noise of the signal should also be plotted. The default is False.
        label: str, optional
            Sets a label to the signal plot. The default is None (No label).
        xlimits : tuple or None, optional
            X-axis limits. The default is None (No limit is applied).
        ylimits_signal : tuple or None, optional
            Y-axis limits for the signal subplot. The default is None (No limit is applied).
        ylimits_rhoa : tuple or None, optional
            Y-axis limits for the rhoa subplot. The default is None (No limit is applied).
        log_rhoa: boolean, optional
            If the y-axis of the rhoa plot should be set to a log-scale. The default is False.
        res2con: boolean, optional
            If the resistivity should be converted to conductivity (Ohm*m to mS/m). The default is None (copies res2con attribute).
        plot_noisy: boolean, optional
            If a an simulated random error should also be considered. The default is False.
        
        Additional keyword parameters
        -----------------------------
        sub0_color_signal (str):
            Markeredgecolor for the sub zero marker in the signal plot. The default is 'k'.
        sub0_color_rhoa (str):
            Markeredgecolor for the sub zero marker in the rhoa plot. The default is 'k'.
        show_sub0_label (bool):
            Add a label to the sub zero values. The default is True.
        signal_title (str):
            The title of the ax with the signal subplot. If set to None a default title is set. The default is None.
        rhoa_title (str):
            The title of the ax with the rhoa subplot. If set to None a default title is set. The default is None.
        rhoa_label (str):
            Sets a label to the rhoa plot. The default is None (No label).
        **kwargs :
            Other keyword arguments are used in the plotting functions of both signal and rhoa plots (loglog() and semilogx()).
        Returns
        -------
        np.ndarray
        Returns either the provided or the created axes in a numpy array.
        """
        if self.response_times is None:
            raise ValueError('No modelled date found.')
        if res2con is None:
            res2con = self.res2con
        if xlimits is None:
            xlimits = self.time_limits
        if ylimits_signal is None:
            ylimits_signal = self.signal_limits
        if ylimits_rhoa is None:
            ylimits_rhoa = self.rhoa_limits

        if show_noise:
            noise_signal = np.abs(self.noise_signal)
        else:
            noise_signal = None
        
        if plot_noisy:
            signal = self.noisy_response
            rhoa = self.noisy_rhoa
        else:
            signal = self.response_signal
            rhoa = self.response_rhoa
        
        ax = np.asarray(ax)
        ax = plot_data(
            time=self.response_times, ax=ax, signal_label=label,
            signal=signal, noise_signal=noise_signal,
            rhoa=rhoa, noise_rhoa=None,
            xlimits=xlimits, ylimits_signal=ylimits_signal,
            ylimits_rhoa=ylimits_rhoa, log_rhoa=log_rhoa,
            res2con=res2con, legend=legend, **kwargs
            )
        
        return ax
        
    def plot_model(self, model: np.ndarray, ax: Union[matplotlib.axes.Axes, None] = None, 
                   xlimits: Union[tuple, None] = None, ylimits: Union[tuple, None] = None,
                   add_bottom: Union[float, int] = 0, res2con: bool = False, 
                   **kwargs) -> matplotlib.axes.Axes:
        """
        Method to plot the subsurface (resistivity) model with a predefined style.
        It plots a 1D model of the subsurface. 
        The provided model must have two dimensions: 
        The depth and the value of the plotted subsurface property. 

        The function was created to plot the resistivity but it can be used to plot other properties as well.
        If something other than the resistivity is plotted, make sure to set res2con to False (or leave it at the default value).

        Parameters
        ----------
        model: np.ndarray
            The model of the subsurface.
        ax : matplotlib.axes.Axes object, optional
            Use this if you want to plot to an existing ax.
            If set to None a new figure and ax is created.
            The default is None.
        xlimits : tuple or None, optional
            X-axis limits. The default is None (No limit is applied).
        ylimits : tuple or None, optional
            Y-axis limits. The default is None (No limit is applied).
        add_bottom: float or int, optional
            By how much meters the last layer should be expanded for the plotting. The default is 0.
        res2con: boolean, optional
            If the resistivity should be converted to conductivity (Ohm*m to mS/m). The default is False.
        
        Additional keyword parameters
        -----------------------------
        title (str):
            Add a title to the plot. The default is None (No title added).
        **kwargs :
            Other keyword arguments are used for the plot() function.
        
        Returns
        -------
        matplotlib.axes.Axes
        Returns either the provided or the created ax.
        """
        if res2con is None:
            res2con = self.res2con
        if xlimits is None:
            xlimits = self.rho_limits
        if ylimits is None:
            ylimits = self.depth_limits
        
        output = plot_model(
            model=model, ax=ax,
            add_bottom=add_bottom, res2con=res2con,
            xlimits=xlimits, ylimits=ylimits,
            **kwargs)
        return output

    def __get_model_parameters(self):
        """
        This function gets all the necessary information on title names, 
        axis-labels and axis-limits for the plotting of different resistivity models.

        **Currently only the simple, pelton and mpa models work.**

        Returns
        -------
        Returns 3 lists: lables, titles, limits.
        """
        res_labels = ['filler']
        pelton_labels = ['filler', 'm', r'$\tau$ (s)', 'c']
        mpa_labels = ['filler', '$\phi_{max}$ (rad)', r'$\tau_{\phi}$ (s)', 'c']
        # cole_cole_labels = [] TODO
        # cc_kozhe_labels = [] TODO
        # dielperm_labels = [] TODO

        title_dict = {
            'm': 'Chargeability',
            r'$\tau$ (s)': 'Relaxation Time', 
            'c': 'Dispersion Coefficient',
            '$\phi_{max}$ (rad)': 'Max. Phase Angle', 
            r'$\tau_{\phi}$ (s)': 'Relaxation Time'
            # TODO: add more
            }
        limits_dict = {
            'm': (-0.1, 1.1),
            r'$\tau$ (s)': None, 
            'c': (-0.1, 1.1),
            '$\phi_{max}$ (rad)': (-0.1, 6.4), 
            r'$\tau_{\phi}$ (s)': None
            # TODO: add more
        }

        if self.model_type is None:
            labels = res_labels
        elif self.model_type == 'pelton':
            labels = pelton_labels
        elif self.model_type == 'mpa':
            labels = mpa_labels
        else:
            print('Model not implemented entirely yet.')
            labels = res_labels
        
        titles = [title_dict.get(name) for name in labels]
        limits = [limits_dict.get(name) for name in labels]

        return labels, titles, limits

    def run(self, description: Union[str, None] = None, color: Union[str, None] = None, 
            plot_noisy: bool = False, legend: bool = True): 
        """
        This function runs the forward modeller with the given settings and model, creating a plot.
        Currently only a **simple** resistivity model as well as the **pelton** and **mpa** models are implemented properly.

        Parameters
        ----------
        description: str, optional
            Add a description as the label of the plot. The default is None (No label is added).
        color: str, optional
            Define a color which should be used for the plot. The default is None (A color from the internal color cycle of the TAB10 colors).
        plot_noisy: boolean, optional
            If a an simulated random error should also be considered (depends on the absolute and relative error attributes).
            The default is False.
        """
        if self.model is None:
            raise ValueError('Model is not set. Please set the model.')
        if self.fig is None:
            self.__last_model_type = self.model_type
        if self.track and self.model_type != self.__last_model_type:
            raise TypeError('For multiple models in the same figure, the modeltype must be the same.\nSet self.track to false to use a new modeltype.')
        
        if self.model_type is None and self.model.shape[1] != 2:
            raise ValueError('Resistivity model must have two columns.')
        elif self.model_type in ['pelton', 'mpa'] and self.model.shape[1] != 5:
            raise ValueError('Model must have 5 columns.')
        elif self.model_type in ['cole_cole', 'dielperm'] and self.model.shape[1] != 6:
            raise ValueError('Model must have 6 columns.')
        elif self.model_type == 'cc_kozhe' and self.model.shape[1] != 7:
            raise ValueError('Model must have 7 columns.')
        
        if color is None:
            color = next(self.__color_cycle)
        
        used_model = self.model
        if float(used_model[-1, 0]) != 0.0:
            used_model = np.vstack((used_model, np.r_[0, used_model[-1, 1:]]))
        
        nlayer = used_model.shape[0]
        nparam = used_model.shape[1]
        nmodel = nparam - 1

        frwrd_solver = empymod_frwrd(
            setup_device=self.prepare_setup_device(),
            setup_solver=self.setup_solver,
            time_range=None, 
            device=self.device,
            nlayer=nlayer, 
            nparam=nparam,
            abserr=self.absolute_error,
            relerr=self.relative_error
        )

        self.response_signal = frwrd_solver.calc_response(model=used_model, ip_modeltype=self.model_type)
        self.noise_signal = frwrd_solver.response_noise
        self.response_times = frwrd_solver.times_rx
        self.response_rhoa = frwrd_solver.rhoa
        self.noisy_response = frwrd_solver.noisy_response
        self.noisy_rhoa = frwrd_solver.noisy_rhoa

        if self.track and self.fig is not None:
            fig = self.fig
            ax_model = self.ax_model
            ax_response = self.ax_response
        else:
            if nmodel > 1:
                fig = plt.figure(figsize=(10, 10) ,constrained_layout=True)

                gs0 = gridspec.GridSpec(2, 1, figure=fig)
                gs1 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs0[0])
                gs2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[1])

                ax_model = [fig.add_subplot(gs1[n]) for n in range(nmodel)]
                ax_response = [fig.add_subplot(gs2[n]) for n in range(2)]
            else:
                fig = plt.figure(figsize=(14, 7) ,constrained_layout=True)

                ax_model = [fig.add_subplot(1,5,1)]
                ax_response = [fig.add_subplot(1,5,(2,3)), fig.add_subplot(1,5,(4,5))]
            
            label_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
            for ax, label in zip(ax_model + ax_response, label_list):
                ax.text(0.96, 0.96, f'({label})', transform=ax.transAxes, fontsize=12, zorder=5,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(facecolor='xkcd:light grey', boxstyle='round,pad=0.4', alpha=0.3))

        if self.track:
            show_noise = False
        else:
            show_noise = True
        self.plot_data(
            ax=ax_response, show_noise=show_noise, 
            label=description, color=color,
            plot_noisy=plot_noisy, legend=legend
            )

        labels, titles, limits = self.__get_model_parameters()
        for i, label, title, limit in zip(range(len(labels)), labels, titles, limits):
            thk = used_model[:,0]
            parameter = used_model[:,i+1]
            model = np.column_stack((thk, parameter))
            if i > 0:
                res2con = False
            else:
                res2con = self.res2con
            self.plot_model(model=model, ax=ax_model[i], res2con=res2con, color=color)
            if i > 0:
                ax_model[i].set_title(title, fontsize=14)
                ax_model[i].set_xlabel(label)
                if limit is None:
                    ax_model[i].autoscale(enable=True, axis='x', tight=False)
                else:
                    ax_model[i].set_xlim(limit)

        fig.suptitle(self.plot_title, fontsize=16, fontweight='bold')
        if self.track and self.fig is not None:
            display.display(fig)

        self.fig = fig
        self.ax_model = ax_model
        self.ax_response = ax_response
    
    def clear_plot(self):
        """
        Clears the figures and axes from memory:

        Resets the fig, ax_model, and ax_response attributes to None.
        Also resets the color cycle.
        
        """
        self.fig = None
        self.ax_model = None
        self.ax_response = None
        self.__color_cycle = itertools.cycle(plt.cm.tab10.colors)

        print('Cleared plots from memory.')
    
    def savefig(self, filename, dpi=300, bbox_inches='tight', **kwargs):
        """
        Function to save the last figure. Based on fig.savefig() method of matplotlib.
        """
        if self.fig is None:
            print('No figure found to save.')
        else:
            self.fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
