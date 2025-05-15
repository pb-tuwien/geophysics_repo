# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 11:59:00 2025

A class for the interactive forward modelling of TEM data in the jupyter lab environment.

@author: jakob welkens @ TU Wien, Research Unit Geophysics
"""
#%% Importing modules
from pathlib import Path
import numpy as np
import pandas as pd
from ipywidgets.widgets import * 
import ipywidgets as widgets
from IPython.display import display, clear_output
from datetime import datetime
import matplotlib.pyplot as plt


from .modeller import ForwardTEM
from .utils import save_as_tem
# from .utils import get_TEMFAST_rampdata


#%% Interactive forward modelling class
class InteractiveTEMLayer():
    def __init__(self, number):
        self.t_layout = Layout(width='150px')
        self.l_layout = Layout(width='450px')
        self.s_thk = FloatSlider(value=5, min=0, max=20, step=0.5, description='', readout=False)
        self.s_rho_a = FloatLogSlider(value=10, base=10, min=0, max=4, step=.01, description='', readout=False)
        self.t_thk = FloatText(value='5', min=0, max=20, description='', layout=self.t_layout)
        self.t_rho_a= FloatText(value='20', description='', layout=self.t_layout)
        self.l_label = Label('Layer {}'.format(number+1))
        widgets.link((self.s_thk, 'value'), (self.t_thk, 'value'))
        widgets.link((self.s_rho_a, 'value'), (self.t_rho_a, 'value'))
        self.layer = HBox([self.l_label, self.s_thk, self.t_thk, self.s_rho_a, self.t_rho_a])
    

# %%
# initial settings and initialisation
class JupyterInteractiveTEM(ForwardTEM):
    def __init__(self, layer_number=3):
        super().__init__()
        self.wd = Path.cwd()
        self.root_path = Path(__file__).parent
        self.figures_path = self.wd / "results" / "figures"
        self.models_path = self.wd / "results" / "models"
        self.responses_path = self.wd / "results" / "responses"
        self.import_model_path = self.root_path / "reference_models"
        self.project_name = 'TEM_project'
        self.author = Path.home().name
        self.__layer_number = layer_number
        self._layout_l = Layout(width='470px')
        self._layout_s = Layout(width='650px')
        self._style_heading = dict(font_weight='bold', font_size='20px')
        
        self.__layer_list = []
        self.__rho_a_list = []
        self.__thk_list = []
        self.settings_visibility = True

        # loop to create the initial layers
        for i in range(self.__layer_number):
            i = InteractiveTEMLayer(i)
            self.__layer_list.append(i)
            i.s_thk.observe(self.on_thk_change, names='value')
            i.s_rho_a.observe(self.on_rho_a_change, names='value')
            self.__rho_a_list.append(i.s_rho_a.value)
            self.__thk_list.append(i.s_thk.value)

    def start(self):
        '''
        initialising the UI
        '''
        self.initialize_model()
        self.layer_ui()
        self.save_import_ui()
        self.settings_ui()
        #self.update()
        return self

    @property
    def author(self):
        """
        A string containing the name of the author of the project.
        """
        return self.__author
    
    @author.setter
    def author(self, author):
        """
        Setter for the author property.
        """
        if not isinstance(author, str):
            print('Author must be a string')
        else:
            self.__author = author
    
    @property
    def project_name(self):
        """
        A string containing the name of the project.
        """
        return self.__project_name
    
    @project_name.setter
    def project_name(self, project_name):
        """
        Setter for the project_name property.
        """
        if not isinstance(project_name, str):
            print('Project name must be a string')
        else:
            self.__project_name = project_name

        # layer UI
    def layer_ui(self):    
        self.header = VBox([Label('Subsurface Model', layout=self._layout_l, style=self._style_heading), 
                            HBox([Label('Thickness of each layer in meters', layout=self._layout_l), 
                            Label('Resistivity of each layer in Ohm meters', layout=self._layout_l)])])
        
        # adding setting changes in the ui
        self.b_add_layer = Button(description='Add layer', button_style='success')
        self.b_add_layer.on_click(self.on_add_layer)
        self.b_remove_layer = Button(description='Remove layer', button_style='danger')
        self.b_remove_layer.on_click(self.on_remove_layer)
            

    def settings_ui(self):
        self.header_settings = Label('Settings', layout=self._layout_l, style=self._style_heading)

        self.c_show_settings = Checkbox(value=self.settings_visibility,
                                        description='Show Settings',
                                        disabled=False,
                                        indent=False)

        self.c_show_settings.observe(self.on_show_settings, names='value')

    
        self.b_current_key = RadioButtons(
            options=[1, 4],
            value=self.currentkey,
            description='Current injection:',
            disabled=False,
            layout={'width': 'max-content'},
            orientation='horizontal'
        )

        # self.b_current_key.observe(self.update_current_key, names='value')

        self.s_loopsize = FloatSlider(
            value=self.loop,
            min=1,
            max=50.0,
            step=0.25,
            description='Loop size in meters:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.2f',
            layout=self._layout_s,
            style={'description_width': '200px'}
        )

        # self.s_loopsize.observe(self.update, names='value')

        self.s_current_inj = FloatSlider(
            value=self.b_current_key.value,
            min=self.b_current_key.value * .75, #something not working
            max=self.b_current_key.value * 1.25, # here as well
            step=.01,
            description='Current injection in Ampere:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.2f',
            layout=self._layout_s,
            style={'description_width': '200px'}
        )

        # self.s_current_inj.observe(self.update, names='value')

        # widgets.link((self.b_current_key, 'value'), (self.s_current_inj, 'max'))
        # widgets.link((self.b_current_key, 'value'), (self.s_current_inj, 'min'), transform=lambda x: x * 0.75)
        # widgets.link((self.b_current_key, 'value'), (self.s_current_inj, 'max'), transform=lambda x: x * 1.25)
        # widgets.dlink((self.b_current_key, 'value'), (self.s_current_inj, 'value'))

        self.s_timekey = IntSlider(
            value=self.timekey,
            min=1,
            max=9,
            step=1,
            description='Time key:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d',
            layout=self._layout_s,
            style={'description_width': '200px'}
        )

        # self.s_timekey.observe(self.update, names='value')


        self.b_filter_powerline = RadioButtons(
            options=[50, 60],
            value=self.filter_powerline,
            description='Filter powerline:',
            disabled=False,
            layout={'width': 'max-content'},
            orientation='horizontal'
        )

        # self.b_filter_powerline.observe(self.update, names='value')

    def show_settings(self):
            if self.c_show_settings.value:
                self.settings = VBox([self.header_settings, self.c_show_settings, self.b_current_key, self.s_loopsize, self.s_timekey, self.b_filter_powerline])

            else:
                self.settings = VBox([self.header_settings, self.c_show_settings])
            

    # save import buttons
    def save_import_ui(self):
        self.b_save_fig = Button(description='Save Figure', button_style='info')
        self.b_save_fig.on_click(self.save_fig)
        self.b_save_model = Button(description='Save Model', button_style='info')
        self.b_save_model.on_click(self.save_model)
        self.b_save_response = Button(description='Save Response', button_style='info')
        self.b_save_response.on_click(self.save_response)
        self.b_save_all = Button(description='Save All', button_style='info')
        self.b_save_all.on_click(self.save_all)
        #self.b_load_model_div = Button(description='Load Saved Model', button_style='info')
        #self.b_load_model_div.on_click(self.load_model_div)
        self.b_load_model_soda = Button(description='Load Soda Lakes', button_style='info')
        self.b_load_model_soda.on_click(self.load_model_soda)
        self.b_load_model_peat = Button(description='Load Peatlands', button_style='info')
        self.b_load_model_peat.on_click(self.load_model_peat)
        #self.b_update = Button(description='Refresh', button_style='warning')
        #self.b_update.on_click(self.on_update)

    # display boxes for convenieces sake
    def renew_ui(self):
        self.show_settings()
        self.slider_display = VBox([layer.layer for layer in self.__layer_list])
        self.save_import = VBox([HBox([self.b_save_fig, self.b_save_model, self.b_save_response, self.b_save_all]),
                                    HBox([self.b_load_model_soda, self.b_load_model_peat])])
        self.ui = VBox([self.settings, self.header, self.slider_display,self.save_import])

    # initial display of the ui
    def display_ui(self):
        display(self.ui)

    # initialisation of the plot - first solving of the model
    def initialize_model(self):
        self.model = np.column_stack((self.__thk_list, self.__rho_a_list))


    def update_current_key(self, change):
        if self.b_current_key.value == 1:
            self.s_current_inj.min = self.b_current_key.value * .75
            self.s_current_inj.max = self.b_current_key.value * 1.25
        else:
            self.s_current_inj.max = self.b_current_key.value * 1.25
            self.s_current_inj.min = self.b_current_key.value * .75
        self.s_current_inj.value = self.b_current_key.value

    def update(self, ramp_data: str = 'donauinsel'):
        '''function to update the model and the plot for all changes in the sliders'''
        clear_output(wait=False)
        for i in range(len(self.__layer_list)):
            self.__rho_a_list[i] = self.__layer_list[i].s_rho_a.value
            self.__thk_list[i] = self.__layer_list[i].s_thk.value
        self.loop = self.s_loopsize.value
        self.currentkey = self.b_current_key.value
        self.timekey = self.s_timekey.value
        self.current_inj = self.s_current_inj.value
        self.filter_powerline = self.b_filter_powerline.value
        self.ramp_data = ramp_data
        self.model = np.column_stack((self.__thk_list, self.__rho_a_list))
        self.renew_ui()
        self.display_ui()
        self.run()

    def show(self, ramp_data:str='donauinsel'):
        '''function to show the plot'''
        clear_output(wait=False)
        for i in range(len(self.__layer_list)):
            self.__rho_a_list[i] = self.__layer_list[i].s_rho_a.value
            self.__thk_list[i] = self.__layer_list[i].s_thk.value
        self.loop = self.s_loopsize.value
        self.currentkey = self.b_current_key.value
        self.timekey = self.s_timekey.value
        self.current_inj = self.s_current_inj.value
        self.filter_powerline = self.b_filter_powerline.value
        self.ramp_data = ramp_data
        self.model = np.column_stack((self.__thk_list, self.__rho_a_list))
        self.run()


    # observation corresponding functions
    def on_show_settings(self, change):
        self.show_settings()
        
    def on_thk_change(self, change):
        pass

    def on_rho_a_change(self, change):
        pass


    def on_add_layer(self, click):
        self.__layer_list.append(InteractiveTEMLayer(self.__layer_number))
        self.__layer_list[-1].s_thk.observe(self.on_thk_change, names='value')
        self.__layer_list[-1].s_rho_a.observe(self.on_rho_a_change, names='value')
        self.__rho_a_list.append(self.__layer_list[-1].s_rho_a.value)
        self.__thk_list.append(self.__layer_list[-1].s_thk.value)
        self.__layer_number +=1

    def on_remove_layer(self, click):
        if len(self.__layer_list) > 2:
            self.__layer_list.pop(-1)
            self.__rho_a_list.pop(-1)
            self.__thk_list.pop(-1)
            self.__layer_number -= 1
        else:
            raise ValueError('You need to have at least 2 layers')
        
    @property
    def layer_number(self):
        return self.__layer_number
    
    @layer_number.setter
    def layer_number(self, number):
        if self.__layer_number == number:
            pass
        elif self.__layer_number < number:
            for i in range(number - self.__layer_number):
                self.b_add_layer.click()
        elif self.__layer_number > number:
            for i in range(self.__layer_number - number):
                self.b_remove_layer.click()
 


    def create_folders(self):
        for path in [self.figures_path, self.models_path, self.responses_path, self.import_model_path]:
            path.mkdir(parents=True, exist_ok=True)

    def save_fig(self, click):
        self.create_folders()
        filename = self.filename_input('figure')
        self.savefig(self.figures_path / '{}.png'.format(filename))

    def save_model(self, click):
        self.create_folders()
        filename = self.filename_input('model')
        model = pd.DataFrame({'thickness':self.__thk_list, 'resistivity':self.__rho_a_list})
        model.to_csv(self.models_path / '{}.csv'.format(filename))


    def save_response(self, click): 
        self.create_folders()
        filename = self.filename_input('response')
        save_as_tem(savepath=str(self.responses_path), 
                template_fid=self.root_path / 'template.tem'  , filename='/{}.tem'.format(filename), 
                metadata={'location':'fwrdmodel', 'snd_name':'{project}{time}'.format(project=self.project_name,time=datetime.now().strftime('%Y-%m-%d_%H-%M-%S') ), 'comments':'', 'x':0, 'y':0, 'z':0},
                setup_device=self.prepare_setup_device(), properties_snd={'rampoff':0.1, 'current_inj':self.s_current_inj.value},
                times=self.response_times, signal=self.response_signal, error=self.noise_signal, rhoa=self.response_rhoa)


    def save_all(self, click):
        self.create_folders()
        self.b_save_fig.click()
        self.b_save_model.click()
        self.b_save_response.click()


    def load_model_div(self, click):
        self.d_file_selection = Dropdown(options=[str(file.name) for file in self.models_path.glob('*.csv')], title='Select the model to load', 
                                         button_text='Load Model', function=self.load_model, layout=self._layout_l)
        self.b_file_selection = Button(description='Choose Model', button_style='info')
        self.b_file_selection.on_click(self.load_model)
        self.file_selection = VBox([Label('Select Model-File to be Loaded', layout=self._layout_l, style=self._style_heading), self.d_file_selection, self.b_file_selection])
        display(self.file_selection)

    def load_model_soda(self, click):
        self.load_model('sodalakes')

    def load_model_peat(self, click):
        self.load_model('peatland')

    def load_model(self, keyword):
        self.create_folders()
        if keyword in ['sodalakes', 'peatland']:
            model = pd.read_csv(self.import_model_path / '{}.csv'.format(keyword))
        else:
            model = pd.read_csv(self.models_path / '{}'.format(self.d_file_selection.value))
        while self.__layer_number != len(model['thickness']):
            if self.__layer_number < len(model['thickness']):
                self.b_add_layer.click()
            elif self.__layer_number > len(model['thickness']):
                self.b_remove_layer.click()
            else:
                break
        for i in self.__layer_list:
            i.s_thk.value = model['thickness'][self.__layer_list.index(i)]
            i.s_rho_a.value = model['resistivity'][self.__layer_list.index(i)]

    def on_update(self, click):
        self.update()


    def filename_input(self, function):
        self.l_filename = Label('Please enter the filename for the {}'.format(function), layout=self._layout_l)
        self.t_filename = Text(value='{project}_{author}_{time}'.format(project=self.project_name,
            author=self.author, time=datetime.now().strftime('%Y-%m-%d_%H-%M-%S')), description='', layout=self._layout_l)
        self.b_filename = Button(description='Save', button_style='info')
        self.b_filename.on_click(self.close_filename)
        #display(HBox([self.l_filename, self.t_filename, self.b_filename]))
        return self.t_filename.value
    
    def close_filename(self, click):
        self.l_filename.close()
        self.t_filename.close()
        self.b_filename.close()

        

# %%
