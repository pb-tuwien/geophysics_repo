import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from emagpy import Problem
from datetime import datetime
import time
import pandas as pd
from pathlib import Path
from ipywidgets.widgets import FloatSlider, FloatText, Label, Button, Layout, HBox, VBox, SelectionSlider, RadioButtons, FloatLogSlider
import ipywidgets as widgets
from IPython.display import display, clear_output

#%% 
class InteractiveEMILayer():
    def __init__(self, number):
        self.t_layout = Layout(width='150px')
        self.l_layout = Layout(width='450px')
        self.s_thk = FloatSlider(value=1, min=0, max=20, step=0.5, description='', readout=False)
        self.s_sigma_a = FloatLogSlider(value=10, base=10, min=0, max=4, step=.01, description='', readout=False)
        self.t_thk = FloatText(value='1', min=0, max=20, description='', layout=self.t_layout)
        self.t_sigma_a= FloatText(value='10', description='', layout=self.t_layout)
        self.l_label = Label('Layer {}'.format(number+1))
        widgets.link((self.s_thk, 'value'), (self.t_thk, 'value'))
        widgets.link((self.s_sigma_a, 'value'), (self.t_sigma_a, 'value'))
        self.layer = HBox([self.l_label, self.s_thk, self.t_thk, self.s_sigma_a, self.t_sigma_a])

#%%

class InteractiveCoil():
    def __init__(self, number, device):
        self.t_layout = Layout(width='150px')
        self.dev_number = number
        self.l_label = Label('{}-Device: {}'.format(device, self.dev_number+1), layout=self.t_layout)
        self.b_orientation = RadioButtons(options=['VCP','HCP'], value='HCP', description='Orientation', orientation='horizontal')
        if device == 'MEMIS':
            freq_options = [3200, 9600, 41600]
            spacing_options = [0.5, 1, 1.7, 3.7]
            self.plt_label =  "MEMIS"
            self.plt_linestyle = "-"
        elif device == 'CMD-Mini Explorer':
            freq_options = [30000]
            spacing_options = [0.32, 0.71, 1.18]
            self.plt_label =  "CMD-Mini Explorer"
            self.plt_linestyle = "--"
        elif device == 'CMD Explorer':
            freq_options = [30000]
            spacing_options = [1.48, 2.82, 4.49]
            self.plt_label =  "CMD Explorer"
            self.plt_linestyle = ":"
        else:
            raise ValueError('Device-Name not recognised')

        self.s_freq = SelectionSlider(options=freq_options, value=freq_options[0], description='Frequency', readout=True)
        self.s_spacing = SelectionSlider(options=spacing_options, value=spacing_options[0], description='Spacing', readout=True)
        self.device = HBox([self.l_label, self.b_orientation, self.s_freq, self.s_spacing])
    
    @property
    def coil_name(self):
        return '{orientation}{spacing}f{frequency}h{height}'.format(orientation=self.b_orientation.value, spacing=str(self.s_spacing.value), frequency=self.s_freq.value, height='0')

    @coil_name.setter
    def coil_name(self, name=''):
        raise ValueError('Dont change the name!!!')

# Coil configurations for different instruments
# memis_coils = ["HCP0.5f3200h0","HCP0.5f9600h0","HCP0.5f41600h0",
#         "HCP1f3200h0","HCP1f9600h0","HCP1f41600h0",
#         "HCP1.7f3200h0","HCP1.7f9600h0","HCP1.7f41600h0",
#         "HCP3.7f3200h0","HCP3.7f9600h0","HCP3.7f41600h0"]

# CMD_mini_coils = ["HCP0.32f30000h0","HCP0.71f30000h0","HCP1.18f30000h0"]
# CMD_expl_coils = ["HCP1.48f10000h0","HCP2.82f10000h0","HCP4.49f10000h0"]


#%%

class InteractiveForwardEMI():
    def __init__(self):
        self.wd = Path.cwd()
        self.wd = self.wd / "EMI"
        self.figures_path = self.wd / "results" / "figures"
        self.models_path = self.wd / "results" / "models"
        self.responses_path = self.wd / "results" / "responses"
        self.project_name = 'EMI_project'
        self.author = Path.home().name
        self.__layer_number = 3
        self._layout_l = Layout(width='470px')
        self._layout_s = Layout(width='650px')
        self._style_heading = dict(font_weight='bold', font_size='20px')
        
        self.__layer_list = []
        self.__sigma_a_list = []
        self.__thk_list = []
        self.__device_list = []

        self.vmin, self.vmax = 0.9, 500 # Plotting limits

        self.__memis_nr = 12
        self.__cmd_mini_nr = 3
        self.__cmd_expl_nr = 3

        # loop to create the initial layers
        for i in range(self.__layer_number):
            i = InteractiveEMILayer(i)
            self.__layer_list.append(i)
            i.s_thk.observe(self.on_thk_change, names='value')
            i.s_sigma_a.observe(self.on_sigma_a_change, names='value')
            self.__sigma_a_list.append(i.s_sigma_a.value)
            self.__thk_list.append(i.s_thk.value)


    def start(self):
        '''
        initialising the UI
        '''
        self.initialize_model()
        self.layer_ui()
        self.save_import_ui()
        self.settings_ui()
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

    @property
    def memis_nr(self):
        """
        Number of MEMIS coils in the model.
        """
        return self.__memis_nr
    
    @memis_nr.setter
    def memis_nr(self, number):
        """
        Setter for the memis_nr property.
        """
        print('MEMIS number', number)
        if not isinstance(number, int):
            print('Number of MEMIS coils must be an integer')
        elif number < 0:
            print('Number of MEMIS coils must be positive')
        else:
            while self.memis_nr != number:
                if self.memis_nr < number:
                    self.on_add_device('MEMIS')
                elif self.memis_nr > number:
                    self.on_remove_device('MEMIS')

    @property
    def cmd_mini_nr(self):
        """
        Number of CMD-Mini coils in the model.
        """
        return self.__cmd_mini_nr
    
    @cmd_mini_nr.setter
    def cmd_mini_nr(self, number):
        """
        Setter for the cmd_mini_nr property.
        """
        if not isinstance(number, int):
            print('Number of CMD-Mini coils must be an integer')
        elif number < 0:
            print('Number of CMD-Mini coils must be positive')
        else:
            while self.cmd_mini_nr != number:
                if self.cmd_mini_nr < number:
                    self.on_add_device('CMD-Mini Explorer')
                elif self.cmd_mini_nr > number:
                    self.on_remove_device('CMD-Mini Explorer')

    @property
    def cmd_expl_nr(self):
        """
        Number of CMD Explorer coils in the model.
        """
        return self.__cmd_expl_nr
    
    @cmd_expl_nr.setter
    def cmd_expl_nr(self, number):
        """ 
        Setter for the cmd_expl_nr property.
        """
        if not isinstance(number, int):
            print('Number of CMD Explorer coils must be an integer')
        elif number < 0:
            print('Number of CMD Explorer coils must be positive')
        else:
            while self.cmd_expl_nr != number:
                if self.cmd_expl_nr < number:
                    self.on_add_device('CMD Explorer')
                elif self.cmd_expl_nr > number:
                    self.on_remove_device('CMD Explorer')

        # layer UI
    def layer_ui(self):    
        self.header = VBox([Label('Subsurface Model', layout=self._layout_l, style=self._style_heading), 
                            HBox([Label('Thickness of each layer in meters', layout=self._layout_l), 
                            Label('Condictivity of each layer in mSiemens per meter', layout=self._layout_l)])])
        
        # adding setting changes in the ui
        self.b_add_layer = Button(description='Add layer', button_style='success')
        self.b_add_layer.on_click(self.on_add_layer)
        self.b_remove_layer = Button(description='Remove layer', button_style='danger')
        self.b_remove_layer.on_click(self.on_remove_layer)
            

    def settings_ui(self): 
        device_list = []
        for i in range(self.memis_nr):
            i = InteractiveCoil(i, 'MEMIS')
            i.b_orientation.observe(self.on_change, names='value')
            i.s_freq.observe(self.on_change, names='value')
            i.s_spacing.observe(self.on_change, names='value')
            device_list.append(i)
        self.__device_list.append(device_list)
        device_list = []
        for i in range(self.cmd_mini_nr):
            i = InteractiveCoil(i, 'CMD-Mini Explorer')
            i.b_orientation.observe(self.on_change, names='value')
            i.s_freq.observe(self.on_change, names='value')
            i.s_spacing.observe(self.on_change, names='value')
            device_list.append(i)

        self.__device_list.append(device_list)
        device_list = []
        for i in range(self.cmd_expl_nr):
            i = InteractiveCoil(i, 'CMD Explorer')
            i.b_orientation.observe(self.on_change, names='value')
            i.s_freq.observe(self.on_change, names='value')
            i.s_spacing.observe(self.on_change, names='value')
            device_list.append(i)
        self.__device_list.append(device_list)
    
    def on_change(self, change):
        pass

    def show_settings(self):
        devices = []
        for sublist in self.__device_list:
            for device in sublist:
                devices.append(device)
        devices = VBox([device.device for device in devices])
        self.settings = VBox([Label('Device Settings', layout=self._layout_l, style=self._style_heading), devices])


    # save import buttons
    def save_import_ui(self):
        self.b_save_fig = Button(description='Save Figure', button_style='info')
        self.b_save_fig.on_click(self.save_fig)
        self.b_save_model = Button(description='Save Model', button_style='info')
        self.b_save_model.on_click(self.save_model)
        self.b_save_all = Button(description='Save All', button_style='info')
        self.b_save_all.on_click(self.save_all)


    # display boxes for convenieces sake
    def renew_ui(self):
        self.show_settings()
        self.slider_display = VBox([layer.layer for layer in self.__layer_list])
        self.save_import = VBox([HBox([self.b_save_fig, self.b_save_model, self.b_save_all])])
        self.ui = VBox([self.settings, self.header, self.slider_display,self.save_import])

    # initial display of the ui
    def display_ui(self):
        display(self.ui)

    # initialisation of the plot - first solving of the model
    def initialize_model(self):
        self.model = np.column_stack((self.__thk_list, self.__sigma_a_list))


    def update(self):
        '''function to update the model and the plot for all changes in the sliders'''
        clear_output(wait=False)
        for i in range(len(self.__layer_list)):
            self.__sigma_a_list[i] = self.__layer_list[i].s_sigma_a.value
            self.__thk_list[i] = self.__layer_list[i].s_thk.value
        self.model = np.column_stack((self.__thk_list, self.__sigma_a_list))
        self.renew_ui()
        self.display_ui()
        self.fig, self.ax, self.figs, self.axs = self.run()

    def show(self):
        '''function to show the plot'''
        clear_output(wait=False)
        for i in range(len(self.__layer_list)):
            self.__sigma_a_list[i] = self.__layer_list[i].s_sigma_a.value
            self.__thk_list[i] = self.__layer_list[i].s_thk.value
        self.model = np.column_stack((self.__thk_list, self.__sigma_a_list))
        self.fig, self.ax = self.run()


    # observation corresponding functions
    def on_thk_change(self, change):
        pass

    def on_sigma_a_change(self, change):
        pass
    def on_add_layer(self, click):
        self.__layer_list.append(InteractiveEMILayer(self.__layer_number))
        self.__layer_list[-1].s_thk.observe(self.on_thk_change, names='value')
        self.__layer_list[-1].s_sigma_a.observe(self.on_sigma_a_change, names='value')
        self.__sigma_a_list.append(self.__layer_list[-1].s_sigma_a.value)
        self.__thk_list.append(self.__layer_list[-1].s_thk.value)
        self.__layer_number +=1

    def on_remove_layer(self, click):
        if len(self.__layer_list) > 2:
            self.__layer_list.pop(-1)
            self.__sigma_a_list.pop(-1)
            self.__thk_list.pop(-1)
            self.__layer_number -= 1
        else:
            raise ValueError('You need to have at least 2 layers')
        
    def on_add_device(self, device_type):
        if device_type == 'MEMIS':
            i = 0
            self.__memis_nr +=1
        elif device_type == 'CMD-Mini Explorer':
            i = 1
            self.__cmd_mini_nr +=1
        elif device_type == 'CMD Explorer':
            i = 2
            self.__cmd_expl_nr +=1
        print(len(self.__device_list[i]))
        self.__device_list[i].append(InteractiveCoil(len(self.__device_list[i]), device_type))
        print(len(self.__device_list[i]))


    def on_remove_device(self, device_type):
        if device_type == 'MEMIS':
            i = 0
            self.__memis_nr -=1
        elif device_type == 'CMD-Mini Explorer':
            i = 1
            self.__cmd_mini_nr -=1
        elif device_type == 'CMD Explorer':
            i = 2
            self.__cmd_expl_nr -=1
        print(len(self.__device_list[i]))
        self.__device_list[i].pop(-1)
        print(len(self.__device_list[i]))
        

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
        for path in [self.figures_path, self.models_path, self.responses_path]:
            path.mkdir(parents=True, exist_ok=True)

    def save_fig(self, click):
        self.create_folders()
        filename = self.filename_input()
        self.fig.savefig(self.figures_path / '{}.png'.format(filename))

    def save_model(self, click):
        self.create_folders()
        filename = self.filename_input()
        model = pd.DataFrame({'thickness':self.__thk_list, 'conductivity':self.__sigma_a_list})
        model.to_csv(self.models_path / '{}.csv'.format(filename))

    def save_all(self, click):
        self.create_folders()
        self.b_save_fig.click()
        self.b_save_model.click()

    def filename_input(self):
        return '{project}_{author}_{time}'.format(project=self.project_name,
            author=self.author, time=datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    def on_update(self, click):
        self.update()


    def run(self):
        # Define theoretical model (with 0 m surface depth for plotting)
        thk_list = [0]
        for i in self.__thk_list:
            thk_list.append(thk_list[-1]+i)
        sigma_a_list = self.__sigma_a_list.copy()
        sigma_a_list.append(sigma_a_list[-1])
        depths_tem = np.array(thk_list)
        models_tem = np.array(sigma_a_list) 

        # Prepare model for emagpy (remove 0 depth and reshape)
        depths_ = depths_tem[1:].reshape(1, -1)
        models_ = models_tem.reshape(1, -1)

        # Inversion parameters
        forward_model = "FSlin" # either "FSlin" or "CS"

        fig, ax = plt.subplots(figsize=(10,10))
        
        figs, axs = plt.subplots(figsize=(10,10))
        
        #loop over all devices
        for i in range(len(self.__device_list)): 
            if len(self.__device_list[i]) == 0:
                continue

            # Get device parameters
            label = self.__device_list[i][0].plt_label
            colors = self.generate_colors(len(self.__device_list[i]))
            coils = [device.coil_name for device in self.__device_list[i]]
            linestyle = self.__device_list[i][0].plt_linestyle
                            
            # ---------- Forward Modeling ----------
            fw_problem = Problem()
            fw_problem.setModels([depths_], [models_])
            fw_results = fw_problem.forward(forwardModel=forward_model, coils=coils)
            df_results = fw_problem.surveys[0].df
            df_results = df_results.drop(labels=["x","y","elevation"],axis=1)

            # Save forward modeling results
            fw_dir = self.responses_path / f"fw_{label}"
            fw_dir.mkdir(parents=True, exist_ok=True)  # Creates the directory if it doesn't exist

            fw_output_file = fw_output_file = Path(fw_dir) / f"fw_{label}.csv"
            
            df_results.to_csv(fw_output_file, index=False) 

            depths_tem_extended = np.append(depths_tem.flatten(), depths_tem.flatten()[-1] + 10) #add one step at 10m
            models_tem_extended = np.append(models_tem.flatten(), models_tem.flatten()[-1]) #with the same value as the last one
            
            # Sensitivity calculation
            depths_sens = np.arange(0, 10, 0.1)[1:].reshape(1, -1)
            models_sens = np.ones(len(depths_sens[0]) + 1).reshape(1, -1) * 150
            sensitivity = Problem()
            sens = sensitivity.computeSens(forwardModel=forward_model, coils=coils, models=[models_sens], depths=[depths_sens])

            S = sens[0]  # Nsample x Ncoils x Nprofiles
            S2 = S[::-1, :, :]
            cumS = np.cumsum(S2, axis=0)[::-1, :, :]
            cumS = cumS / np.max(cumS, axis=0)  # normalize so that top is 1

            j = 0
            for i, c, in zip(df_results.columns, colors):
                value=0
                if type(df_results.iloc[0][i]) == pd.Series:
                    value = df_results.iloc[0][i][0]
                else:
                    value = df_results.iloc[0][i]
                ax.axvline(x=value, linestyle=linestyle, color=c, lw=1, zorder=0, label='{coiltype}: {coilname}'.format(coiltype=label, coilname=i)) # vertical line at x=fw_model

                # Plotting the sensitivity
                axs.plot(cumS[:,j][:-1], np.transpose(depths_sens), label='{coiltype}: {coilname}'.format(coiltype=label, coilname=coils[j]), color=c, linestyle=linestyle)
                j += 1

        ax.step(models_tem_extended.flatten(), depths_tem_extended.flatten(),
                where="pre", color='black', linestyle="--",
                label='theoretical model', marker='.',zorder=1)

        # Final plot formatting
        ax.grid(True, zorder=0)
        ax.legend(loc="right", fontsize=8)
        ax.set_xlim(self.vmin,max(models_tem)*1.1)
        ax.set_xscale("log")
        ax.set_xlabel('$\sigma$ (mS/m)', fontsize=16)
        ax.set_ylabel("Depth (m)", fontsize=16)
        ax.set_ylim(max(thk_list), 0)
        ax.set_yticks([0,1,2,3,4,5],
                    labels=['0','1','2','3','4','5'])
        

        axs.invert_yaxis()
        axs.set_xlabel("normalised local sensitivity")
        axs.set_ylabel("depth (m)")
        axs.set_ylim([6,0])
        axs.grid(True)
        axs.legend()
        axs.set_title("Sensitivity Analysis: homogenous 150 mS/m")

        return fig, ax, figs, axs


    def generate_colors(self, num_colors):
        """
        Generiert eine Liste von möglichst verschiedenen Farbcodes in Hex-Format.

        Parameters:
            num_colors (int): Anzahl der gewünschten Farbcodes.

        Returns:
            list: Liste von Farbcodes als Hex-Werte.
        """
        cmap = plt.get_cmap('tab20')  # Verwende eine Colormap mit vielen verschiedenen Farben
        colors = [cmap(i / num_colors) for i in range(num_colors)]  # Normalisiere die Farben
        hex_colors = [rgb2hex(color[:3]) for color in colors]  # Konvertiere in Hex
        return hex_colors



