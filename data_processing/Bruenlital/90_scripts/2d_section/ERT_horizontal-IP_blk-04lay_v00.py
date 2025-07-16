#%% Imports

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union
import matplotlib
import matplotlib.cm
import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.colors as mcolor

#%% Inputs

# Name of the 2D section
profile_name: str  = '20250710_ERT_horizontal'

# Inversion type
inversion_type: str = 'pyGIMLi'

# Version of the 2D section
profile_version: str = 'v00'

# Coordinates (relative paths from .../30_inv-results/{inversion_type}/)
coordinate_file: str = '50_coord/20250710_coords_centers.csv'

# Column names of DataFrame
name_column: str = 'Name'
easting_column: str = 'Easting'
northing_column: str = 'Northing'
elevation_column: Union[str, bool] = 'Elevation'

# Consider elevation
# elevation_column = False

# How thick the last layer should be plotted
add_bottom: Union[int, float] = 3 # in (m)

# Chosen soundings (relative paths from .../30_inv-results/{inversion_type}/)
sounding_files: list = [
    'A4-IP_blk-04lay/20250710_A4/v00_tr5-100us/B021/csv/invrun005_B021.csv',
    'A3-IP_blk-04lay/20250710_A3/v00_tr5-100us/B016/csv/invrun005_B016.csv',
    'A2-IP_blk-04lay/20250710_A2/v00_tr5-100us/B009/csv/invrun005_B009.csv',
    'A1-IP_blk-04lay/20250710_A1/v00_tr5-100us/B004/csv/invrun005_B004.csv',
    'A5-IP_blk-04lay/20250710_A5/v00_tr5-100us/B027/csv/invrun005_B027.csv',
    'A6-IP_blk-04lay/20250710_A6/v00_tr5-100us/B034/csv/invrun005_B034.csv',
    'P1-IP_blk-04lay/20250710_P/v00_tr5-100us/B037/csv/invrun005_B037.csv',
    'P1-IP_blk-04lay/20250710_P/v00_tr5-100us/B038/csv/invrun005_B038.csv'
]

# Index of the first and last sounding of the line
start_index: int = 0
end_index: int = -1

#%% Paths

root_dir = Path(__file__).parents[3] / 'Bruenlital' # Path to root directory
source_path = root_dir / '30_inv-results' / inversion_type # Path to inversion results
result_path = root_dir / '70_2D-sections' / profile_name # Path to where the result-plot should be saved
coordinate_path = root_dir / coordinate_file # Path to the coordinate file
sounding_paths = [source_path / sounding for sounding in sounding_files] # Paths to the inversion results

# Make sure the target directory exists
if not result_path.exists():
    result_path.mkdir(parents=True, exist_ok=True)
# Checking if all inversion result-files exist
if not all([p.exists() for p in sounding_paths]):
    missing = [p for p in sounding_paths if not p.exists()]
    raise FileNotFoundError(f'Inversion results not found: {missing}')

print('All paths found.')

#%% Read coordinates and make sure all necessary columns exist

coordinates = pd.read_csv(coordinate_path) # DataFrame with all coordinates

# Checking if necessary columns exist
if name_column not in coordinates.columns:
    raise KeyError('No name column found in coordinates')
if easting_column not in coordinates.columns:
    raise KeyError('No easting column found in coordinates')
if northing_column not in coordinates.columns:
    raise KeyError('No northing column found in coordinates')
if elevation_column not in coordinates.columns:
    print('No elevation column found in coordinates. Not considering elevation.')
    elevation_column = False

# Name of the sounding, as they are called in the coordinates
sounding_names = [name.stem.split('_')[-1] for name in sounding_paths]
# Only keeping the coordinates of the used soundings
sounding_coordinates = coordinates[coordinates['Name'].isin(sounding_names)].copy()

# Check if no coordinates are missing
if len(sounding_coordinates) != len(sounding_files):
    for name in sounding_names:
        if not any(coordinates['Name'].isin([name])):
            print(f'The sounding "{name}" is missing from the coordinates')
    raise ValueError('Coordinates for some soundings not found.')

print('Coordinates found.')


#%% Reproject to line-distance
starting_name, ending_name = sounding_names[start_index], sounding_names[end_index]
print(f'Profile "{profile_name}": Starts with "{starting_name}" and ends with "{ending_name}".')

# start and end points of the line
starting_point = sounding_coordinates[sounding_coordinates['Name'] == starting_name][[easting_column, northing_column]].values
ending_point = sounding_coordinates[sounding_coordinates['Name'] == ending_name][[easting_column, northing_column]].values

# Calculate the direction vector of the line
profile_vector = ending_point - starting_point
profile_length = np.linalg.norm(profile_vector)
profile_unit_vector = profile_vector / profile_length

# Function to project a point onto the line and calculate the distance from the start
def project_point(point):
    point_vector = np.array([point[easting_column], point[northing_column]]) - starting_point
    projection_length = np.dot(point_vector, profile_unit_vector.T)  # Scalar projection
    return projection_length.item()

# Apply the projection function to each point
sounding_coordinates['Distance'] = sounding_coordinates.apply(project_point, axis=1)
# Sort the points by their distance along the line
sounding_coordinates.sort_values(by='Distance', inplace=True)

#%% Read Data and create a Dataframe with all results
# Needed columns ("Name" will be added)
inv_columns = ['Name', 'layer_number', 'depth(m)', 'layer_thickness', 'rho(Ohmm)', 'mpa(rad)', 'tau_p(s)', 'c()']

# Reading all the results
inversion_data = [pd.read_csv(data_path) for data_path in sounding_paths]

# Adding them to one DataFrame
inversion_df = pd.DataFrame()
for name, df in zip(sounding_names, inversion_data):
    df['Name'] = name
    depths = df['depth(m)'].copy()
    depths[depths == 0] = add_bottom
    df['layer_thickness'] = depths
    df['depth(m)']= np.cumsum(depths)
    df['layer_number'] = np.arange(len(df))
    inversion_df = pd.concat([inversion_df, df[inv_columns]])

# Reset index
inversion_df.reset_index(drop=True, inplace=True)

# Merging with the coordinates
subsurface_model =  inversion_df.merge(sounding_coordinates, how='left', on='Name')

#%% Plotting function

def plot_2d_section(elevation: Union[str, bool], mode: str = 'scatter') -> matplotlib.figure.Figure:
    fig, ax = plt.subplots(2,2, figsize=(16, 8))
    ax = ax.ravel()

    mode_list = ['scatter', 'bar']

    model_df = subsurface_model.copy()

    if 'model_depth' not in model_df.columns:
        if elevation:
            model_df['model_depth'] = model_df[elevation] - model_df['depth(m)']
        else:
            model_df['model_depth'] = model_df['depth(m)']

    model_parameters = ['rho(Ohmm)', 'mpa(rad)', 'tau_p(s)', 'c()']

    first_layer_df = model_df[model_df['layer_number'] == 0].copy()
    first_layer_df.reset_index(drop=True, inplace=True)

    for i, parameter in enumerate(model_parameters):

        y_lower_lim = max(model_df['model_depth'].min() * 0.9, model_df['model_depth'].min() - 10)
        y_upper_lim = min(model_df['model_depth'].max() * 1.1, model_df['model_depth'].max() + 10)


        if mode == 'scatter':
            sc = ax[i].scatter(
                model_df['Distance'], model_df['model_depth'], 
                c=model_df[parameter], cmap='viridis', s=100, 
                label='Resistivity', marker='s', 
                vmin=model_df[parameter].min(), vmax=model_df[parameter].max())
            _ = fig.colorbar(sc, ax=ax[i], label=parameter[parameter.find('('):])

        elif mode == 'bar':
            if elevation:
                bar_bottom = first_layer_df[elevation].copy().values
                ax[i].plot(first_layer_df['Distance'], bar_bottom, c='k', zorder=0, linewidth=0.8)
            else:
                bar_bottom = np.zeros_like(np.arange(len(sounding_files)), dtype='float64')

            norm = mcolor.Normalize(vmin=model_df[parameter].min(), vmax=model_df[parameter].max())
            cmap = matplotlib.colormaps['viridis']

            for _, layer_df in model_df.groupby('layer_number'):
                if elevation:
                    bar_bottom -= layer_df['layer_thickness'].values
                color = cmap(norm(layer_df[parameter].values))
                ax[i].bar(layer_df['Distance'], layer_df['layer_thickness'], 2, bottom=bar_bottom, color=color)
                if not elevation:
                    bar_bottom += layer_df['layer_thickness'].values
            
            _ = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax[i], label=parameter[parameter.find('('):])
        else:
            raise ValueError(f'"{mode}" is an invalid input. Must be from {mode_list}')

        for j in range(len(first_layer_df)):
            name = first_layer_df['Name'][j]
            x_val = first_layer_df['Distance'][j]
            y_val = y_upper_lim if elevation else 0
            ax[i].text(
                x_val, y_val, name, 
                fontsize=10, color="black", 
                ha='center', va="bottom")

        ax[i].set_xlabel("Horizontal Distance (m)")
        ax[i].set_ylabel("Depth (m)")
        ax[i].set_ylim((y_lower_lim, y_upper_lim))
        ax[i].set_title(parameter[:parameter.find('(')], fontweight='bold', fontsize=16, pad = 15.0)
        if not elevation:
            ax[i].set_ylim((0, y_upper_lim))
            ax[i].invert_yaxis()

    fig.suptitle(f'2D-section of the profile "{profile_name}"', fontweight='bold', fontsize=20)
    fig.tight_layout()

    return fig

# %% Create plots

fig1 = plot_2d_section(elevation=elevation_column, mode='scatter')
fig1.savefig(result_path / f'{profile_name}_{profile_version}_scatter_elevation.png', dpi=300)

fig2 = plot_2d_section(elevation=False, mode='scatter')
fig2.savefig(result_path / f'{profile_name}_{profile_version}_scatter.png', dpi=300)

fig3 = plot_2d_section(elevation=elevation_column, mode='bar')
fig3.savefig(result_path / f'{profile_name}_{profile_version}_bar_elevation.png', dpi=300)

fig4 = plot_2d_section(elevation=False, mode='bar')
fig4.savefig(result_path / f'{profile_name}_{profile_version}_bar.png', dpi=300)
