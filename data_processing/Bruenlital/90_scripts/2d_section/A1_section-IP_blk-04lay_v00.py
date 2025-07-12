#%% Imports

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

#%% Inputs

# Name of the 2D section
profile_name  = '20250710_A1'

# Inversion type
inversion_type = 'pyGIMLi'

# Version of the 2D section
section_version = 'v00'

# Coordinates (relative paths from .../30_inv-results/{inversion_type}/)
coordinate_file = '50_coord/20250710_coords_centers.csv'

# Column names of DataFrame
name_column = 'Name'
easting_column = 'Easting'
northing_column = 'Northing'
elevation_column = 'Elevation'

# Consider elevation
# elevation_column = False

#

# Chosen soundings (relative paths from .../30_inv-results/{inversion_type}/)
sounding_files = [
    'A1-IP_blk-04lay/20250710_A1/v00_tr5-100us/B001/csv/invrun000_B001.csv',
    'A1-IP_blk-04lay/20250710_A1/v00_tr5-100us/B002/csv/invrun000_B002.csv',
    'A1-IP_blk-04lay/20250710_A1/v00_tr5-100us/B003/csv/invrun000_B003.csv',
    'A1-IP_blk-04lay/20250710_A1/v00_tr5-100us/B004/csv/invrun000_B004.csv',
    'A1-IP_blk-04lay/20250710_A1/v00_tr5-100us/B005/csv/invrun000_B005.csv',
    'A1-IP_blk-04lay/20250710_A1/v00_tr5-100us/B006/csv/invrun000_B006.csv'
]

# Index of the first and last sounding of the line
start_index = 0
end_index = -1

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
profile_length = line_length = np.linalg.norm(profile_vector)
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
inv_columns = ['Name', 'depth(m)', 'rho(Ohmm)', 'mpa(rad)', 'tau_p(s)', 'c()']

# Reading all the results
inversion_data = [pd.read_csv(data_path) for data_path in sounding_paths]

# Adding them to one DataFrame
inversion_df = pd.DataFrame()
for name, df in zip(sounding_names, inversion_data):
    df['Name'] = name
    inversion_df = pd.concat([inversion_df, df[inv_columns]])

# Reset index
inversion_df.reset_index(drop=True, inplace=True)

# Merging with the coordinates
subsurface_model =  inversion_df.merge(sounding_coordinates, how='left', on='Name')

#%% Create plots

fig, ax = plt.subplots(2,2, figsize=(16, 8))
ax = ax.ravel()
sc = ax[0].scatter(
    subsurface_model['Distance'], subsurface_model['depth(m)'], 
    c=subsurface_model['rho(Ohmm)'], cmap='viridis', s=100, 
    label='Resistivity', marker='s', 
    vmin=subsurface_model['rho(Ohmm)'].min(), vmax=subsurface_model['rho(Ohmm)'].max())
cbar = fig.colorbar(sc, ax=ax[0])
ax[0].invert_yaxis()

def inversion_plot_2D_section(lam=183, lay_thk={0:1, 5:1.5, 15:2}):


    fig, ax = plt.subplots(figsize=(10, 6))
    # Plotting the data
    # Customize colorbar to match the data values
    model_unit_list, dist_list, thks_list = [], [], []
    thks = [1, 2, 3, 4, 5, 6.5, 8.0, 9.5, 11.0]
    for sounding, dist in zip(range(11), distances):
        model_unit = resistivities[sounding]
        dist = [dist for _ in range(len(thks))]
        for i, j, k in zip(model_unit, dist, thks):
            model_unit_list.append(i)
            dist_list.append(j)
            thks_list.append(k)

        
    sc = ax.scatter(dist_list,thks_list, c=model_unit_list, cmap='viridis', s=150, label='pyGIMLI')
    
    cbar = fig.colorbar(sc, ax=ax)
    # cbar.set_ticks(np.linspace(np.min(unit_values), np.max(unit_values), num=6),
    #                 fontsize=14)  # Set ticks according to data range
    cbar.set_label(r'$\rho$ ($\Omega$m)', fontsize=16)  # Set colorbar label
    ax.set_xlabel("Horizontal Distance (m)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(f"2D Crosssection with Multiple Soundings\n$\lambda$ = 183", fontsize=22, fontweight='bold')
    ax.set_ylim(-2, 12)
    for x_val, y_val, label in zip(distances, [0 for _ in range(len(distances))], intersections['Name']):
        ax.text(x_val, y_val, label, fontsize=10, color="black", ha="center", va="bottom", rotation=45)
   
    ax.invert_yaxis()  # Invert y-axis to show depth increasing downward
    fig.savefig('inversion_2Dsection_scatter_{}.png'.format('rhoa'))


# %%
# inversion_plot_2D_section()