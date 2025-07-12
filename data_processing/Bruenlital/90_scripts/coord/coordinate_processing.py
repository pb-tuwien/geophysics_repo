#%% Import class

from gp_tools.core import CoordinateHandler
import pandas as pd
from pathlib import Path

dir_path = Path(__file__).parents[3] / 'Bruenlital/50_coord'
raw_path = dir_path / '20250710_coords_raw.csv'
corners_path = dir_path / '20250710_coords_corners.csv'
centers_path = dir_path / '20250710_coords_centers.csv'
#%% Reading coordinates from txt-File

# raw_coords = pd.read_csv(raw_path)
# raw_coords['Antenna height'] = 2.0
# raw_coords = raw_coords[['Name', 'Longitude', 'Latitude', 'EllipsoidHeight']]
# raw_coords.rename(columns={'EllipsoidHeight': 'Ellipsoidal height'}, inplace=True)
# raw_coords.to_csv(raw_path, sep=',', index=False)

#%% Create CSV with all the corner coordinates

handler = CoordinateHandler()
handler.read(raw_path)
handler.rename_points({'-': '_'})
handler.reproject(reproj_key='wgs84_utm32n')
handler.write(corners_path)

# %% Find centres of soundings

handler = CoordinateHandler()
handler.read(raw_path)
handler.rename_points({'-': '_'})
corner_coordinates = handler.coordinates

soundings_names = [f'B{i:03d}' for i in range(1, 46)]

profile_corners = [[f'p1_{i+j:03d}' for j in range(4)] for i in range(1, 1+9*2, 2)]

all_corner_points = []
for area_number in range(1, 7):
    all_corner_points += [[f'a{area_number}_{i+j:03d}' for j in range(4)] for i in range(1, 1+6*2, 2)]

all_corner_points += profile_corners

center_coordinates = pd.DataFrame()

for sounding, sounding_corner in zip(soundings_names, all_corner_points):
    sounding_coordinates = corner_coordinates[corner_coordinates['Name'].isin(sounding_corner)]
    sounding_coordinates = sounding_coordinates[['Longitude', 'Latitude', 'Ellipsoidal height', 'Antenna height']]
    one_sounding = sounding_coordinates.sum() / 4
    one_sounding['Name'] = sounding
    center_coordinates = pd.concat([center_coordinates, one_sounding.to_frame().T])

center_coordinates = center_coordinates[['Name', 'Longitude', 'Latitude', 'Ellipsoidal height', 'Antenna height']]

handler.coordinates = center_coordinates
handler.reproject(reproj_key='wgs84_utm32n')
handler.write(centers_path)