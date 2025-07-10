#%% Import class

from gp_tools.core import CoordinateHandler
import pandas as pd
from pathlib import Path

#%% Reading coordinates from txt-File

raw_path = Path(__file__).parents[3] / 'Bruenlital/50_coord/testing.csv'
raw_coords = pd.read_csv(raw_path)
raw_coords['Antenna height'] = 2.0
# raw_coords = raw_coords[['Name', 'Longitude', 'Latitude', 'EllipsoidHeight']]
# raw_coords.rename(columns={'EllipsoidHeight': 'Ellipsoidal height'}, inplace=True)
raw_coords.to_csv(raw_path, sep=',', index=False)

#%%

handler = CoordinateHandler()
handler.read(raw_path)
# handler.rename_points({'-': '_'})
handler.reproject(reproj_key='wgs84_utm32n')

# %%
