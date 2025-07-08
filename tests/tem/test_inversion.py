#%% Import class

import gp_tools.tem.survey_tem as st
import numpy as np

# %% Test Survey

tem_data = './test_data/test_data.tem'
tem_coords = './test_data/test_coords.csv'

rename_points = {
    'M11': 'M011', 'M12': 'M012', 'M13': 'M013', 'M14': 'M014',
    'M15z': 'M015', 'M16': 'M016', 'M17': 'M017', 'M18': 'M018',
    'M19': 'M019', 'M20': 'M020', 'M21': 'M021', 'M22': 'M022',
    'M23': 'M023', 'M24': 'M024', 'M25': 'M025', 'M26': 'M026',
    'M27': 'M027', 'M28': 'M028', 'M29': 'M029', 'M30': 'M030',
    'M31': 'M031', 'M32': 'M032', 'M33': 'M033', 'M34': 'M034',
    'M35': 'M035', 'M36': 'M036', 'M37': 'M037', 'M38': 'M038',
    'M39': 'M039', 'M40': 'M040', 'M41': 'M041', 'M42': 'M042',
    'M43': 'M043', 'M44': 'M044', 'M45': 'M045'
    }
parsing_coords = {'EP1': 'Mtest', 'TEM_test': 'Mtest'}

survey = st.SurveyTEM('./test_dir')
# survey.coords_read(coords=tem_coords)
# survey.coords_rename_points(rename_dict=rename_points)
# survey.coords_sort_points()
# survey.coords_reproject()
# survey.coords_extract_save()
# survey.data_read(data=tem_data)
survey.coords_read()
survey.data_read()
survey.data_preprocess(parsing_dict=parsing_coords)

# survey.plot_inversion(
#     lam=89,
#     subset=['M024'],
#     layer_type='dict',
#     layers={0:1, 5:1.5},
#     max_depth=20,
#     filter_times=(8, 150),
#     fname='test.png'
#     )

survey.optimised_inversion_plot(
    sounding='M024',
    lam=89,
    layer_type='dict',
    layers={0: 1, 5: 1.5},
    max_depth=20,
    test_range=(10, 1000, 20),
    filter_times=(8, 150),
    fname=False
)
