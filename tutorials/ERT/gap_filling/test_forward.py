#%% Imports
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from resipy import Project
from gp_tools.ert.gap_filling.knn_imputer import KNNfiller
from gp_tools.ert.utils import plot_pseudosection

#%% Forward Model
k = Project(typ='R2', dirname='.')

# defining electrode array
number_electrodes = 64
seperation_electrodes = 1 # in Meters
elec = np.zeros((number_electrodes, 3))
elec[:,0] = np.arange(0, number_electrodes*seperation_electrodes, seperation_electrodes)
k.setElec(elec)

# creating mesh
k.createMesh(typ='trian', show_output=False, res0=200)

# add region
k.addRegion(np.array([[2,-4],[2,-10],[50,-10],[50,-4],[2,-4]]), 50)

# define sequence
k.createSequence([('dpdp1', 2, number_electrodes)])
print(k.sequence)

# forward modelling
k.forward(noise=0.0)

# show the initial and recovered section
k.showResults(index=0) # initial

#%% Reconstruct Data

reduction_factor = 0.2
neighbors = 6
horizontal_weight = 7

apparent_res = k.surveys[0].df['app'].values.copy()
xpos, _, ypos = k.surveys[0]._computePseudoDepth()

data_df = pd.DataFrame(
    {
        'x': xpos,
        'y': ypos,
        'rhoa': apparent_res
    }
)

fig, ax = plt.subplots(1,3, figsize=(18,5))
ax[0].set_title('Original')
ax[1].set_title(f'Reduced \nby {reduction_factor*100} %')
ax[2].set_title(f'Reconstructed\nwith {neighbors} neighbors and \nhorizontal smoothing factor of {horizontal_weight}')

plot_pseudosection(data_df, ax=ax[0])

delete_rows = np.random.choice(
    data_df.index,
    size=int(0.2*len(data_df)),
    replace=False
)

reduced_df = data_df.copy()
reduced_df['rhoa'][delete_rows] = np.nan

plot_pseudosection(reduced_df, ax=ax[1])

test_data = reduced_df.values
imputer = KNNfiller(neighbors=neighbors, horizontal_weight=horizontal_weight)
imputed_data = imputer.fit_transform(test_data)
imputed_df = pd.DataFrame(data=imputed_data, columns=data_df.columns)

plot_pseudosection(imputed_df, ax=ax[2])
fig.tight_layout()

misfit = np.sqrt(((data_df['rhoa'].values - imputed_df['rhoa'].values)**2).sum()) / len(data_df)
print(misfit)
