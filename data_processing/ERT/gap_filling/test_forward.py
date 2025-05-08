#%% Imports
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.impute import KNNImputer
import numpy as np
from resipy import Project
import matplotlib.pyplot as plt

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
# k.showPseudo()

#%% Reconstruct Data

reduction_factor = 0.2
n_neighbors = 6
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
def showPseudoGen(df, column='rhoa', vmin=None, vmax=None, label=None, ax=None):
    """Plot a pseudosection
    :param proj: resipy project
    :param column: column name to plot
    :param vmin: colorbar minimum value
    :param vmax: colorbar maximum value
    :param label: colorbar label
    :param ax
    """

    values = df[column].values
    xpos = df['x'].values
    ypos = df['y'].values

    if label is None:
        if 'rhoa' in column:
            label = r'$\rho_a$ ($\Omega$m)'
        elif 'phia' in column:
            label = r'$\phi$ [mrad]'

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    im = ax.scatter(xpos, ypos, c=values, s=50, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)
    cbar.set_label(label)

    ax.invert_yaxis()
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Pseudo depth (m)')
    if ax is None:
        return fig

fig, ax = plt.subplots(1,3, figsize=(18,5))
ax[0].set_title('Original')
ax[1].set_title(f'Reduced \nby {reduction_factor*100} %')
ax[2].set_title(f'Reconstructed\nwith {n_neighbors} neighbors and \nhorizontal smoothing factor of {horizontal_weight}')

showPseudoGen(data_df, ax=ax[0])

delete_rows = np.random.choice(
    data_df.index,
    size=int(0.2*len(data_df)),
    replace=False
)

reduced_df = data_df.copy()
reduced_df['rhoa'][delete_rows] = np.nan

showPseudoGen(reduced_df, ax=ax[1])


def distance(x, y, missing_values=np.nan):
    x = x[:2].reshape(1, -1)
    y = y[:2].reshape(1, -1)
    return euclidean(x[0], y[0])

def weighted_distance(x, y, missing_values=np.nan):
    x_distance = (x[0] - y[0])
    y_distance = (x[1] - y[1])
    return np.sqrt(x_distance**2 + (y_distance**2)*horizontal_weight)

test_data = reduced_df.values
imputer = KNNImputer(n_neighbors=n_neighbors, metric=weighted_distance, weights='distance')
imputed_data = imputer.fit_transform(test_data)

imputed_df = pd.DataFrame(data=imputed_data, columns=data_df.columns)
showPseudoGen(imputed_df, ax=ax[2])
fig.tight_layout()

misfit = np.sqrt(((data_df['rhoa'].values - imputed_df['rhoa'].values)**2).sum()) / len(data_df)
print(misfit)