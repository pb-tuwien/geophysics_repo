#%% Imports
import warnings
import pandas as pd
warnings.filterwarnings('ignore')
import numpy as np # numpy for electrode generation
from resipy import Project

#%% Forward Model

k = Project(typ='R2', dirname='.')

# defining electrode array
elec = np.zeros((24, 3))
elec[:,0] = np.arange(0, 24*0.5, 0.5)
k.setElec(elec)

# creating mesh
k.createMesh()

# add region
k.addRegion(np.array([[2,-0.3],[2,-2],[3,-2],[3,-0.3],[2,-0.3]]), 50)

# define sequence
k.createSequence([('dpdp1', 1, 10)])

# forward modelling
k.forward(noise=0.0)

# show the initial and recovered section
k.showResults(index=0) # initial
k.showPseudo()

#%% Retrieve Data

apparent_res = k.surveys[0].df['app'].values.copy()
xpos, _, ypos = k.surveys[0]._computePseudoDepth()

data_df = pd.DataFrame(
    {
        'x': xpos,
        'y': ypos,
        'rhoa': apparent_res
    }
)
