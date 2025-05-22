# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 19:42:19 2021

function library for 1D smooth pygimli inversion of tem data.
uses empymod as forward solver.

based upon pygimli example

@author: lukas aigner @ TU Wien, Research Unit Geophysics
"""

# %% import modules
import numpy as np
import pygimli as pg
from gp_tools.tem.forward import empymod_frwrd

# %% class_lib
class tem_smooth1D_fwd(pg.Modelling):
    """
    class for forward modeling of tem data with pyGIMLi and empymod
    """
    def __init__(self, depths, empy_frwrd):
        """Initialize the model and forward class from empymod."""
        self.dep = depths
        self.mesh1d = pg.meshtools.createMesh1D(len(self.dep))
        self.empy_frwrd = empy_frwrd
        super().__init__()
        self.setMesh(self.mesh1d)

    def response(self, values):
        """Forward response of a given model."""
        return self.empy_frwrd.calc_response(np.column_stack((self.dep, values)), ip_modeltype=None)


class tem_inv_smooth1D(pg.Inversion):
    """
    derived class for the smooth inversion of TEM data
    """
    def __init__(self, setup_device):
        self.test_response = None
        self.start_model = None
        self.fwd = None
        self.nlayer = None
        self.max_depth = None
        self.depth_fixed = None
        self.setup_device = setup_device

        # super(pg.Inversion, self).__init__(**kwargs)
        super().__init__()


    def prepare_fwd(self, depth_vector, start_model, filter_times=None, max_depth=30, times_rx=None):
        """
        method to initialize the forward solver

        Parameters
        ----------
        times_rx : np.Array
            times for the receiver response.
        depth_vector : np.Array
            depth vector for the fixed layer inversion.
        start_model : np.Array
            resistivity vector describing the initial model.
        filter_times : tuple
            (t_min, t_may) in s, filtering range.
        max_depth : int, optional
            maximum depth of the fixed layer vector. The default is 30.

        Returns
        -------
        np.Array
            response of the start model.

        """
        self.max_depth = max_depth  # 4 * self.setup_device['tx_loop']
        self.start_model = start_model
        self.depth_fixed = depth_vector
        self.nlayer = len(self.depth_fixed)
        if len(self.depth_fixed) != len(self.start_model):
            raise ValueError('depth vector and start model have different lengths')
        
        empy_frwrd = empymod_frwrd(setup_device=self.setup_device,
                                   setup_solver=None,
                                   times_rx=times_rx,
                                   time_range=None, device='TEMfast',
                                   nlayer=self.nlayer, nparam=2)

        # self.depth_fixed = np.linspace(0., max_depth, self.nlayer)      #todo: fixed depth vector for inversion, add as parameter
        self.fop = tem_smooth1D_fwd(self.depth_fixed, empy_frwrd)

        self.test_response = self.fop.response(self.start_model)
        
        return self.test_response


    def prepare_inv(self, maxIter=20, verbose=True):
        """
        Method to initialize the inversion

        Parameters
        ----------
        maxIter : int, optional
            maximum number of iterations. The default is 20.
        verbose : Boolean, optional
            show output of pygimli. The default is True.

        Returns
        -------
        None.

        """
        transRho = pg.trans.TransLogLU(1, 1000)
        transData = pg.trans.TransLog()

        self.verbose = verbose

        self.transModel = transRho
        self.transData = transData

        self.maxIter = maxIter
