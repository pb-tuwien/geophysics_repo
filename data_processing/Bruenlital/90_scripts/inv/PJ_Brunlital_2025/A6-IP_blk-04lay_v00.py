# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 16:21:10 2022

script for blocky batch inversion of TEM-IP data with empymod and pygimli
empymod for forward solution and pygimli for inversion framework

adds the time range to the version folder name; this allows for
running versions with different time ranges:
--> it is possible to read the same data file multiple times when
    providing different time ranges

intended for inverting a full dataset with similar parameters
(maybe for slight changes to the number of layers or lambda)


@author: lukas
"""


# %% import modules
import os
import sys
from pathlib import Path
path_to_libs = ("C:/Users/jakob/Documents/Meine Ordner/TU/Geophysik/GP_Lukas/em_modeling/")

if not path_to_libs in sys.path:
    sys.path.append(path_to_libs)
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.backends.backend_pdf import PdfPages

import pygimli as pg
from pygimli.viewer.mpl import drawModel1D

from TEM.library.TEM_frwrd.empymod_frwrd import empymod_frwrd as empyfrwrd
from TEM.library.TEM_frwrd.empymod_frwrd_ip import empymod_frwrd as empyfrwrd_ip

from TEM.library.utils.TEM_inv_tools import prep_mdl_para_names
from TEM.library.utils.TEM_inv_tools import vecMDL2mtrx
from TEM.library.utils.TEM_inv_tools import mtrxMDL2vec
from TEM.library.utils.TEM_inv_tools import plot_diffs
from TEM.library.utils.TEM_inv_tools import calc_doi

from TEM.library.utils.timer import Timer
from TEM.library.utils.TEM_proc_tools import parse_TEMfastFile

from TEM.library.TEM_inv.pg_temip_inv import tem_block1D_fwd
from TEM.library.TEM_inv.pg_temip_inv import temip_block1D_fwd
from TEM.library.TEM_inv.pg_temip_inv import LSQRInversion
from TEM.library.TEM_inv.pg_temip_inv import setup_initialmdl_constraints
from TEM.library.TEM_inv.pg_temip_inv import setup_initialipmdl_constraints
from TEM.library.TEM_inv.pg_temip_inv import filter_data

from TEM.library.utils.universal_tools import round_up
from TEM.library.utils.universal_tools import get_float_from_string
from TEM.library.utils.universal_tools import query_yes_no
from TEM.library.utils.universal_tools import plot_signal
from TEM.library.utils.universal_tools import plot_rhoa
from TEM.library.utils.universal_tools import calc_rhoa

from TEM.library.tem_tools.survey import Survey

logging.basicConfig()
logging.getLogger().setLevel(logging.WARNING)
# logging.getLogger().setLevel(logging.DEBUG)
# logging.getLogger().setLevel(logging.INFO)


# %% plot style and limits for plot
plt.style.use('ggplot')

adapt_labelsize = 2
# plot font sizes
plt.rcParams['axes.labelsize'] = 16 + adapt_labelsize
plt.rcParams['axes.titlesize'] = 16 + adapt_labelsize
plt.rcParams['xtick.labelsize'] = 14 + adapt_labelsize
plt.rcParams['ytick.labelsize'] = 14 + adapt_labelsize
plt.rcParams['legend.fontsize'] = 16 + adapt_labelsize

# limits
min_time = 3e-6
limits_sign = [1e-10, 1e-2]
limits_rhoa = [1e1, 1e4]

limits_dpt = (50, 0)
limits_rho = [1e1, 1e4]
limits_m = [-0.1, 1.1]
limits_tau = [3e-7, 1e-2]
limits_c = [-0.1, 1.1]

log_rho = True
log_rhoa = True


# %% booolean switches
start_inv = True
# start_inv = False

show_results = True
# show_results = False

show_forward_comp = True
# show_forward_comp = False

save_forward_comp = True
# save_dataplot = False

save_resultplot = True
# save_resultplot = False

save_data = True

# save_to_xls = True  # necessary for profile plotting
save_to_xls = False

test_single_sounding = False  # run for all soundings 
# test_single_sounding = 'first'
# test_single_sounding = 'last'


# %% directions data
main = Path(__file__).parents[3]

proj_dir = '00_data/selected/'
path = main / proj_dir

path_data = str(path)+'/'
main = str(main)+'/'
fnames = ['20250710_A6.tem']

scriptname = 'A6-IP_blk-04lay_v00.py'
print(f'running {scriptname} ...')
version = scriptname.split('.')[0].split('_')[-1]
batch_type = scriptname.split('.')[0].split('_')[-2]
inv_type = scriptname.split('.')[0].split('_')[-3]


# %% inversion settings, filtering and limits ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
lambdas = [500]  # in blocky case: initial lambdas
cooling_factor = [0.8]
max_iter = 25
mys = [1e2]  # regularization for parameter constraint
noise_floors = np.arange(0.05, 0.21, 0.03) # Code OV paper
# noise_floors = np.r_[0.04]

#gen_max_depth = None   # if None, will be automatically determined from loop
#gen_max_depth = 'individual'  # or use a float value that limits the actual depth!
# gen_max_depth = 'constraint'
# gen_max_depth = 50.0
# gen_nlayers = 5
gen_nlayers = 6 # NOT > 15

# time_ranges = np.array([[8., 200.], [8., 400.], [8., 600.]])  # in us
# time_ranges = np.array([[8., 800.], [8., 1000.], [8., 1000.], [8., 1000.], [8., 1000.]])  # in us
time_ranges = np.array([[5., 100.]])  # in us

# thk_lu = (2, 30) # (1, 10)    # boundaries for layer thicknesses
# res_lu = (1, 800) # (0.5, 500)  # boundaries for resistivity layers
thk_lu = (1, 12)    # boundaries for layer thicknesses
res_lu = (100, 5000)  # boundaries for resistivity layers

ip_modeltype = 'mpa'  # or None if no IP
# boundaries for IP params: [pol, e.g.: m, mpa];    [time constant, e.g.: tau, taup];     [dispersion: c]
ip_params_lu = np.array([[0.01, 1], [1e-6, 1e-3], [0.01, 1]])
ramp_data = 'salzlacken'


# %% manual initial model and constraints
gen_max_depth = 'constraint'
# inlay_thk = np.r_[2.3, 8.0, 8.0, 8.0, 6.0, 8.0, 8.0, 8.0]
inlay_thk = np.r_[4, 4, 4, 8]
# inlay_res = np.r_[30.0, 8.0, 8.0, 100.0, 40.0, 25.0, 20.0, 20.0, 20.0]
# inlay_res = np.r_[30, 30, 30, 30, 30, 30]
inlay_res = np.r_[50, 1500, 1500, 500, 500]

# inlay_mpa = np.r_[0.0, 0.4, 0.4, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0]
# inlay_mpa = np.r_[0.0, 0.3, 0.3, 0.3, 0.3, 0.3]
inlay_mpa = np.r_[0.0, 0.4, 0.4, 0.4, 0.0]
# inlay_taup = np.r_[1e-6, 5e-4, 5e-4, 5e-4, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]
inlay_taup = np.r_[1e-6, 5e-4, 5e-4, 1e-6, 1e-6]
# inlay_c = np.r_[0.01, 0.5, 0.5, 0.5, 0.01, 0.01, 0.01, 0.01, 0.01]
inlay_c = np.r_[0.1, 0.7, 0.7, 0.1, 0.1]

# constr_thk = np.r_[1, 0, 0, 0, 0, 0, 0, 0]
constr_thk = np.r_[0, 0, 0, 0]
# constr_res = np.r_[0, 0, 0, 0, 0, 0, 0, 0, 0]
constr_res = np.r_[0, 0, 0, 0, 0]

# constr_mpa = np.r_[1, 0, 0, 0, 1, 1, 1, 1, 1]
constr_mpa = np.r_[1, 0, 0, 1, 1]
# constr_taup = np.r_[1, 0, 0, 0, 1, 1, 1, 1, 1]
constr_taup = np.r_[1, 0, 0, 1, 1]
# constr_c = np.r_[1, 0, 0, 0, 1, 1, 1, 1, 1]
constr_c = np.r_[1, 0, 0, 1, 1]


# %% automatic parameters for IP inversion ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# init_rho_background = 30
init_rho_background = 50
init_rho_ip = 100
initpmax = 0.5
inittaup = 1e-4
initc = 0.5

ip_depth_range = (0.0, 0.6)  # quantiles for IP depth range
bottom_ip_switch = False     # whether or not the last layer has IP


# %% loop over files in fnames list (in case you want to invert multiple files)
# e.g.: useful for testing different time ranges, use same filename multiple times
t = Timer()
full_time = Timer()
full_time.start()

for idx, fname_data_full in enumerate(fnames):
    print('\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(f' - starting preparation for inversion of file: {fname_data_full}\n')

    fname_data = fname_data_full.split('.')[0]
    fID_savename = fname_data[:-4] if fname_data[-4:] == '_txt' else fname_data
    # pdf_filename = '{:s}_{:s}_{:02d}.pdf'.format(log_id_savename,
                                                 # name_snd, log_id)
    pdf_filename = '{:s}_batch_{:s}.pdf'.format(fID_savename, version)

    if len(time_ranges) == len(fnames):
        time_range = time_ranges[idx] * 1e-6
        tr0 = time_range[0]
        trN = time_range[1]
        print('\n-----------------------------------------------------------------')
        print('selected time range (us): ', int(tr0*1e6), '-', int(np.round(trN*1e6, 0)))
        print('-----------------------------------------------------------------\n')
        print(f'using this time range for all soundings in file: {fname_data_full}\n')
        version = f'{version}_tr{int(tr0*1e6)}-{int(np.round(trN*1e6, 0))}us'
        
    elif len(time_ranges) > len(fnames):
        msg = ('more time ranges than files available, assuming individual ' +
               'time ranges for each sounding.')
        logging.info(msg)
    else:
        msg = ('please make sure that you provide time_ranges that match either ' +
              'the length of the files provided or the number of soundings in a file')
        raise ValueError(msg)

    pre = f'pyGIMLi/{inv_type}_{batch_type}'
    savepath_data = main + f'30_inv-results/{pre}/{fID_savename}/{version}/'
    if not os.path.exists(savepath_data):
        os.makedirs(savepath_data)


    # read data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    survey = Survey()
    survey.parse_temfast_data(filename=fname_data_full, path=path_data)
    metadata = survey.metainfos
    loop_sizes = np.asarray(metadata.tx_side1)

    soundings = survey.soundings  # ordered dict containing all soundings from survey
    n_logs = len(loop_sizes)

    # setup initial model
    transThk = pg.trans.TransLogLU(thk_lu[0], thk_lu[1])  # log-transform ensures thk>0
    transRho = pg.trans.TransLogLU(res_lu[0], res_lu[1])  # lower and upper bound

    # prepare inversion protokoll ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    main_prot_fid = savepath_data + f'{fID_savename}.log'
    invprot_hdr = ('name\tri\tminT(us)\tmaxT(us)\tlam\tlam_fin\tmy\tcf\tnoifl\t' +
                   'max_iter\tn_iter\taRMS\trRMS\tchi2\truntime(min)')
    if start_inv:
        with open(main_prot_fid, 'w') as main_prot:
            main_prot.write(invprot_hdr + '\n')

    snd_names = []
    snd_positions = []

    if not test_single_sounding:
        snds = soundings.values()
    elif test_single_sounding == 'first':
        name_sel = survey.sounding_names[0]
        snds = [soundings[name_sel]]
    elif test_single_sounding == 'last':
        name_sel = survey.sounding_names[0]
        snds = [soundings[name_sel]]
    n_logs = len(snds)

    if query_yes_no(f'\n\n ? Proceed with iteration over n={n_logs} soundings', default='no'):
    # if start_preps:
        with PdfPages(savepath_data + pdf_filename) as pdf:
            for log_id, snd in enumerate(snds):  # for running all soundings

                # start by reading a single .tem data file # %% was eliminated
                # create sounding params from header:
                setup_device = snd.get_device_settings()
                setup_device["ramp_data"] = ramp_data
                snd_name = snd.name

                dmeas = snd.get_obsdata_dataframe()

                posX = snd.metainfo['posX']
                posY = snd.metainfo['posY']
                posZ = snd.metainfo['posZ']

                obs_dat = dmeas.signal.values
                obs_err = dmeas.err.values
                times_all = dmeas.time.values
                max_time = round_up(max(times_all)*1e6, 100) / 1e6
                rhoa = dmeas.rhoa.values
                init_rho = np.nanmedian(rhoa)

                print('\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                print(f' - starting preparation for inversion of sounding: {snd_name}\n')

                savepath_autoplot = savepath_data + f'{snd_name}/autoplot/'
                if not os.path.exists(savepath_autoplot):
                    os.makedirs(savepath_autoplot)
                savepath_csv = savepath_data + f'{snd_name}/csv/'
                if not os.path.exists(savepath_csv):
                    os.makedirs(savepath_csv)


                # prepare constraints and initial values
                lays_log_parts = np.logspace(-1, 0.0, gen_nlayers, endpoint=True)
                if gen_max_depth is None:
                    print('automatic selection of layer depth based on max loop size in survey')
                    gen_max_depth = np.max(loop_sizes) * 4
                    gen_init_thk = np.diff(lays_log_parts * gen_max_depth)
                    gen_init_thkcum = np.cumsum(gen_init_thk)

                    mask_thk = gen_init_thkcum < gen_max_depth
                    init_layer_thk = gen_init_thk[mask_thk]
                    nlayer = init_layer_thk.shape[0] + 1  # +1 for bottom layer

                elif gen_max_depth == 'individual':
                    depth_max = snd.tx_loop * 4
                    init_layer_thk = np.diff(lays_log_parts * depth_max)

                    gen_init_thkcum = np.cumsum(init_layer_thk)
                    nlayer = init_layer_thk.shape[0] + 1  # +1 for bottom layer

                elif isinstance(gen_max_depth, float):
                    gen_init_thk = np.diff(lays_log_parts * gen_max_depth)
                    gen_init_thkcum = np.cumsum(gen_init_thk)

                    # mask_thk = gen_init_thkcum < depth_max
                    init_layer_thk = gen_init_thk
                    nlayer = init_layer_thk.shape[0] + 1  # +1 for bottom layer
                
                elif gen_max_depth == 'constraint':
                    init_layer_thk = inlay_thk

                else:
                    raise ValueError('please select either None, "individual" or a float value for the gen_maximum depth')


                transM = pg.trans.TransLogLU(ip_params_lu[0, 0], ip_params_lu[0, 1])  # lower and upper bound chargeability, mpa
                transTau = pg.trans.TransLogLU(ip_params_lu[1, 0], ip_params_lu[1, 1])  # lower and upper bound tau
                transC = pg.trans.TransLogLU(ip_params_lu[2, 0], ip_params_lu[2, 1])  # lower and upper bound dispersion coeff

                if ip_modeltype != None:
                    if gen_max_depth != 'constraint':
                        print('automatically deriving the initial model and constraints')
                        lower_bound_ip = np.quantile(gen_init_thkcum, ip_depth_range[0])
                        upper_bound_ip = np.quantile(gen_init_thkcum, ip_depth_range[1])
        
                        condition = ((gen_init_thkcum >= lower_bound_ip)
                                    & (gen_init_thkcum <= upper_bound_ip))
                        mask_ip_layers = np.r_[condition, bottom_ip_switch]
        
        
                        # IP model parameters
                        print('adding resistivity contrast to IP layer; overwriting init resistivity from apparent resistivity')
                        constr_thk = np.zeros_like(init_layer_thk)  # set to 1 if parameter should be fixed
    
                        # resistivity
                        init_layer_res = np.ones((len(init_layer_thk) + 1,)) * init_rho_background
                        init_layer_res[mask_ip_layers] = init_rho_ip
                        constr_res = np.zeros_like(init_layer_res)
    
                        # chargeability, mpa
                        init_layer_mpa = np.zeros_like(constr_res)
                        init_layer_mpa[mask_ip_layers] = initpmax
                        constr_mpa = np.ones_like(init_layer_mpa)
                        constr_mpa[mask_ip_layers] = 0
    
                        # tau and tau phi
                        init_layer_taup = np.full_like(constr_res, 1e-6)
                        init_layer_taup[mask_ip_layers] = inittaup
                        constr_taup = np.ones_like(init_layer_taup)
                        constr_taup[mask_ip_layers] = 0
    
                        # dispersion coefficient
                        init_layer_c = np.zeros_like(constr_res)
                        init_layer_c[mask_ip_layers] = initc
                        constr_c = np.ones_like(init_layer_c)
                        constr_c[mask_ip_layers] = 0
                    else:
                        print('using the initial model and constraints that were set manually')
                        init_layer_res = inlay_res
                        init_layer_thk = inlay_thk
                        
                        init_layer_mpa = inlay_mpa
                        init_layer_taup = inlay_taup
                        init_layer_c = inlay_c

                else:
                    if gen_max_depth != 'constraint':
                        init_layer_res = np.full((nlayer, ), init_rho)
                        constr_thk = np.zeros_like(init_layer_thk)  # set to 1 if parameter should be fixed
                        constr_res = np.zeros_like(init_layer_res)
                    else:
                        init_layer_res = inlay_res

                # setup constraints, basic parameters


                if ip_modeltype == None:
                    (initmdl_pgvec, initmdl_arr,
                     Gi, constrain_mdl_params,
                     param_names) = setup_initialmdl_constraints(constr_thk, constr_res,
                                                                 init_layer_thk, init_layer_res)
                    cstr_vals = None
                else:
                    (initmdl_pgvec, initmdl_arr,
                      Gi, constrain_mdl_params,
                      param_names) = setup_initialipmdl_constraints(ip_modeltype, constr_thk, constr_res,
                                                                  constr_mpa, constr_taup, constr_c,
                                                                  init_layer_thk, init_layer_res,
                                                                  init_layer_mpa, init_layer_taup, init_layer_c)
                    cstr_vals = None

                mdl_para_names = prep_mdl_para_names(param_names, n_layers=len(init_layer_res))
                nlayers = initmdl_arr.shape[0]
                nparams = initmdl_arr.shape[1]


                # filtering the data - # select subset according to time range
                if len(time_ranges) == len(snds):
                    time_range = time_ranges[log_id] * 1e-6
                    tr0 = time_range[0]
                    trN = time_range[1]
                    print('\n-----------------------------------------------------------------')
                    print('selected time range (us): ', int(tr0*1e6), '-', int(np.round(trN*1e6, 0)))
                    print('-----------------------------------------------------------------\n')
                    print(f'using this time range for sounding ID: {snd_name}\n')
                elif len(time_ranges) == len(fnames):
                    time_range = time_ranges[idx] * 1e-6
                    tr0 = time_range[0]
                    trN = time_range[1]
                    print('\n-----------------------------------------------------------------')
                    print('selected time range (us): ', int(tr0*1e6), '-', int(np.round(trN*1e6, 0)))
                    print('-----------------------------------------------------------------\n')
                    print(f'using this time range for all soundings in file: {fname_data_full}\n') ## wrong message
                else:
                    msg = ('please make sure that you provide time_ranges that match either ' +
                          'the length of the files provided or the number of soundings in a file')
                    raise ValueError(msg)

                (rxtimes_sub, obsdat_sub, obserr_sub,
                 dmeas_sub, time_range) = filter_data(dmeas,
                                                      time_range,
                                                      ip_modeltype)

                tr0 = time_range[0]
                trN = time_range[1]
                relerr_sub = abs(obserr_sub) / obsdat_sub
                # rhoa_median = np.round(np.median(dmeas_sub.rhoa.values), 2)


                # setup system and forward solver
                device = 'TEMfast'
                # 'ftarg': 'key_81_CosSin_2009', 'key_201_CosSin_2012', 'ftarg': 'key_601_CosSin_2009'
                setup_solver = {'ft': 'dlf',                     # type of fourier trafo
                                  'ftarg': 'key_601_2009',  # ft-argument; filter type # https://empymod.emsig.xyz/en/stable/api/filters.html#module-empymod.filters -- for filter names
                                  'verbose': 0,                    # level of verbosity (0-4) - larger, more info
                                  'srcpts': 3,                     # Approx. the finite dip. with x points. Number of integration points for bipole source/receiver, default is 1:, srcpts/recpts < 3 : bipole, but calculated as dipole at centre
                                  'recpts': 3,                     # Approx. the finite dip. with x points. srcpts/recpts >= 3 : bipole
                                  'ht': 'dlf',                     # type of fourier trafo
                                  'htarg': 'key_401_2009',         # hankel transform filter type, 'key_401_2009', 'key_101_2009'
                                  'nquad': 3,                      # Number of Gauss-Legendre points for the integration. Default is 3.
                                  'cutoff_f': 1e8,                 # TODO add automatisation for diff loops;  cut-off freq of butterworthtype filter - None: No filter applied, WalkTEM 4.5e5
                                  'delay_rst': 0,                  # ?? unknown para for walktem - keep at 0 for fasttem
                                  'rxloop': 'vert. dipole'}        # or 'same as txloop' - not yet operational


                # inversion setup and test startmodel using first entry in mesh related lists
                if ip_modeltype != None:
                    empy_frwrd = empyfrwrd_ip(setup_device=setup_device,
                                           setup_solver=setup_solver,
                                           time_range=time_range, device='TEMfast',
                                           nlayer=nlayers, nparam=nparams)
                    fop = temip_block1D_fwd(empy_frwrd, ip_mdltype=ip_modeltype,
                                            nPara=nparams-1, nLayers=nlayers,
                                            verbose=True)

                    transData = pg.trans.TransLin()  # lin transformation for data
                    fop.region(0).setTransModel(transThk)  # 0=thickness
                    fop.region(1).setTransModel(transRho)  # 1=resistivity
                    fop.region(2).setTransModel(transM)    # 2=m
                    fop.region(3).setTransModel(transTau)  # 3=tau
                    fop.region(4).setTransModel(transC)    # 4=c

                    fop.setMultiThreadJacobian(1)
                else:
                    empy_frwrd = empyfrwrd(setup_device=setup_device,
                                              setup_solver=setup_solver,
                                              time_range=time_range, device='TEMfast',
                                              nlayer=nlayers, nparam=nparams)
                    fop = tem_block1D_fwd(empy_frwrd, nPara=nparams-1,
                                          nLayers=nlayers, verbose=True)

                    transData = pg.trans.TransLog()  # log transformation for data, possible, because we dont expect any negative values in the data
                    fop.region(0).setTransModel(transThk)  # 0=thickness
                    fop.region(1).setTransModel(transRho)  # 1=resistivity

                    fop.setMultiThreadJacobian(1)

                t.start()
                simdata = fop.response(initmdl_pgvec)  # simulate start model response
                frwrd_time = t.stop(prefix='forward-')


                # visualize rawdata and filtering plus error!!
                if show_forward_comp:
                    fg_raw, ax_raw = plt.subplots(1, 2, figsize=(12, 6))

                    # rawdata first
                    _ = plot_signal(ax_raw[0], times_all, np.asarray(snd.sgnl_o),
                                    marker='o', ls='-', color='grey', sub0color='orange',
                                    label='data raw')

                    ax_raw[0].loglog(times_all, np.asarray(snd.error_o),  # noise
                                      'd', ms=4, ls=':',
                                      color='grey', alpha=0.5,
                                      label='noise raw')

                    _ = plot_rhoa(ax_raw[1], times_all, np.asarray(snd.rhoa_o),
                                  marker='o', ls='--', color='grey',
                                  label='rhoa raw')

                    # filtered data
                    _ = plot_signal(ax_raw[0], rxtimes_sub, obsdat_sub,
                                    marker='d', ls='--', color='k', sub0color='orange',
                                    label='data subset')

                    _ = plot_rhoa(ax_raw[1], rxtimes_sub, np.asarray(dmeas_sub.rhoa),
                                  marker='d', ls='--', color='k', sub0color='orange',
                                  label='rhoa subset')

                    # and comparing it to the measured (observed) subset of the data
                    _ = plot_signal(ax_raw[0], rxtimes_sub, simdata,
                                    marker='x', color='crimson', ls='None', sub0color='orange',
                                    label='data sim. initial mdl')

                    sim_rhoa = calc_rhoa(setup_device, simdata, rxtimes_sub)
                    _ = plot_rhoa(ax_raw[1], rxtimes_sub, sim_rhoa,
                                  marker='x', color='crimson', ls='None', sub0color='orange',
                                  label='rhoa sim')

                    max_time = round_up(max(times_all)*1e6, 100) / 1e6
                    ax_raw[0].set_xlabel('time (s)')
                    ax_raw[0].set_ylabel(r'$\frac{\delta B}{\delta t}$ (V/m²)')
                    ax_raw[0].set_ylim((limits_sign[0], limits_sign[1]))
                    # ax_raw[0].set_xlim((min_time, max_time))
                    ax_raw[0].grid(True, which='major', color='white', linestyle='-')
                    ax_raw[0].grid(True, which='minor', color='white',  linestyle=':')
                    ax_raw[0].legend()
                    ax_raw[0].set_title(f'{snd.name}, {snd.tx_loop:.1f} m')

                    ax_raw[1].set_xlabel('time (s)')
                    ax_raw[1].set_ylabel(r'$\rho_a$ ($\Omega$m)')
                    ax_raw[1].set_ylim(limits_rhoa)
                    ax_raw[1].set_xlim((min_time, max_time))
                    ax_raw[1].yaxis.set_label_position("right")
                    ax_raw[1].yaxis.tick_right()
                    ax_raw[1].yaxis.set_ticks_position('both')
                    ax_raw[1].yaxis.set_minor_formatter(ticker.FuncFormatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y))))
                    for label in ax_raw[1].yaxis.get_minorticklabels()[1::2]:
                        label.set_visible(False)  # remove every second label
                    ax_raw[1].grid(True, which='major', color='white', linestyle='-')
                    ax_raw[1].grid(True, which='minor', color='white',  linestyle=':')
                    ax_raw[1].set_title(f'timekey: {snd.timekey}, {snd.current} A')
                    if log_rhoa:
                        ax_raw[1].set_yscale('log')

                    ax_raw[1].legend()
                    plt.tight_layout()

                    if save_forward_comp:
                        fig_savefid = (savepath_autoplot +
                                       'data_{:s}_{:02d}.png'.format(snd_name, log_id))
                        print('saving rawdata plot to:\n', fig_savefid)
                        fg_raw.savefig(fig_savefid, dpi=150)
                        print('saving rawdata to auto pdf at:\n', pdf)
                        fg_raw.savefig(pdf, format='pdf')
                        plt.close('all')
                    else:
                        print('rawdata plot not saved...')
                        plt.show()


                # start the inversion
                # prepare inversion protokoll for individual sounding
                snd_prot_fid = savepath_csv.replace('csv/', '') + f'{fID_savename}_snd{snd_name}.log'

                total_runs = len(lambdas) * len(mys) * len(cooling_factor) * len(noise_floors)
                message = (f'proceed with inversion using n={total_runs:.0f} different settings\n' +
                           '"no" proceeds with plotting - only if inversion was done already ...')
                # if start_inv and query_yes_no(message, default='no'):

                if start_inv:
                    with open(snd_prot_fid, 'w') as snd_prot:
                        snd_prot.write(invprot_hdr + '\n')
                    inv_run = 0

                    for lam in lambdas:
                        for my in mys:
                            for cf in cooling_factor:
                                for noise_floor in noise_floors:

                                    tem_inv = LSQRInversion(verbose=True)

                                    tem_inv.setTransData(transData)
                                    tem_inv.setForwardOperator(fop)

                                    tem_inv.setMaxIter(max_iter)
                                    tem_inv.setLambda(lam)  # (initial) regularization parameter

                                    tem_inv.setMarquardtScheme(cf)  # decrease lambda by factor 0.9
                                    tem_inv.setBlockyModel(True)
                                    tem_inv.setModel(initmdl_pgvec)  # set start model
                                    tem_inv.setData(obsdat_sub)

                                    print('\n\n####################################################################')
                                    print('####################################################################')
                                    print('####################################################################')
                                    print(f'--- about to start the inversion run: ({inv_run}) ---')
                                    print('lambda - cool_fac - my - noise_floor(%) - max_iter:')
                                    print((f'{lam:6.3f} - ' +
                                           f'{cf:.1e} - ' +
                                           f'{my:.1e} - ' +
                                           f'{noise_floor*100:.2f} - ' +
                                           f'{max_iter:.0f} - '))
                                    print('and initial model:\n', initmdl_pgvec)

                                    rel_err = np.copy(relerr_sub)
                                    if any(rel_err < noise_floor):
                                        logging.warning(f'Encountered rel. error below {noise_floor*100}% - setting those to {noise_floor*100}%')
                                        rel_err[rel_err < noise_floor] = noise_floor
                                        abs_err = abs(obsdat_sub * rel_err)
                                    else:
                                        abs_err = abs(obsdat_sub * rel_err)

                                    tem_inv.setAbsoluteError(abs(abs_err))
                                    if constrain_mdl_params:
                                        tem_inv.setParameterConstraints(G=Gi, c=cstr_vals, my=my)
                                    else:
                                        print('no constraints used ...')

                                    t.start()
                                    model_inv = tem_inv.run()
                                    inv_time = t.stop(prefix='inv-')

                                    inv_res, inv_thk = model_inv[nlayers-1:nlayers*2-1], model_inv[0:nlayers-1]
                                    chi2 = tem_inv.chi2()
                                    relrms = tem_inv.relrms()

                                    model_inv_mtrx = vecMDL2mtrx(model_inv, nlayers, nparams)

                                    if ip_modeltype != None:
                                        inv_m = model_inv_mtrx[:, 2]
                                        inv_tau = model_inv_mtrx[:, 3]
                                        inv_c = model_inv_mtrx[:, 4]

                                    fop = tem_inv.fop()
                                    col_names = mdl_para_names
                                    row_names = [f'dB/dt tg{i:02d}' for i in range(1, len(obsdat_sub)+1)]
                                    jac_df = pd.DataFrame(np.array(fop.jacobian()), columns=col_names, index=row_names)

                                    print('\ninversion runtime: {:.1f} min.'.format(inv_time[0]/60))
                                    print('--------------------   INV finished   ---------------------')


                                    # save result and fit
                                    chi2 = tem_inv.chi2()
                                    rrms = tem_inv.relrms()
                                    arms = tem_inv.absrms()
                                    lam_fin = tem_inv.getLambda()
                                    n_iter = tem_inv.n_iters

                                    pred_data = np.asarray(tem_inv.response())
                                    pred_rhoa = calc_rhoa(setup_device, pred_data,
                                                          rxtimes_sub)

                                    if ip_modeltype == 'pelton':
                                        header_result = 'X,Y,Z,depth(m),rho(Ohmm),m(),tau(s),c()'
                                        labels_CC = ['chargeability m ()', r'rel. time $\tau$ (s)']
                                        result_arr = np.column_stack((np.r_[inv_thk, 0], inv_res,
                                                                      inv_m, inv_tau, inv_c))
                                    elif ip_modeltype == 'mpa':
                                        header_result = 'X,Y,Z,depth(m),rho(Ohmm),mpa(rad),tau_p(s),c()'
                                        labels_CC = ['mpa (rad)', r'rel. time $\tau_{\phi}$ (s)']
                                        result_arr = np.column_stack((np.r_[inv_thk, 0], inv_res,
                                                                      inv_m, inv_tau, inv_c))
                                    elif ip_modeltype == None:
                                        header_result = 'X,Y,Z,depth(m),rho(Ohmm)'
                                        result_arr = np.column_stack((np.r_[inv_thk, 0], inv_res))
                                    else:
                                        raise ValueError('this ip modeltype is not implemented here ...')
                                    export_array = np.column_stack((np.full((len(result_arr),), posX),
                                                                    np.full((len(result_arr),), posY),
                                                                    np.full((len(result_arr),), posZ),
                                                                    result_arr))

                                    header_fit = ('time(s),signal_pred(V/m2),' +
                                                  'signal_obs(V/m2),err_obs(V/m2),err_scl(V/m2),' +
                                                  'rhoa_pred(V/m2),rhoa_obs(V/m2)')
                                    export_fit = np.column_stack((rxtimes_sub,
                                                                  pred_data, obsdat_sub,
                                                                  obserr_sub, abs_err,
                                                                  pred_rhoa, dmeas_sub.rhoa))

                                    header_startmdl = 'X,Y,Z,thk(m),rho(Ohmm),m(),tau(s),c()'
                                    exportSM_array = np.column_stack((np.full((len(result_arr),), posX),
                                                                      np.full((len(result_arr),), posY),
                                                                      np.full((len(result_arr),), posZ),
                                                                      initmdl_arr))


                                    savename = ('invrun{:03d}_{:s}'.format(inv_run, snd_name))
                                    if save_data == True:
                                        print(f'saving data from inversion run: {inv_run}')
                                        if ip_modeltype != None:
                                            formatting = '%.3f,%.3f,%.3f,%.3f,%.3f,%.1f,%.1e,%.3f'
                                        else:
                                            formatting = '%.3f,%.3f,%.3f,%.3f,%.3f'
                                        np.savetxt(savepath_csv + savename +'.csv',
                                                   export_array, comments='',
                                                   header=header_result,
                                                   fmt=formatting)
                                        np.savetxt(savepath_csv + savename +'_startmodel.csv',
                                                   exportSM_array, comments='',
                                                   header=header_result,
                                                   fmt=formatting)

                                        np.savetxt(savepath_csv + savename +'_fit.csv',
                                                   export_fit,
                                                   comments='',
                                                   header=header_fit,
                                                   fmt='%.6e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e')
                                        jac_df.to_csv(savepath_csv + savename + '_jac.csv')


                                    # save main log
                                    logline = ("%s\t" % (snd_name) +
                                                "r%03d\t" % (inv_run) +
                                                "%.1f\t" % (tr0*1e6) +
                                                "%.1f\t" % (trN*1e6) +
                                                "%8.1f\t" % (lam) +
                                                "%8.1f\t" % (lam_fin) +
                                                "%.1e\t" % (my) +
                                                "%.1e\t" % (cf) +  # cooling factor
                                                "%.2f\t" % (noise_floor) +
                                                "%d\t" % (max_iter) +
                                                "%d\t" % (n_iter) +
                                                "%.2e\t" % (arms) +
                                                "%7.3f\t" % (rrms) +
                                                "%7.3f\t" % (chi2) +
                                                "%4.1f\n" % (inv_time[0]/60))  # to min
                                    with open(main_prot_fid,'a+') as f:
                                        f.write(logline)
                                    with open(snd_prot_fid,'a+') as f:
                                        f.write(logline)

                                    inv_run += 1


                # reload results and plot
                if show_results:
                    # read main protocol
                    snd_log = pd.read_csv(snd_prot_fid, delimiter='\t', dtype=str)

                    if ip_modeltype == 'pelton':
                        labels_CC = ['chargeability m ()', r'rel. time $\tau$ (s)']
                    elif ip_modeltype == 'mpa':
                        labels_CC = ['mpa (rad)', r'rel. time $\tau_{\phi}$ (s)']
                    elif ip_modeltype == None:
                        pass
                    else:
                        raise ValueError('this ip modeltype is not implemented here ...')

                    # TODO calc and plot the DOI
                        # [] calssical approach
                        # [] from jacobian

                    for inv_run in range(0, len(snd_log)):
                        survey_n = Survey()
                        survey_n.parse_temfast_data(filename=fname_data_full, path=path_data)
                        survey_n.parse_inv_results(result_folder=savepath_data, invrun=f'{inv_run:03d}')

                        soundings_n = survey_n.soundings

                        snd_n = soundings_n[survey_n.sounding_names[log_id]]

                        print('\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                        print(f' - starting visualization of inversion result for sounding: {snd_name}\n')

                        log = survey_n.invlog_info[survey_n.invlog_info.name == snd_n.name.strip()]
                        rrms = log.rRMS.values[0]
                        chi2 = log.chi2.values[0]
                        lam = log.lam.values[0]
                        lamfin = log.lam_fin.values[0]
                        cf = log.cf.values[0]
                        noifl = log.noifl.values[0]
                        niter = log.n_iter.values[0]

                        model_inv = mtrxMDL2vec(snd_n.inv_model)
                        nlayer = int(len(model_inv) / 2)

                        # TODO DOI calculation to survey class
                        res_inv, thk_inv = model_inv[nlayer-1:nlayer*2-1], model_inv[0:nlayer-1]
                        mdl4doi = np.column_stack((np.r_[0, thk_inv], res_inv))
                        doi = calc_doi(current=snd_n.current,
                                       tx_area=snd_n.tx_loop**2,
                                       eta=snd_n.sgnl_c.iloc[-1],
                                       mdl_rz=mdl4doi, x0=30,
                                       verbose=True)

                        if ip_modeltype != None: # plot IP case
                            fg_fit, ax_fit = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True,
                                                          sharex=True)
                            
                            # snd_n.plot_dBzdt(which='observed', ax=ax_fit[0],
                            #                  label='data', color='k', show_sub0_label=True,
                            #                  ms=8, lw=1.8, marker='.', ls='--', sub0_ms=8, sub0_mew=1.5, sub0_mfc='none')
                            # snd_n.plot_dBzdt(which='calculated', ax=ax_fit[0], color='crimson',
                            #                  marker='x', ls=':', label='inv. response', show_sub0_label=False,
                            #                  ms=8, lw=1.8, sub0_ms=8, sub0_mew=1.5, sub0_mfc='none')
                            snd_n.plot_dBzdt(which='observed', ax=ax_fit[0],
                                             label='data', color='k', show_sub0_label=True,
                                             marker='.', ls='--')
                            snd_n.plot_dBzdt(which='calculated', ax=ax_fit[0], color='crimson',
                                             marker='x', ls=':', label='inv. response', show_sub0_label=False)                            
                            ax_fit[0].set_xlim((min_time, max_time))
                            ax_fit[0].set_ylim(limits_sign)
                            ttl = (f'inv-run: {inv_run:03d}, sndID: {snd_name}' +
                                    f'\nchi2 = {chi2:0.2f}, rrms = {rrms:0.2f}%')
                            ax_fit[0].set_title(ttl)
                            ax_fit[0].legend()

                            # snd_n.plot_rhoa(which='observed', ax=ax_fit[1],
                            #                  label='data', color='k', show_sub0_label=True,
                            #                  ms=8, lw=1.8, marker='.', ls='--', sub0_ms=8, sub0_mew=1.5, sub0_mfc='none')
                            # snd_n.plot_rhoa(which='calculated', ax=ax_fit[1], color='crimson',
                            #                 marker='x', ls=':', label='inv. response', show_sub0_label=False,
                            #                 ms=6, lw=1.8, sub0_ms=8, sub0_mew=1.5, sub0_mfc='none')
                            snd_n.plot_rhoa(which='observed', ax=ax_fit[1],
                                             label='data', color='k', show_sub0_label=True,
                                             marker='.', ls='--')
                            snd_n.plot_rhoa(which='calculated', ax=ax_fit[1], color='crimson',
                                            marker='x', ls=':', label='inv. response', show_sub0_label=False)                            
                            ax_fit[1].set_xlim((min_time, max_time))
                            ax_fit[1].set_ylim(limits_rhoa)
                            ax_fit[1].legend()
                            ttl = (f'init. lam {lam:.1f} cooling: {cf}, iters {niter}\n' +
                                   f'fin. lam {lamfin:.1f}, noise floor = {noifl*100}%')
                            ax_fit[1].set_title(ttl)
                            if log_rhoa:
                                ax_fit[1].set_yscale('log')

                            # plot model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                            fg_mdl, ax_mdl = plt.subplots(2, 2, figsize=(9, 9), constrained_layout=True)
                            
                            # TODO
                            # ax_mdl[0,0].plot(...) plot the non-IP result

                            snd_n.plot_initial_IPmodel(ax=ax_mdl, add_bottom=50,
                                           res2con=False, ip_modeltype='mpa',
                                           color='gray', ls='--', marker='.',
                                           ms=8, lw=1.8)
                            snd_n.plot_inv_IPmodel(ax=ax_mdl, add_bottom=50,
                                           res2con=False, ip_modeltype='mpa',
                                           color='deeppink', ls='-', marker='.',
                                           ms=8, lw=1.8)

                            ax_mdl[0, 0].set_xscale('log')
                            ax_mdl[0, 0].set_ylim(limits_dpt)
                            ax_mdl[0, 0].set_xlim(limits_rho)

                            # ax_mdl[0, 1].legend()
                            ax_mdl[0, 1].set_ylim(limits_dpt)
                            ax_mdl[0, 1].set_xlim(limits_m)

                            # ax_mdl[1, 0].legend()
                            ax_mdl[1, 0].set_xscale('log')
                            ax_mdl[1, 0].set_ylim(limits_dpt)
                            ax_mdl[1, 0].set_xlim(limits_tau)
                            ax_mdl[1, 0].legend() # move to script!!

                            # ax_mdl[1, 1].legend()
                            ax_mdl[1, 1].set_ylim(limits_dpt)
                            ax_mdl[1, 1].set_xlim(limits_c)

                        else:
                            fg_all, ax_all = plt.subplots(1, 3, figsize=(18, 6),
                                                          constrained_layout=True)

                            snd_n.plot_dBzdt(which='observed', ax=ax_all[0],
                                             label='data', color='gray', marker='d',
                                             show_sub0_label=False)
                            snd_n.plot_dBzdt(which='calculated', ax=ax_all[0], color='deeppink',
                                             marker='x', ls='--', label='inv. response', show_sub0_label=False)
                            ax_all[0].set_xlim((min_time, max_time))
                            ax_all[0].set_ylim(limits_sign)
                            ttl = (f'inv-run: {inv_run:03d}, sndID: {snd_name}' +
                                    f'\nchi2 = {chi2:0.2f}, rrms = {rrms:0.2f}%')
                            ax_all[0].set_title(ttl)
                            ax_all[0].legend()
                            ax_all[0].grid(True, axis='x', which='major', color='white', linestyle='-')
                            ax_all[0].grid(True, axis='x', which='minor', color='white',  linestyle=':')

                            snd_n.plot_rhoa(which='observed', ax=ax_all[1],
                                             label='data', color='gray', marker='d',
                                             show_sub0_label=False)
                            snd_n.plot_rhoa(which='calculated', ax=ax_all[1], color='deeppink',
                                            marker='x', ls=':', label='inv. response', show_sub0_label=False)
                            ax_all[1].set_xlim((min_time, max_time))
                            ax_all[1].set_ylim(limits_rhoa)
                            ax_all[1].legend()
                            ax_all[1].grid(True, axis='x', which='major', color='white', linestyle='-')
                            ax_all[1].grid(True, axis='x', which='minor', color='white',  linestyle=':')
                            if log_rhoa:
                                ax_all[1].set_yscale('log')

                            snd_n.plot_initial_model(ax=ax_all[2], color='dodgerblue',
                                                     label='init. model', ls='-', marker='.')
                            snd_n.plot_inv_model(ax=ax_all[2], color='deeppink',
                                                 label='inv. model', ls='--', marker='.')
                            ax_all[2].axhline(y=doi[0], ls='--', color='green', label=f'DOI = {doi[0]:.1f} m')
                            ax_all[2].set_title((f'init. lam {lam:.1f} cooling: {cf}, iters {niter}\n' +
                                                    f'fin. lam {lamfin:.1f}, noise floor = {noifl*100}%'))
                            ax_all[2].invert_yaxis()
                            ax_all[2].set_xlim(limits_rho)
                            ax_all[2].set_ylim(limits_dpt)
                            ax_all[2].set_xlabel(r'$\rho$ ($\Omega$m)')
                            ax_all[2].set_ylabel('z (m)')
                            ax_all[2].legend()
                            if log_rho:
                                ax_all[2].set_xscale('log')

                        fig_savefid = (savepath_autoplot +
                                        'invrun{:03d}_{:s}_model.png'.format(inv_run, snd_name))
                        figfit_savefid = (savepath_autoplot +
                                        'invrun{:03d}_{:s}_fit.png'.format(inv_run, snd_name))
                        figjac_savefid = (savepath_autoplot +
                                        'invrun{:03d}_{:s}_jac.png'.format(inv_run, snd_name))


                        # save plots
                        figj, axj = snd_n.plot_jacobian()

                        if save_resultplot == True:
                            if ip_modeltype != None:
                                print('saving result plot to:\n', fig_savefid)
                                fg_fit.savefig(figfit_savefid, dpi=150)
                                fg_mdl.savefig(fig_savefid, dpi=150)
                                figj.savefig(figjac_savefid, dpi=150)

                                print('saving result to auto pdf at:\n', pdf)
                                fg_fit.savefig(pdf, format='pdf')
                                fg_mdl.savefig(pdf, format='pdf')
                                figj.savefig(pdf, format='pdf')
                                plt.close('all')

                            else:
                                print('saving result plot to:\n', fig_savefid)
                                fg_all.savefig(fig_savefid, dpi=150)
                                figj.savefig(figjac_savefid, dpi=150)

                                print('saving result to auto pdf at:\n', pdf)
                                fg_all.savefig(pdf, format='pdf')
                                figj.savefig(pdf, format='pdf')
                                plt.close('all')
                        else:
                            print('plot not saved...')
                            plt.show()

                        # sys.exit('exit for testing ...')


if save_to_xls:
    snd_prot_fid = savepath_csv.replace('csv/', '') + f'{fID_savename}_snd{snd_name}.log'
    snd_log = pd.read_csv(snd_prot_fid, delimiter='\t')

    for inv_run in range(0, len(snd_log)):
        survey_n = Survey()
        survey_n.parse_temfast_data(filename=fname_data_full, path=path_data)
        survey_n.parse_inv_results(result_folder=savepath_data, invrun=f'{inv_run:03d}')

        save_filename = f'{fID_savename}_{inv_run:03d}_{version}'
        survey_n.save_inv_as_zondxls(save_filename=save_filename, fid_coordinates=None)

total_time = full_time.stop(prefix='total runtime-')

# %%
