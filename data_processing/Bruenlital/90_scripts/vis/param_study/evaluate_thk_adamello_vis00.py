#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 14:47:22 2021

script to simulate TEM-IP effect in high resistivity media (eg permafrost, ice)
evaluate effect of thickness of polarizing layer

3-layer model
1) Ice
2) Ice
3) compact bedrock

ToDo:
    [Done] add scatter plots for the peak positions
    [Done] extract info from response
    [Done] fix colormap
    [Done] test other parameters
    [] read inversion result and replace the simulated inv result


@author: laigner
"""

# %% modules
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import matplotlib.ticker as ticker
from matplotlib import cm
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import LogLocator, NullFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.constants import epsilon_0

# path_to_libs = ('/shares/laigner/home/nextcloud/09-python_coding/')
path_to_libs = (r"C:/Users/lukas/Documents/Programming/PYTHON/")
if not path_to_libs in sys.path:
    sys.path.append(path_to_libs)


# custom
from TEM.library.TEM_frwrd.empymod_frwrd_ip import empymod_frwrd

from TEM.library.utils.TEM_inv_tools import mtrxMDL2vec
# from TEM.library.utils.TEM_inv_tools import vecMDL2mtrx
# from TEM.library.utils.TEM_inv_tools import plot_diffs
from TEM.library.utils.universal_tools import plot_signal
# from TEM.library.utils.universal_tools import plot_rhoa
# from TEM.library.utils.universal_tools import calc_rhoa
# from TEM.library.utils.universal_tools import query_yes_no
from TEM.library.utils.universal_tools import simulate_error
from TEM.library.utils.TEM_ip_tools import plot_ip_model

# from TEM.library.utils.timer import Timer
from TEM.library.tem_tools.survey import Survey


# %% plot appearance
plt.style.use('ggplot')

fs_shift = -4
plt.rcParams['axes.labelsize'] = 18 + fs_shift
plt.rcParams['axes.titlesize'] = 18 + fs_shift
plt.rcParams['xtick.labelsize'] = 16 + fs_shift
plt.rcParams['ytick.labelsize'] = 16 + fs_shift
plt.rcParams['legend.fontsize'] = 18 + fs_shift

cmap = cm.viridis


# %% main settings
rho_min, rho_max = 5000, 50000
z_max, z_min = -20, 400

savefigs = True
# savefigs = False


# %% directions
scriptname = os.path.basename(sys.argv[0])
print(f'running {scriptname} ...')
version = scriptname.split('.')[0].split('_')[-1]
para = scriptname.split('_')[1]

main = '../../../'
savepath = main + f'40_vis/evaluate_{para}/'
if not os.path.exists(savepath):
    os.makedirs(savepath)

relerr = 1e-6
abserr = 1e-28

# eval_thk = np.r_[8, 12, 16]
eval_thk = np.arange(150, 310, 30)


# %% read field data
fname_data_full = '20240709_sel.tem'
path_data = main + '00_data/selected/'

survey = Survey()
survey.parse_temfast_data(filename=fname_data_full, path=path_data)
soundings = survey.soundings  # ordered dict containing all soundings from survey

snd = soundings['ADA2-T4']
dmeas = snd.get_obsdata_dataframe()
obs_dat = dmeas.signal.values
obs_err = dmeas.err.values
times_all = dmeas.time.values

time_range = np.r_[20, 400] * 1e-6


# %% setup solver
device = 'TEMfast'
setup_device = {"timekey": 9,
                "currentkey": 4,
                "txloop": 150,
                "rxloop": 150,
                "current_inj": 4.1,
                "filter_powerline": 50,
                "ramp_data": "salzlacken"}
# 'ftarg': 'key_81_CosSin_2009', 'key_201_CosSin_2012', 'ftarg': 'key_601_CosSin_2009'
setup_solver = {'ft': 'dlf',                     # type of fourier trafo
                'ftarg': 'key_601_CosSin_2009',  # ft-argument; filter type # https://empymod.emsig.xyz/en/stable/api/filters.html#module-empymod.filters -- for filter names
                'verbose': 0,                    # level of verbosity (0-4) - larger, more info
                'srcpts': 3,                     # Approx. the finite dip. with x points. Number of integration points for bipole source/receiver, default is 1:, srcpts/recpts < 3 : bipole, but calculated as dipole at centre
                'recpts': 3,                     # Approx. the finite dip. with x points. srcpts/recpts >= 3 : bipole
                'ht': 'dlf',                     # type of fourier trafo
                'htarg': 'key_401_2009',         # hankel transform filter type, 'key_401_2009', 'key_101_2009'
                'nquad': 3,                      # Number of Gauss-Legendre points for the integration. Default is 3.
                'cutoff_f': 1e8,                 # TODO add automatisation for diff loops;  cut-off freq of butterworthtype filter - None: No filter applied, WalkTEM 4.5e5
                'delay_rst': 0,                  # ?? unknown para for walktem - keep at 0 for fasttem
                'rxloop': 'vert. dipole'}        # or 'same as txloop' - not yet operational


# %% loop for plotting
empymod_frwrds = []
start = 0.0; stop = 0.91 # prepare colormap
cm_subsection = np.linspace(start, stop, len(eval_thk))
colors = [cmap(x) for x in cm_subsection]

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

for idx, thk in enumerate(eval_thk):
    ip_modeltype = 'mpa'
    layer_IP_idx = 1
    thk1 = thk / 2
    thk2 = thk / 2
    thks = np.r_[thk1, thk2, 0]
    rho_0 = np.r_[8400, 6400, 9400]

    charg = np.r_[0.7, 0.8, 0.0]
    taus = np.r_[2.7e-5, 3.4e-5, 1e-7]
    cs = np.r_[0.8, 0.98, 0.01]

    true_model_2d = np.column_stack((thks, rho_0, charg, taus, cs))  # pelton model
    true_model_2d = np.column_stack((thks, rho_0, charg, taus, cs))  # pelton model
    true_model_1d = mtrxMDL2vec(true_model_2d)

    nlayers = true_model_2d.shape[0]
    nparams = true_model_2d.shape[1]

    nlayer = true_model_2d.shape[0]
    nparam = true_model_2d.shape[1]

    params_fixed = f'tau={taus[layer_IP_idx]}, c={cs[layer_IP_idx]}, r0={rho_0[layer_IP_idx]}'

    # %% initialize forward solver
    empymod_frwrds.append(empymod_frwrd(setup_device=setup_device,
                                        setup_solver=setup_solver,
                                        time_range=time_range, device='TEMfast',
                                        relerr=1e-6, abserr=1e-28,
                                        nlayer=nlayer, nparam=nparam))
    curr_empym_frwrd = empymod_frwrds[idx]

    curr_empym_frwrd.calc_response(model=true_model_1d,
                                   ip_modeltype=ip_modeltype,
                                   show_wf=False)
    t_mdld = curr_empym_frwrd.times_rx
    sigIP_mdld = curr_empym_frwrd.response
    sigIP_error = simulate_error(relerr, abserr, sigIP_mdld)
    rhoaIP = curr_empym_frwrd.calc_rhoa(response=sigIP_mdld)


    sub0label = 'negative values' if idx == 0 else None
    rawlabel = 'raw data' if idx == 0 else None

    # plot raw data first, todo
    if idx == 0:
        plot_signal(axis=ax,
                    time=times_all,
                    signal=obs_dat,
                    color='gray',
                    marker='.',
                    ms=15, lw=3.5,
                    sub0color='k', sub0marker='d', sub0ms=10,
                    label='raw data')

    if idx == 1:
        empy_full_range = empymod_frwrd(setup_device=setup_device,
                                        setup_solver=setup_solver,
                                        time_range=None, device='TEMfast',
                                        relerr=1e-6, abserr=1e-28,
                                        nlayer=nlayer, nparam=nparam)
        empy_full_range.calc_response(model=true_model_1d,
                                       ip_modeltype=ip_modeltype,
                                       show_wf=False)
        full_time = empy_full_range.times_rx
        full_sgnl = empy_full_range.response

        plot_signal(axis=ax,
                    time=full_time,
                    signal=full_sgnl,
                    color='crimson',
                    marker='.',
                    ms=10, lw=1.5,
                    sub0color='k', sub0marker='d', sub0ms=10,
                    label='inv result')

    if idx == len(eval_thk)-1:
        plot_signal(axis=ax,
                    time=t_mdld,
                    signal=sigIP_mdld,
                    color=colors[idx],
                    marker='.', ms=15, lw=3.5,
                    label='thk$_{1+2}$ = ' + f'{thk:.1f} m',
                    sub0color='k', sub0marker='d', sub0ms=10,
                    sub0label='negative readings')
    else:
        plot_signal(axis=ax,
                    time=t_mdld,
                    signal=sigIP_mdld,
                    color=colors[idx],
                    marker='.', ms=15, lw=3.5,
                    label='thk$_{1+2}$ = ' + f'{thk:.1f} m',
                    sub0color='k', sub0marker='d', sub0ms=10,
                    sub0label=None)

    if ip_modeltype == 'cole_cole':
        short_name = 'Cole Cole'
        header = ','.join(['thk'.rjust(7), 'rho0'.rjust(7), 'sig0'.rjust(7), 'sig8'.rjust(7), 'tau'.rjust(7), 'c'.rjust(7)])
        cc_f_tex = (r'$\sigma(\omega) = \sigma_{\infty} + $' +
                    r'$\frac{\sigma_0 - \sigma_{\infty}}{1 + (i\omega\tau)^c}$'
                    )
    elif ip_modeltype == 'pelton':
        short_name = 'Pelton'
        header = ','.join(['thk'.rjust(7), 'rho0'.rjust(7), 'm'.rjust(7), 'tau'.rjust(7), 'c'.rjust(7)])
        cc_f_tex = (r'$\rho(\omega) = \rho_0 \left[1 - m\left(1 - \frac{1}{1 + (i\omega\tau)^c} \right) \right]$'
                    )
    elif ip_modeltype == 'cc_kozhe':
        header = ','.join(['thk'.rjust(7), 'rho0'.rjust(7), 'sig0'.rjust(7), 'eps0'.rjust(7), 'eps8'.rjust(7), 'tau'.rjust(7), 'c'.rjust(7)])
        short_name = 'Kozhevnikov & Antonov'
        cc_f_tex = (r'$\sigma(\omega) = \sigma_0 + i\omega\epsilon_0 \left[\epsilon_{\infty} + \frac{\epsilon_s - \epsilon_{\infty}}{1 + (i\omega\tau)^c}  \right]$'
                    )
    elif ip_modeltype == 'mpa':
        shortname = "MPA"
        header = ','.join(['thk'.rjust(7), 'rho0'.rjust(7), 'phi_max'.rjust(7), 'tau_phi'.rjust(7), 'c'.rjust(7)])
        cc_f_tex = (r'$\rho(\omega) = \rho_0 \left[1 - m\left(1 - \frac{1}{1 + (i\omega\tau)^c} \right) \right]$'
                    )
    else:
        raise ValueError('not yet implemented')


# %% plot model
ax_mdl = ax.inset_axes([0.6, 0.6, 0.39, 0.39])
axis, coleparams = plot_ip_model(axis=ax_mdl, ip_model=true_model_2d,
                                  ip_modeltype='mpa', add_bottom=100,
                                  layer_ip=0, rho2log=True)
axis, coleparams = plot_ip_model(axis=ax_mdl, ip_model=true_model_2d,
                                  ip_modeltype='mpa', add_bottom=100,
                                  layer_ip=1, rho2log=True)
ax_mdl.set_xlim((rho_min, rho_max))
ax_mdl.set_ylim((z_max, z_min))
ax_mdl.grid(which='major', alpha=0.75, ls='-')
ax_mdl.grid(which='minor', alpha=0.75, ls=':')
ax_mdl.invert_yaxis()


# %%
# cb.set_label('m ()')
ax.legend(loc='lower left')
ax.set_xlabel('time (s)')
ax.set_ylabel(r"$\mathrm{d}\mathrm{B}_\mathrm{z}\,/\,\mathrm{d}t$ (V/m²)")
# ax.set_title(f'evaluating m - {short_name}: ' + cc_f_tex + '\n' + params_fixed)
# ax.set_title(f'evaluating m - {short_name} et al. model: {cc_f_tex}')

plt.tight_layout()

# %% save figs
if savefigs:
    fig.savefig(savepath + f'evaluate_{para}_{version}.png', dpi=300)
    # fg.savefig(savepath + f'correlate_chargeability_{version}.png', dpi=300)


# %% plot correlation between subzero shape and tau
# fg, ax_corr = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)
# ax_c = ax_corr.flat

# ax_c[0].scatter(eval_m, pos_first_zero)
# ax_c[0].set_xlabel('m ()')
# ax_c[0].set_ylabel('position first negative reading (s)')

# ax_c[1].scatter(eval_m, pos_last_zero)
# ax_c[1].set_xlabel('m ()')
# ax_c[1].set_ylabel('position last negative reading (s)')

# ax_c[2].scatter(eval_m, pos_med_zeros)
# ax_c[2].set_xlabel('m ()')
# ax_c[2].set_ylabel('median of negative reading positions (s)')

# ax_c[3].scatter(eval_m, n_zeros)
# ax_c[3].set_xlabel('m ()')
# ax_c[3].set_ylabel('number of negative readings ()')

# for i, ax in enumerate(ax_c):
#     if i == 3:
#         pass
#     else:
#         ax.set_yscale('log')
#     # ax.xaxis.set_minor_formatter(ticker.FuncFormatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y))))
#     # ax.yaxis.set_minor_formatter(ticker.FuncFormatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y))))
#     ax.set_xlim(0.1, 1.0)
#     ax.grid(which='minor', ls=':')


