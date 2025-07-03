#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 14:47:22 2021

script to simulate TEM-IP effect in high resistivity media (eg permafrost, ice)
evaluate tau and correlate with negative voltage position in time

3-layer model
1) snow
2) Ice
4) compact bedrock

ToDo:
    [Done] add scatter plots for the peak positions
    [Done] extract info from response
    [Done] fix colormap
    [Done] test other parameters


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

# custom
from TEM_frwrd.empymod_frwrd_ip import empymod_frwrd

from TEM_frwrd.TEMIP_tools import mtrxMdl2vec
from TEM_frwrd.TEMIP_tools import vectorMDL2mtrx
from TEM_frwrd.TEMIP_tools import plot_signal
from TEM_frwrd.TEMIP_tools import plot_rhoa
from TEM_frwrd.TEMIP_tools import calc_rhoa
from TEM_frwrd.TEMIP_tools import query_yes_no
from TEM_frwrd.TEMIP_tools import simulate_error
from TEM_frwrd.TEMIP_tools import plot_diffs
from TEM_frwrd.TEMIP_tools import plot_ip_model

from TEM_frwrd.timer import Timer


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
rho_min, rho_max = 80, 20000
z_max, z_min = -2, 40

savefigs = True
# savefigs = False


# %% directions
scriptname = os.path.basename(sys.argv[0])
print(f'running {scriptname} ...')
version = scriptname.split('.')[0].split('_')[-1]
para = scriptname.split('.')[0].split('_')[-2]

savepath = f'../04-vis/example_iceglacier_3lay/corr-{para}/{version}/'
if not os.path.exists(savepath):
    os.makedirs(savepath)

relerr = 1e-6
abserr = 1e-28

# eval_tau = np.logspace(-6, 0, 19)
eval_m = np.r_[0.7, 0.8, 0.85, 0.9, 0.95]
eval_m = np.r_[0.85, 0.9, 0.95]

eval_thk = np.r_[8, 12, 16]


# %% setup solver
device = 'TEMfast'
setup_device = {"timekey": 4,
            "currentkey": 4,
            "txloop": 50,
            "rxloop": 50,
            "current_inj": 4.1,
            "filter_powerline": 50}

# 'ftarg': 'key_81_CosSin_2009', 'key_201_CosSin_2012', 'ftarg': 'key_601_CosSin_2009'
setup_solver = {'ft': 'dlf',                     # type of fourier trafo
                  'ftarg': 'key_601_CosSin_2009',  # ft-argument; filter type # https://empymod.emsig.xyz/en/stable/api/filters.html#module-empymod.filters -- for filter names
                  'verbose': 0,                    # level of verbosity (0-4) - larger, more info
                  'srcpts': 3,                     # Approx. the finite dip. with x points. Number of integration points for bipole source/receiver, default is 1:, srcpts/recpts < 3 : bipole, but calculated as dipole at centre
                  'recpts': 3,                     # Approx. the finite dip. with x points. srcpts/recpts >= 3 : bipole
                  'ht': 'dlf',                     # type of fourier trafo
                  'htarg': 'key_401_2009',         # hankel transform filter type, 'key_401_2009', 'key_101_2009'
                  'nquad': 3,                     # Number of Gauss-Legendre points for the integration. Default is 3.
                  'cutoff_f': 5e6,               # cut-off freq of butterworthtype filter - None: No filter applied, WalkTEM 4.5e5
                  'delay_rst': 0,                 # ?? unknown para for walktem - keep at 0 for fasttem
                  'rxloop': 'vert. dipole'}       # or 'same as txloop' - not yet operational


# %% loop for plotting
empymod_frwrds = []
start = 0.0; stop = 0.91 # prepare colormap
cm_subsection = np.linspace(start, stop, len(eval_m))
colors = [cmap(x) for x in cm_subsection]

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

pos_first_zero = np.zeros_like(eval_m)
pos_last_zero = np.zeros_like(eval_m)
pos_med_zeros = np.zeros_like(eval_m)
n_zeros = np.zeros_like(eval_m)

# for idx, m in enumerate(eval_m):
for idx, thk in enumerate(eval_thk):
    ip_modeltype = 'pelton'
    layer_IP_idx = 1
    
    thks = np.r_[5, thk, 0]
    rho_0 = np.r_[400, 10000, 500]
    # charg = np.r_[0, m, 0]
    charg = np.r_[0, 0.95, 0]
    taus = np.r_[0, 7e-4, 0]
    cs = np.r_[0, 0.95, 0]

    con_0 = 1 / rho_0
    rho_8 = rho_0 - (charg * rho_0)
    con_8 = 1/rho_8
    taus_CC = taus * (1 - charg) ** (1/cs)

    # cc_model = np.column_stack((thks, rho_0, con_0, con_8, taus_CC, cs))  # cole cole model
    true_model_2d = np.column_stack((thks, rho_0, charg, taus, cs))  # pelton model
    true_model_1d = mtrxMdl2vec(true_model_2d)

    nlayers = true_model_2d.shape[0]
    nparams = true_model_2d.shape[1]

    nlayer = true_model_2d.shape[0]
    nparam = true_model_2d.shape[1]
    
    params_fixed = f'tau={taus[layer_IP_idx]}, c={cs[layer_IP_idx]}, r0={rho_0[layer_IP_idx]}'


    # %% initialize forward solver
    empymod_frwrds.append(empymod_frwrd(setup_device=setup_device,
                                        setup_solver=setup_solver,
                                        filter_times=None, device='TEMfast',
                                        relerr=1e-6, abserr=1e-28,
                                        nlayer=nlayer, nparam=nparam))
    curr_empym_frwrd = empymod_frwrds[idx]

    curr_empym_frwrd.calc_response(model=true_model_1d,
                                   ip_modeltype=ip_modeltype,
                                   show_wf=False)
    t_mdld = curr_empym_frwrd.times_rx
    sigIP_mdld = curr_empym_frwrd.response
    sigIP_error = simulate_error(relerr, abserr, sigIP_mdld)
    rhoaIP = curr_empym_frwrd.calc_rhoa()
    
    sub0_mask = sigIP_mdld < 0
    times_sub0 = t_mdld[sub0_mask]
    
    if len(times_sub0) == 0:
        pos_first_zero[idx] = np.nan
        pos_last_zero[idx] = np.nan
        pos_med_zeros[idx] = np.nan
        n_zeros[idx] = len(times_sub0)
    else:
        pos_first_zero[idx] = np.nanmin(times_sub0)
        pos_last_zero[idx] = np.nanmax(times_sub0)
        pos_med_zeros[idx] = np.nanmedian(times_sub0)
        n_zeros[idx] = len(times_sub0)

    plot_signal(axis=ax,
                time=t_mdld,
                signal=sigIP_mdld,
                color=colors[idx],
                marker='.',
                ms=15, lw=3.5,
                sub0color=colors[idx],
                label=f'thk$_2$ = {thk:.1f} m')

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
    else:
        raise ValueError('not yet implemented')


# %% plot model
ax_mdl = ax.inset_axes([0.6, 0.6, 0.39, 0.39])
axis, coleparams = plot_ip_model(axis=ax_mdl, ip_model=true_model_2d,
                                 ip_modeltype='pelton',
                                 layer_ip=layer_IP_idx, rho2log=True)
ax_mdl.set_xlim((rho_min, rho_max))
ax_mdl.set_ylim((z_max, z_min))
ax_mdl.grid(which='major', alpha=0.75, ls='-')
ax_mdl.grid(which='minor', alpha=0.75, ls=':')
ax_mdl.invert_yaxis()


# %% add colorbar, labels, etc
# norm = mpl.colors.BoundaryNorm(eval_m, cmap.N) #, extend='both')

# # norm = plt.Normalize(np.min(eval_m), np.max(eval_m))
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# # sm.set_array(eval_m)

# divider = make_axes_locatable(ax)
# cax = divider.append_axes('right', size='5%', pad=0.05)
# cb = fig.colorbar(sm, cax=cax, orientation='vertical', format='%.1e')

# cb.set_label('m ()')
ax.legend(loc='lower left')
ax.set_xlabel('time (s)')
ax.set_ylabel(r"$\mathrm{d}\mathrm{B}_\mathrm{z}\,/\,\mathrm{d}t$ (V/m²)")
# ax.set_title(f'evaluating m - {short_name}: ' + cc_f_tex + '\n' + params_fixed)
# ax.set_title(f'evaluating m - {short_name} et al. model: {cc_f_tex}')

plt.tight_layout()


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

# %% save figs
if savefigs:
    fig.savefig(savepath + f'evaluate_chargeability_{version}.png', dpi=300)
    # fg.savefig(savepath + f'correlate_chargeability_{version}.png', dpi=300)

# %% older unused stuff
    # calculate without IP effect - set m to 0
    # mdl = ip_mdl.copy()
    # mdl[:,2] = np.zeros((mdl.shape[0],))

    # model_vec = mtrxMdl2vec(mdl)
    # curr_empym_frwrd.calc_response(model=model_vec,
    #                                ip_modeltype=ip_modeltype,
    #                                show_wf=False)
    # sig_noIP = curr_empym_frwrd.response
    # sig_noIP_error = simulate_error(relerr, abserr, sig_noIP)
    # rhoa_noIP = calc_rhoa(curr_empym_frwrd, sig_noIP)


    # %% plot empymod forward
    # plot_signal(axis=ax,
    #             time=t_mdld,
    #             signal=sig_noIP,
    #             color='gray',
    #             alpha=0.75,
    #             marker='.',
    #             sub0color='k',
    #             label='without IP effect (m=0)')


    # %% plotting spectra
    # if ip_modeltype == 'cole_cole':
    #     cmplx_con = CCM(rho0=ip_mdl[layer_ip, 1],
    #                     con0=ip_mdl[layer_ip, 2],
    #                     con8=ip_mdl[layer_ip, 3],
    #                     tau=ip_mdl[layer_ip, 4],
    #                     c=ip_mdl[layer_ip, 5],
    #                     f=frq2plot)
    # elif ip_modeltype == 'pelton':
    #     cmplx_con = PEM(rho0=ip_mdl[layer_ip, 1],
    #                     m=ip_mdl[layer_ip, 2],
    #                     tau=ip_mdl[layer_ip, 3],
    #                     c=ip_mdl[layer_ip, 4],
    #                     f=frq2plot)
    # elif ip_modeltype == 'cc_kozhe':
    #     cmplx_con = CCM_K(con0=ip_mdl[layer_ip, 2],
    #                       eps_s=ip_mdl[layer_ip, 3],
    #                       eps_8=ip_mdl[layer_ip, 4],
    #                       tau=ip_mdl[layer_ip, 5],
    #                       c=ip_mdl[layer_ip, 6],
    #                       f=frq2plot)
    # else:
    #     raise ValueError('not yet implemented')


    # figCC, axCC = plt.subplots(4,1, figsize=(6,8),
    #                            sharex=True)

    # omega_peak = get_omega_peak_PE(m=ip_mdl[layer_ip, 2],
    #                                tau=ip_mdl[layer_ip, 3],
    #                                c=ip_mdl[layer_ip, 4])

    # axCC[0].loglog(frq2plot, np.abs(1 / cmplx_con), '.k-')
    # axCC[1].loglog(frq2plot, -np.angle(1 / cmplx_con)*1000, '.k-')
    # axCC[2].loglog(frq2plot, cmplx_con.real, '.k-')
    # axCC[3].loglog(frq2plot, cmplx_con.imag, '.k-')
    
    # for axi in axCC:
    #     axi.axvline(omega_peak / (2*np.pi), ls='--', color='crimson',
    #                 label=r'$f^{peak}_{\phi P} = \frac{1}{2\pi\cdot\tau_P} \cdot \frac{1}{(1-m)^{1/c}}$' + ' = {:.1f} Hz'.format(omega_peak / (2*np.pi)))
    #     axi.axvline(1 / (ip_mdl[layer_ip, 3] * 2*np.pi),
    #                 ls='--', color='orange',
    #                 label=r'$f^{peak} = 1/(2\pi\cdot\tau)$' + ' = {:.1f} Hz'.format(1 / (ip_mdl[layer_ip, 3] * 2*np.pi)))
    #     # axi.axvline(1 / (ip_mdl[layer_ip, 3]),
    #     #             ls='--', color='blue',
    #     #             label=r'$\omega^{peak}_{\phi P} =  1/\tau$')

    # axCC[0].set_ylim(CC_res_lims)
    # axCC[0].set_ylabel(r'$|\rho|$ ($\Omega$m)')
    
    # axCC[1].set_ylim(CC_phi_lims)
    # axCC[1].set_ylabel(r'$-\phi$ (mrad)')

    # axCC[2].set_ylim(CC_sig1_lims)
    # axCC[2].set_ylabel(r"$\sigma' (S/m)$")

    # axCC[3].set_ylim(CC_sig2_lims)
    # axCC[3].set_ylabel(r"$\sigma'' (S/m)$")
    
    # axCC[3].set_xlim(CC_fre_lims)
    # axCC[3].set_xlabel('frequencies (Hz)')
    # axCC[1].legend()

    # at = AnchoredText(coleparams,
    #                   prop={'color': 'C2', 'fontsize': 12}, frameon=True,
    #                   loc='lower left')
    # at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    # axCC[0].add_artist(at)

    # axCC[0].set_title(f'{short_name} Model: ' + cc_f_tex)
    
    # plt.tight_layout()
