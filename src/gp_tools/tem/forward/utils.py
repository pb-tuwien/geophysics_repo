# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 15:21:05 2022

utility function for empymod_frwrd_ip

@author: lukas aigner @ TU Wien, Research Unit Geophysics
"""
#%% Necessary imports

import numpy as np
import pandas as pd
import datetime
from typing import Union, Optional
import matplotlib
import matplotlib.pyplot as plt
from scipy.constants import epsilon_0
from scipy.special import roots_legendre
from scipy.interpolate import InterpolatedUnivariateSpline as iuSpline
from scipy.constants import mu_0

#%% Helper function for empymod_frwrd()

def scaling(signal):
    signal_min = np.min(np.abs(signal))
    signal_max = np.max(np.abs(signal))
    s = np.abs(signal_max / (10 * (np.log10(signal_max) - np.log10(signal_min))))
    return s

def arsinh(signal):
    s = scaling(signal)
    return np.log((signal/s) + np.sqrt((signal/s)**2 + 1))

def kth_root(x, k):
    """
    kth root, returns only real roots

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    k : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if k % 2 != 0:
        res = np.power(np.abs(x),1./k)
        return res*np.sign(x)
    else:
        return np.power(np.abs(x),1./k)

def reshape_model(model, nLayer, nParam):
    """
    function to reshape a 1D bel1d style model to a n-D model containing as
    many rows as layers and as many columns as parameters (thk + nParams)

    Parameters
    ----------
    model : np.array
        bel1D vector model:
            thk_lay_0
            thk_lay_1
            .
            .
            thk_lay_n-1
            param1_lay_0
            param1_lay_1
            .
            .
            param1_lay_n
            .
            .
            .
            param_n_lay_0
            param_n_lay_1
            .
            .
            param_n_lay_n
    nLayer : int
        number of layers in the model.
    nParam : int
        number of parameters in the model, thk also counts!!

    Returns
    -------
    model : np.array
        n-D array with the model params.

    """
    mdlRSHP = np.zeros((nLayer,nParam))
    i = 0
    for col in range(0, nParam):
        for row in range(0, nLayer):
            if col == 0 and row == nLayer-1:
                pass
            else:
                mdlRSHP[row, col] = model[i]
                i += 1
    model = mdlRSHP
    return model

def simulate_error(relerr, abserr, data):
    np.random.seed(42)
    rndm = np.random.randn(len(data))

    rand_error_abs = (relerr * np.abs(data) +
                  abserr) * rndm

    return rand_error_abs

#%% Preparation functions for empymod_frwrd()

def get_TEMFAST_rampdata(location, current_key='1A'):

    column_list = ['cable', 'side', 'turns', 'ramp_off']

    if current_key == '1A':
        if location == 'donauinsel':
            ramp_data_array = np.array([[  6. ,  1.5 , 1. , 0.15],
                                        [ 12. ,  3.  , 1. , 0.23],
                                        [ 25. ,  6.25, 1. , 0.4 ],
                                        [ 50. , 12.5 , 1. , 0.8 ],
                                        [100. , 25.  , 1. , 1.3 ]])

        elif location == 'salzlacken':
            ramp_data_array = np.array([[  8. ,  2.  , 1.  ,  0.21],
                                        [ 25. ,  6.25, 1.  ,  0.44],
                                        [ 50. , 12.5 , 1.  ,  0.8 ],
                                        [100. , 25.  , 1.  ,  1.3 ]])

        elif location == 'hengstberg':
            ramp_data_array = np.array([[ 25. ,  6.25 , 1. ,  0.44],
                                        [ 50. ,  12.5 , 1. ,  0.82]])

        elif location == 'sonnblick':
            ramp_data_array = np.array([[ 50. ,  12.5 , 1. ,  1.0],
                                        [100. ,  25.  , 1. ,  2.5],
                                        [200. ,  50.  , 1. ,  4.2]])

        else:
            raise ValueError('location of ramp data measurements is not available ...')

    elif current_key == '4A':
        if location == 'donauinsel':
            ramp_data_array = np.array([[  6. ,  1.5  , 1. ,  0.17],
                                        [ 25. ,  6.25 , 1. ,  0.45],
                                        [ 50. ,  12.5 , 1. ,  0.95],
                                        [100. ,  25.  , 1. ,  1.5],
                                        [400. , 100.  , 1. , 10.0]])

        elif location == 'salzlacken':
            ramp_data_array = np.array([[  8. ,  2.  , 1. ,  0.21],
                                        [ 25. ,  6.25, 1. ,  0.5 ],
                                        [ 50. , 12.5 , 1. ,  0.95],
                                        [100. , 25.  , 1. ,  1.5 ],
                                        [200. , 50.  , 1. ,  4.3 ],
                                        [400. ,100.  , 1. , 10.0]])

        elif location == 'hengstberg':
            ramp_data_array = np.array([[ 25. ,  6.25 , 1. ,  0.48],
                                        [ 50. ,  12.5 , 1. ,  0.98]])

        elif location == 'sonnblick':
            ramp_data_array = np.array([[ 50. ,  12.5 , 1. ,  1.15],
                                        [100. ,  25.  , 1. ,  2.70],
                                        [200. ,  50.  , 1. ,  5.10]])

        else:
            raise ValueError('location of ramp data measurements is not available ...')

    ramp_data = pd.DataFrame(ramp_data_array, columns=column_list)

    return ramp_data

def get_TEMFAST_timegates():
    tg_raw = np.array([[1.00000e+00, 3.60000e+00, 4.60000e+00, 4.06000e+00, 1.00000e+00],
       [2.00000e+00, 4.60000e+00, 5.60000e+00, 5.07000e+00, 1.00000e+00],
       [3.00000e+00, 5.60000e+00, 6.60000e+00, 6.07000e+00, 1.00000e+00],
       [4.00000e+00, 6.60000e+00, 7.60000e+00, 7.08000e+00, 1.00000e+00],
       [5.00000e+00, 7.60000e+00, 9.60000e+00, 8.52000e+00, 2.00000e+00],
       [6.00000e+00, 9.60000e+00, 1.16000e+01, 1.05300e+01, 2.00000e+00],
       [7.00000e+00, 1.16000e+01, 1.36000e+01, 1.25500e+01, 2.00000e+00],
       [8.00000e+00, 1.36000e+01, 1.56000e+01, 1.45600e+01, 2.00000e+00],
       [9.00000e+00, 1.56000e+01, 1.96000e+01, 1.74400e+01, 4.00000e+00],
       [1.00000e+01, 1.96000e+01, 2.36000e+01, 2.14600e+01, 4.00000e+00],
       [1.10000e+01, 2.36000e+01, 2.76000e+01, 2.54900e+01, 4.00000e+00],
       [1.20000e+01, 2.76000e+01, 3.16000e+01, 2.95000e+01, 4.00000e+00],
       [1.30000e+01, 3.16000e+01, 3.96000e+01, 3.52800e+01, 8.00000e+00],
       [1.40000e+01, 3.96000e+01, 4.76000e+01, 4.33000e+01, 8.00000e+00],
       [1.50000e+01, 4.76000e+01, 5.56000e+01, 5.14000e+01, 8.00000e+00],
       [1.60000e+01, 5.56000e+01, 6.36000e+01, 5.94100e+01, 8.00000e+00],
       [1.70000e+01, 6.36000e+01, 7.96000e+01, 7.16000e+01, 1.60000e+01],
       [1.80000e+01, 7.96000e+01, 9.56000e+01, 8.76000e+01, 1.60000e+01],
       [1.90000e+01, 9.56000e+01, 1.11600e+02, 1.03600e+02, 1.60000e+01],
       [2.00000e+01, 1.11600e+02, 1.27600e+02, 1.19600e+02, 1.60000e+01],
       [2.10000e+01, 1.27600e+02, 1.59600e+02, 1.43600e+02, 3.20000e+01],
       [2.20000e+01, 1.59600e+02, 1.91600e+02, 1.75600e+02, 3.20000e+01],
       [2.30000e+01, 1.91600e+02, 2.23600e+02, 2.07600e+02, 3.20000e+01],
       [2.40000e+01, 2.23600e+02, 2.55600e+02, 2.39600e+02, 3.20000e+01],
       [2.50000e+01, 2.55600e+02, 3.19600e+02, 2.85000e+02, 6.40000e+01],
       [2.60000e+01, 3.19600e+02, 3.83600e+02, 3.50000e+02, 6.40000e+01],
       [2.70000e+01, 3.83600e+02, 4.47600e+02, 4.14000e+02, 6.40000e+01],
       [2.80000e+01, 4.47600e+02, 5.11600e+02, 4.78000e+02, 6.40000e+01],
       [2.90000e+01, 5.11600e+02, 6.39600e+02, 5.70000e+02, 1.28000e+02],
       [3.00000e+01, 6.39600e+02, 7.67600e+02, 6.99000e+02, 1.28000e+02],
       [3.10000e+01, 7.67600e+02, 8.95600e+02, 8.28000e+02, 1.28000e+02],
       [3.20000e+01, 8.95600e+02, 1.02360e+03, 9.56000e+02, 1.28000e+02],
       [3.30000e+01, 1.02360e+03, 1.27960e+03, 1.15200e+03, 2.56000e+02],
       [3.40000e+01, 1.27960e+03, 1.53560e+03, 1.40800e+03, 2.56000e+02],
       [3.50000e+01, 1.53560e+03, 1.79160e+03, 1.66400e+03, 2.56000e+02],
       [3.60000e+01, 1.79160e+03, 2.04760e+03, 1.92000e+03, 2.56000e+02],
       [3.70000e+01, 2.04760e+03, 2.55960e+03, 2.30400e+03, 5.12000e+02],
       [3.80000e+01, 2.55960e+03, 3.07160e+03, 2.81600e+03, 5.12000e+02],
       [3.90000e+01, 3.07160e+03, 3.58360e+03, 3.32800e+03, 5.12000e+02],
       [4.00000e+01, 3.58360e+03, 4.09560e+03, 3.84000e+03, 5.12000e+02],
       [4.10000e+01, 4.09560e+03, 5.11960e+03, 4.60800e+03, 1.02400e+03],
       [4.20000e+01, 5.11960e+03, 6.14360e+03, 5.63200e+03, 1.02400e+03],
       [4.30000e+01, 6.14360e+03, 7.16760e+03, 6.65600e+03, 1.02400e+03],
       [4.40000e+01, 7.16760e+03, 8.19160e+03, 7.68000e+03, 1.02400e+03],
       [4.50000e+01, 8.19160e+03, 1.02396e+04, 9.21600e+03, 2.04800e+03],
       [4.60000e+01, 1.02396e+04, 1.22876e+04, 1.12640e+04, 2.04800e+03],
       [4.70000e+01, 1.22876e+04, 1.43356e+04, 1.33120e+04, 2.04800e+03],
       [4.80000e+01, 1.43356e+04, 1.63836e+04, 1.53600e+04, 2.04800e+03]])

    return pd.DataFrame(tg_raw, columns=['id', 'startT', 'endT', 'centerT', 'deltaT'])

def get_time(times, wf_times):
    """Additional time for ramp.

    Because of the arbitrary waveform, we need to compute some times before and
    after the actually wanted times for interpolation of the waveform.

    Some implementation details: The actual times here don't really matter. We
    create a vector of time.size+2, so it is similar to the input times and
    accounts that it will require a bit earlier and a bit later times. Really
    important are only the minimum and maximum times. The Fourier DLF, with
    `pts_per_dec=-1`, computes times from minimum to at least the maximum,
    where the actual spacing is defined by the filter spacing. It subsequently
    interpolates to the wanted times. Afterwards, we interpolate those again to
    compute the actual waveform response.

    Note: We could first call `waveform`, and get the actually required times
          from there. This would make this function obsolete. It would also
          avoid the double interpolation, first in `empymod.model.time` for the
          Fourier DLF with `pts_per_dec=-1`, and second in `waveform`. Doable.
          Probably not or marginally faster. And the code would become much
          less readable.

    Parameters
    ----------
    times : ndarray
        Desired times

    wf_times : ndarray
        Waveform times

    Returns
    -------
    time_req : ndarray
        Required times
    """
    tmin = np.log10(max(times.min()-wf_times.max(), 1e-10))
    tmax = np.log10(times.max()-wf_times.min())
    return np.logspace(tmin, tmax, times.size+2)

def waveform(times, resp, times_wanted, wave_time, wave_amp, nquad=3):
    """Apply a source waveform to the signal.

    Parameters
    ----------
    times : ndarray
        Times of computed input response; should start before and end after
        `times_wanted`.

    resp : ndarray
        EM-response corresponding to `times`.

    times_wanted : ndarray
        Wanted times. Rx-times at which the decay is observed

    wave_time : ndarray
        Time steps of the wave. i.e current pulse

    wave_amp : ndarray
        Amplitudes of the wave corresponding to `wave_time`, usually
        in the range of [0, 1].

    nquad : int
        Number of Gauss-Legendre points for the integration. Default is 3.

    Returns
    -------
    resp_wanted : ndarray
        EM field for `times_wanted`.

    """

    # Interpolate on log.
    PP = iuSpline(np.log10(times), resp)

    # Wave time steps.
    dt = np.diff(wave_time)
    dI = np.diff(wave_amp)
    dIdt = dI/dt

    # Gauss-Legendre Quadrature; 3 is generally good enough.
    # (Roots/weights could be cached.)
    g_x, g_w = roots_legendre(nquad)

    # Pre-allocate output.
    resp_wanted = np.zeros_like(times_wanted)

    # Loop over wave segments.
    for i, cdIdt in enumerate(dIdt):

        # We only have to consider segments with a change of current.
        if cdIdt == 0.0:
            continue

        # If wanted time is before a wave element, ignore it.
        ind_a = wave_time[i] < times_wanted
        if ind_a.sum() == 0:
            continue

        # If wanted time is within a wave element, we cut the element.
        ind_b = wave_time[i+1] > times_wanted[ind_a]

        # Start and end for this wave-segment for all times.
        ta = times_wanted[ind_a]-wave_time[i]
        tb = times_wanted[ind_a]-wave_time[i+1]
        tb[ind_b] = 0.0  # Cut elements

        # Gauss-Legendre for this wave segment. See
        # https://en.wikipedia.org/wiki/Gaussian_quadrature#Change_of_interval
        # for the change of interval, which makes this a bit more complex.
        logt = np.log10(np.outer((tb-ta)/2, g_x)+(ta+tb)[:, None]/2)
        fact = (tb-ta)/2*cdIdt
        resp_wanted[ind_a] += fact*np.sum(np.array(PP(logt)*g_w), axis=1)

    return resp_wanted

#%% Functions to parse different model to empymod

def cole_cole(inp, p_dict):
    """Cole and Cole (1941).
    code from: https://empymod.emsig.xyz/en/stable/examples/time_domain/cole_cole_ip.html#sphx-glr-examples-time-domain-cole-cole-ip-py
    """
    # Compute complex conductivity from Cole-Cole
    iwtc = np.outer(2j*np.pi*p_dict['freq'], inp['tau'])**inp['c']
    condH = inp['cond_8'] + (inp['cond_0']-inp['cond_8']) / (1 + iwtc)
    condV = condH/p_dict['aniso']**2

    # Add electric permittivity contribution
    etaH = condH + 1j*p_dict['etaH'].imag
    etaV = condV + 1j*p_dict['etaV'].imag
    return etaH, etaV

def pelton_res(inp, p_dict):
    """ Pelton et al. (1978).
    code from: https://empymod.emsig.xyz/en/stable/examples/time_domain/cole_cole_ip.html#sphx-glr-examples-time-domain-cole-cole-ip-py
    """
    # Compute complex resistivity from Pelton et al.
    # print('\n   shape: p_dict["freq"]\n', p_dict['freq'].shape)
    iwtc = np.outer(2j*np.pi*p_dict['freq'], inp['tau'])**inp['c']
    rhoH = inp['rho_0'] * (1 - inp['m']*(1 - 1/(1 + iwtc)))
    rhoV = rhoH*p_dict['aniso']**2

    # Add electric permittivity contribution
    etaH = 1/rhoH + 1j*p_dict['etaH'].imag
    etaV = 1/rhoV + 1j*p_dict['etaV'].imag
    return etaH, etaV

def mpa_model(inp, p_dict):
    """
    maximum phase angle model (Fiandaca et al 2018)
    Formula 8 - 11, appendix A.1 - A.08

    Parameters
    ----------
    inp : dictionary
        dictionary containing the cole-cole parameters:
            'rho_0' - DC resistivity
            'phi_max' - maximum phase angle, peak value of the phase of complex res (rad).
            'tau_phi' - relaxation time, specific for mpa model, see Formula 10 (s).
            'c' - dispersion coefficient
    p_dict : dictionary
        additional dictionary with empymod specific parameters.

    Returns
    -------
    etaH, etaV : dtype??

    """
    # obtain chargeability and tau from mpa model
    m, tau_rho = get_m_taur_MPA(inp['phi_max'], inp['tau_phi'], inp['c'], verbose=False)
    iwtc = np.outer(2j*np.pi*p_dict['freq'], tau_rho)**inp['c']

    # Compute complex resistivity using the pelton resistivity model
    rhoH = inp['rho_0'] * (1 - m*(1 - 1/(1 + iwtc)))
    rhoV = rhoH*p_dict['aniso']**2

    # Add electric permittivity contribution
    etaH = 1/rhoH + 1j*p_dict['etaH'].imag
    etaV = 1/rhoV + 1j*p_dict['etaV'].imag
    return etaH, etaV

def cc_eps(inp, p_dict):
    """
    Mudler et al. (2020) - after Zorin and Ageev, 2017 with HF EM part - dielectric permittivity
    """
    # Compute complex permittivity
    iwtc = np.outer(2j*np.pi*p_dict['freq'], inp['tau']) ** inp['c']
    # iwe0rhoDC = np.outer(2j*np.pi*p_dict['freq'], epsilon_0, inp["rho_DC"])
    iwe0rhoDC = np.outer(2j*np.pi*p_dict['freq'], inp["rho_DC"]) * epsilon_0
    eps_H = inp["epsilon_HF"] + ((inp["epsilon_DC"] - inp["epsilon_HF"]) / (1 + iwtc)) + (1 / iwe0rhoDC)
    eps_V = eps_H * p_dict['aniso']**2
    
    rho_H = inp["rho_DC"]
    rho_V = rho_H * p_dict['aniso']**2

    etaH = 1/rho_H + 1j*eps_H.imag
    etaV = 1/rho_V + 1j*eps_V.imag
    return etaH, etaV

def cc_con_koz(inp, p_dict):
    """
    compl. con from Kozhevnikov, Antonov (2012 - JaGP)
    using perm0 and perm8 - Formula 5
    # TODO: Test and finish method!!
    """
    # Compute complex permittivity,
    io = 2j * np.pi * p_dict['freq'] ## i*omega --> from frequency to angular frequency
    iwtc = np.outer(io, inp['tau'])**inp['c']

    etaH = (inp["sigma_0"] + np.outer(io, epsilon_0) *
            (inp["epsilon_8"] + ((inp["epsilon_s"] - inp["epsilon_8"]) / (1 + iwtc)))
            )
    etaV = etaH * p_dict['aniso']**2
    return etaH, etaV

def get_m_taur_MPA(phi_max, tau_phi, c, verbose=False, backend='loop',
                   max_iters=42, threshhold=1e-7):
    """
    function to obtain the classical cc params from the mpa ones:
        uses an iterative approach and stops once the difference between
        two consecutive m values equals 0
    (after Fiandaca et al, 2018), appendix A.1 - A.08

    Parameters
    ----------
    phi_max : float, numpy.ndarray
        maximum phase angle, peak value of the phase of complex res (rad).
    tau_phi : float, numpy.ndarray
        relaxation time, specific for mpa model, see Formula 10 (s).
    c : float (0 - 1), numpy.ndarray
        dispersion coefficient.
    verbose : boolean, default is True
        suppress output?
    backend : str, default is loop
        only for array-like inputs, whether to loop or
    max_iters : int, default is 42
        maximum number of iterations
    threshhold : float, default is 1e-7
        difference between consecutive iterations at which the solution
        will be accepted

    Raises
    ------
    ValueError
        in case the iteration doesn't converge after max_iters iters.

    Returns
    -------
    m : float ()
        chargeability (0-1).
    tau_rho : float (s)
        relaxation time.

    """
    th = threshhold

    m0s = []
    tau_rs = []
    areal = []
    bimag = []
    delta_ms = []

    if isinstance(phi_max, np.ndarray) and isinstance(tau_phi, np.ndarray) and isinstance(c, np.ndarray):
        if verbose:
            print('input MPA model params are all numpy array, converting all of them . . .')

        if backend == 'loop':
            phi_max_sub = np.copy(phi_max)
            tau_phi_sub = np.copy(tau_phi)
            c_sub = np.copy(c)

            if any(phi_max == 0):
                if verbose:
                    print('encountered phi_max == 0, assuming no-IP effect, setting m also to 0')
                mask = phi_max == 0
                phi_max_sub = phi_max[~mask]
                tau_phi_sub = tau_phi_sub[~mask]
                c_sub = c_sub[~mask]

            m, tau_rho = np.zeros_like(phi_max), np.zeros_like(tau_phi)
            ms, tau_rhos = np.zeros_like(phi_max_sub), np.zeros_like(tau_phi_sub)

            for i, phimax in enumerate(phi_max_sub):
                mns = []
                tau_rs = []
                areal = []
                bimag = []
                delta_ms = []

                for n in range(0, max_iters):
                    if n == 0:
                        mns.append(0)

                    tau_rs.append(mpa_get_tau_rho(m=mns[n],
                                                  tau_phi=tau_phi_sub[i],
                                                  c=c_sub[i]))
                    areal.append(mpa_get_a(tau_rs[n], tau_phi_sub[i], c_sub[i]))
                    bimag.append(mpa_get_b(tau_rs[n], tau_phi_sub[i], c_sub[i]))
                    mns.append(mpa_get_m(a=areal[n],
                                          b=bimag[n],
                                          phi_max=phimax))
                    delta_ms.append(mpa_get_deltam(mc=mns[n+1], mp=mns[n]))

                    if verbose:
                        print('delta_m: ', delta_ms[n])

                    if delta_ms[n] <= th:  # stop if the difference is below th
                        if verbose:
                            print(f'iteration converged after {n} iters')
                            print('solved m:', mns[n])
                            print('solved tau_rho:', tau_rs[n])
                        ms[i] = mns[n]
                        tau_rhos[i] = tau_rs[n]
                        break

            if any(phi_max == 0):
                m[~mask] = ms
                tau_rho[~mask] = tau_rhos
                
                tau_rho[mask] = mpa_get_tau_rho(m=0,
                                                tau_phi=tau_phi[mask],
                                                c=c[mask])
            else:
                m = ms
                tau_rho = tau_rhos

        elif backend == 'vectorized':
            # TODO
            raise ValueError('not yet implemented . . .')
            pass

        else:
            raise ValueError("Please select either 'loop' or 'vectorized' for the backend kwarg")

    elif isinstance(phi_max, float) and isinstance(tau_phi, float) and isinstance(c, float):
        if verbose:
            print('input MPA model params are all single floats . . .')

        if phi_max == 0:
            m = 0
            tau_rho = mpa_get_tau_rho(m=m, tau_phi=tau_phi, c=c)

        elif (phi_max > 0):
            for n in range(0, max_iters):
                if n == 0:
                    m0s.append(0)

                tau_rs.append(mpa_get_tau_rho(m=m0s[n],
                                              tau_phi=tau_phi,
                                              c=c))
                areal.append(mpa_get_a(tau_rs[n], tau_phi, c))
                bimag.append(mpa_get_b(tau_rs[n], tau_phi, c))
                m0s.append(mpa_get_m(a=areal[n],
                                      b=bimag[n],
                                      phi_max=phi_max))
                delta_ms.append(mpa_get_deltam(mc=m0s[n+1], mp=m0s[n]))

                if verbose:
                    print('delta_m: ', delta_ms[n])

                if delta_ms[n] <= th:  # stop if the difference is below the th
                    if verbose:
                        print(f'iteration converged after {n} iters')
                        print('solved m:', m0s[n])
                        print('solved tau_rho:', tau_rs[n])

                    m = m0s[n]
                    tau_rho = tau_rs[n]
                    break

            if delta_ms[n] > th:
                raise ValueError(f'the iterations did not converge after {max_iters} iterations, please check input!')

    else:
        raise ValueError('imputs have to be all floats or all numpy arrays')

    return m, tau_rho

def mpa_get_tau_rho(m, tau_phi, c):
    """
    after Fiandaca et al. (2018), Appendix A.05
    needs |1 - m|, otherwise values of m > 1 will result in nan!!

    Parameters
    ----------
    m : float ()
        chargeability (0-1).
    tau_phi : float
        relaxation time, specific for mpa model, see Formula 10 (s).
    c : float (0 - 1)
        dispersion coefficient.

    Returns
    -------
    tau_rho : float (s)
        relaxation time.

    """
    tau_rho = tau_phi * (abs(1 - m)**(-1/(2*c)))  # abs is essential here
    return tau_rho

def mpa_get_a(tau_rho, tau_phi, c):
    """
    after Fiandaca et al. (2018), Appendix A.06

    Parameters
    ----------
    tau_rho : float (s)
        relaxation time.
    tau_phi : float
        relaxation time, specific for mpa model, see Formula 10 (s).
    c : float (0 - 1)
        dispersion coefficient.

    Returns
    -------
    a : float
        real part of complex variable.

    """
    a = np.real(1 / (1 + (1j*(tau_rho / tau_phi))**c))
    return a

def mpa_get_b(tau_rho, tau_phi, c):
    """
    after Fiandaca et al. (2018), Appendix A.07

    Parameters
    ----------
    tau_rho : float (s)
        relaxation time.
    tau_phi : float
        relaxation time, specific for mpa model, see Formula 10 (s).
    c : float (0 - 1)
        dispersion coefficient.

    Returns
    -------
    b : float
        imaginary part of complex variable.

    """
    b = np.imag(1 / (1 + (1j*(tau_rho / tau_phi))**c))
    return b

def mpa_get_m(a, b, phi_max):
    """
    after Fiandaca et al. (2018), Appendix A.08

    Parameters
    ----------
    a : float
        real part of complex variable. see mpa_get_a
    b : float
        imaginary part of complex variable. see mpa_get_b
    phi_max : float
        maximum phase angle, peak value of the phase of complex res (rad).

    Returns
    -------
    m : float ()
        chargeability (0-1).

    """
    tan_phi = np.tan(-phi_max)
    m = tan_phi / ((1 - a) * tan_phi + b)
    return m

def mpa_get_deltam(mc, mp):
    """
    after Fiandaca et al. (2018), Appendix A.04

    Parameters
    ----------
    mc : float
        m of current iteration.
    mp : TYPE
        m of previous iteration.

    Returns
    -------
    float
        delta_m, difference between current and previous m.

    """
    return np.abs(mc - mp) / mc

def mpa_get_tau_rho(m, tau_phi, c):
    """
    after Fiandaca et al. (2018), Appendix A.05
    needs |1 - m|, otherwise values of m > 1 will result in nan!!

    Parameters
    ----------
    m : float ()
        chargeability (0-1).
    tau_phi : float
        relaxation time, specific for mpa model, see Formula 10 (s).
    c : float (0 - 1)
        dispersion coefficient.

    Returns
    -------
    tau_rho : float (s)
        relaxation time.

    """
    tau_rho = tau_phi * (abs(1 - m)**(-1/(2*c)))  # abs is essential here
    return tau_rho

#%% Functions to save sounding as an '.tem'-file
def get_temfast_date() -> str:
    """
    get current date and time and return temfast date string.
    eg. Thu Dec 30 09:34:11 2021

    Returns
    -------
    temf_datestr : str
        current date including name of day, month and adding year at the end.

    """
    tdy = datetime.datetime.today()
    time_fmt = ('{:02d}:{:02d}:{:02d}'.format(tdy.hour,
                                              tdy.minute,
                                              tdy.second))
    temf_datestr = ('{:s} '.format(tdy.strftime('%a')) +  # uppercase for long name of day
                    '{:s} '.format(tdy.strftime('%b')) +  # uppercase for long name of month
                    '{:d} '.format(tdy.day) + 
                    '{:s} '.format(time_fmt) + 
                    '{:d}'.format(tdy.year))
    return temf_datestr

def save_as_tem(
        savepath: str, template_fid: str,
        filename: str, metadata: dict, 
        setup_device: dict, properties_snd: dict,
        times: np.ndarray, signal: np.ndarray, 
        error: np.ndarray, rhoa: np.ndarray,
        save_rhoa: bool = True, append_to_existing: bool = False
        ) -> pd.DataFrame:
    """
    Function to save tem data as .tem file (TEM-FAST style).
    The signal and error are normalized and the time is converted to microseconds.

    Parameters
    ----------
    savepath : str
        Path where the .tem file should be saved.
    template_fid : str
        Path to template file.
    filename : str
        Name of file.
    metadata : dict
        Dictionary containing 'location', 'snd_name' (sounding name), 'comment', 'x', 'y', and 'z'.
    setup_device : dict
        Dictionary containing 'timekey', 'filter_powerline', 'txloop', and 'rxloop'.
    properties_snd : dict
        Dictionary containing 'current_inj' (injected current) and 'rampoff' (ramp off time in seconds).
    times : np.ndarray (s)
        Times at which the signal was measured.
    signal : np.ndarray (V)
        Measured voltages.
    error : np.ndarray (V)
        Observed/Modeled (abs.) data error.
    rhoa : np.ndarray (Ohmm)
        Apparent resistivity.
    save_rhoa : boolean, optional
        Whether to save rhoa to the .tem file. The default is True.
    append_to_existing : boolean, optional
        Whether to append to an already existing file. The default is False.

    Returns
    -------
    pd.DataFrame
    Dataframe containing the data in the format of the template.
    """
    myCols = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"]
    template = pd.read_csv(template_fid, names=myCols,
                           sep='\\t', engine="python")

    tf_date = get_temfast_date()
    template.iat[0,1] = tf_date                          # set date
    template.iat[1,1] = f'{metadata["location"]}'             # set location
    template.iat[2,1] = metadata["snd_name"]

    template.iat[3,1] = f'{setup_device["timekey"]}'
    template.iat[3,4] = 'ramp={:.2f} us'.format(properties_snd['rampoff']*1e6)
    template.iat[3,5] = 'I={:.1f} A'.format(properties_snd['current_inj'])
    template.iat[3,6] = 'FILTR={:d} Hz'.format(setup_device['filter_powerline'])

    template.iat[4,1] = '{:.3f}'.format(setup_device['txloop'])
    template.iat[4,3] = '{:.3f}'.format(setup_device['rxloop'])

    template.iat[5,1] = metadata["comments"]

    template.iat[6,1] = '{:+.3f}'.format(metadata["x"])  # x
    template.iat[6,3] = '{:+.3f}'.format(metadata["y"])       # y
    template.iat[6,5] = '{:+.3f}'.format(metadata["z"])       # z

    template.iat[7,1] = 'Time[us]'

    chnls_act = np.arange(1, len(times)+1)
    data_norm = signal * (setup_device['txloop']**2) / properties_snd['current_inj']
    err_norm = error * (setup_device['txloop']**2) / properties_snd['current_inj']

    # clear data first:
    chnls_id = len(times) + 8
    template.iloc[8:, :] = np.nan

    # add new data
    template.iloc[8:chnls_id, 0] = chnls_act
    template.iloc[8:chnls_id, 1] = times*1e6  # to us
    template.iloc[8:chnls_id, 2] = data_norm
    template.iloc[8:chnls_id, 3] = abs(err_norm)

    if save_rhoa:
        template.iloc[8:chnls_id, 4] = rhoa
        exp_fmt = '%d\t%.2f\t%.5e\t%.5e\t%.2f'
        data_fid = savepath + f'{filename}.tem'
    else:
        exp_fmt = '%d\t%.2f\t%.5e\t%.5e'
        data_fid = savepath + f'{filename}.tem'

    # write to file
    data4exp = np.asarray(template.iloc[8:, :], dtype=np.float64)
    data4exp = data4exp[~np.isnan(data4exp).all(axis=1)]
    data4exp = data4exp[:, ~np.isnan(data4exp).all(axis=0)]

    if append_to_existing:
        print('saving data to: ', data_fid)
        with open(data_fid, 'a') as fid:
            fid.write('\n')  # new line
        header = template.iloc[:8, :]
        header.to_csv(data_fid, header=None,
                        index=None, sep='\t', mode='a')

        with open(data_fid, 'a') as fid:
            np.savetxt(fid, X=data4exp,
                       header='', comments='',
                       delimiter='\t', fmt=exp_fmt)

    else:
        print('saving data to: ', data_fid)
        header = template.iloc[:8, :]
        header.to_csv(data_fid, header=None,
                        index=None, sep='\t', mode='w')

        with open(data_fid, 'a') as fid:
            np.savetxt(fid, X=data4exp,
                       header='', comments='',
                       delimiter='\t', fmt=exp_fmt)

    with open(data_fid) as file: # remove trailing spaces of each line
        lines = file.readlines()
        lines_clean = [l.strip() for l in lines if l.strip()]
    with open(data_fid, "w") as f:
        f.writelines('\n'.join(lines_clean))

    return template

#%% Plotting functions

def plot_signal(
        time: np.ndarray, signal: np.ndarray, 
        noise: Optional[np.ndarray] = None, ax: Optional[matplotlib.axes.Axes] = None,
        xlimits: Optional[tuple] = None, ylimits: Optional[tuple] = None,
        sub0col: str = 'k', show_sub0_label: bool = True,
        sub0_ms: Union[float, int] = 5, sub0_mew: Union[float, int] = 1, 
        sub0_mfc: str = 'none', **kwargs
        ) -> matplotlib.axes.Axes:
    """
    Method to plot a signal response with a predefined style.

    Parameters
    ----------
    time : np.ndarray
        The time values.
    signal : np.ndarray
        The signal values.
    noise: np.ndarray, optional
        The noise values. If set to None it is not plotted. The default is None.
    ax : matplotlib.axes.Axes, optional
        Use this if you want to plot to an existing ax.
        If set to None a new figure and ax is created.
        The default is None.
    xlimits : tuple or None, optional
        X-axis limits. The default is None (No limit is applied).
    ylimits : tuple or None, optional
        Y-axis limits. The default is None (No limit is applied).
    sub0col : str, optional
        Markeredgecolor for the sub zero marker. The default is 'k'.
    show_sub0_label : bool, optional
        Add a label to the sub zero values. The default is True.
    sub0_ms : float or int, optional
        Markersize for the sub zero marker. The default is 5.
    sub0_mew : float or int, optional
        Markeredgewidth for the sub zero marker. The default is 1.
    sub0_mfc: str, optional
        Markerfacecolor (fill) for the sub zero marker. The default is 'none' (no fill).
    **kwargs : dict
        Keyword arguments for the loglog() function.
    
    Returns
    -------
    matplotlib.axes.Axes
    Returns either the provided or the created ax.
    """
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1,
                                figsize=(7, 7))

    # select neg values, to mark them explicitly within the plot
    time_sub0 = time[signal < 0]
    sgnl_sub0 = signal[signal < 0]

    if kwargs.get('label', 'not_in_dict') != 'not_in_dict':
        label = kwargs.pop('label')
    else:
        label = None

    ax.loglog(time, abs(signal), label=label, **kwargs)
    if show_sub0_label:
        ax.loglog(time_sub0, abs(sgnl_sub0), marker='s', ls='none',
                    mfc=sub0_mfc, ms=sub0_ms, mew=sub0_mew,
                    mec=sub0col, label='negative vals')
    else:
        ax.loglog(time_sub0, abs(sgnl_sub0), marker='s', ls='none',
                    mfc=sub0_mfc, ms=sub0_ms, mew=sub0_mew,
                    mec=sub0col)
        
    if noise is not None:
        if kwargs.get('alpha') is not None:
            _ = kwargs.pop('alpha')
        if kwargs.get('marker') is not None:
            _ = kwargs.pop('marker')
        if kwargs.get('linestyle') is not None:
            _ = kwargs.pop('linestyle')
            
        ax.loglog(time, noise, linestyle='--', alpha=0.3, **kwargs)
    ax.set_xlabel(r'time (s)')
    ax.set_ylabel(r"$\mathrm{d}\mathrm{B}_\mathrm{z}\,/\,\mathrm{d}t$ (V/m²)")

    if xlimits is not None:
        ax.set_xlim(xlimits)
    if ylimits is not None:
        ax.set_ylim(ylimits)

    return ax

def plot_rhoa(
        time: np.ndarray, rhoa: np.ndarray, 
        noise: Optional[np.ndarray] = None, ax: Optional[matplotlib.axes.Axes] = None, 
        log_yaxis: bool = False, res2con: bool = False,
        xlimits: Optional[tuple] = None, ylimits: Optional[tuple] = None,
        sub0col: str = 'k', show_sub0_label: bool = True,
        sub0_ms: Union[float, int] = 5, sub0_mew: Union[float, int] = 1, 
        sub0_mfc: str = 'none', **kwargs
        ) -> matplotlib.axes.Axes:
    """
    Method to plot the apparent resistivity with a predefined style.

    Parameters
    ----------
    time : np.ndarray
        The time values.
    rhoa : np.ndarray
        The apparent resistivity values.
    noise: np.ndarray, optional
        The noise values. If set to None it is not plotted. The default is None.
    ax : matplotlib.axes.Axes object, optional
        Use this if you want to plot to an existing ax.
        If set to None a new figure and ax is created.
        The default is None.
    log_yaxis: boolean, optional
        If the y-axis should be set to a log-scale. The default is False.
    res2con: boolean, optional
        If the resistivity should be converted to conductivity (Ohm*m to mS/m). The default is False:
    xlimits : tuple or None, optional
        X-axis limits. The default is None (No limit is applied).
    ylimits : tuple or None, optional
        Y-axis limits. The default is None (No limit is applied).
    sub0col : str, optional
        Markeredgecolor for the sub zero marker. The default is 'k'.
    show_sub0_label : bool, optional
        Add a label to the sub zero values. The default is True.
    sub0_ms : float or int, optional
        Markersize for the sub zero marker. The default is 5.
    sub0_mew : float or int, optional
        Markeredgewidth for the sub zero marker. The default is 1.
    sub0_mfc: str, optional
        Markerfacecolor (fill) for the sub zero marker. The default is 'none' (no fill).
    **kwargs : dict
        Keyword arguments for the loglog() function.
    
    Returns
    -------
    matplotlib.axes.Axes
    Returns either the provided or the created ax.
    """
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1,
                                figsize=(7, 7))

    if res2con:
        app_val = 1000 / rhoa
        ylabel = r'$\sigma_a$ (mS/m)'
    else:
        app_val = rhoa
        ylabel = r'$\rho_a$ ($\Omega$m)'

    time_sub0 = time[app_val < 0]
    app_val_sub0 = app_val[app_val < 0]

    if kwargs.get('label', 'not_in_dict') != 'not_in_dict':
        label = kwargs.pop('label')
    else:
        label = None

    ax.semilogx(time, abs(app_val), label=label, **kwargs)
    if show_sub0_label:
        ax.semilogx(time_sub0, abs(app_val_sub0), marker='s', ls='none',
                    mfc=sub0_mfc, ms=sub0_ms, mew=sub0_mew, 
                    mec=sub0col, label='negative vals')
    else:
        ax.semilogx(time_sub0, abs(app_val_sub0), marker='s', ls='none',
                    mfc=sub0_mfc, ms=sub0_ms, mew=sub0_mew,
                    mec=sub0col)
        
    if noise is not None:
        if kwargs.get('alpha') is not None:
            _ = kwargs.pop('alpha')
        if kwargs.get('marker') is not None:
            _ = kwargs.pop('marker')
        if kwargs.get('linestyle') is not None:
            _ = kwargs.pop('linestyle')

        ax.semilogx(time, noise, linestyle='--', alpha=0.3, **kwargs)
    ax.set_xlabel(r'time (s)')
    ax.set_ylabel(ylabel)
    
    if xlimits is not None:
        ax.set_xlim(xlimits)
    if ylimits is not None:
        ax.set_ylim(ylimits)

    if log_yaxis:
        ax.set_yscale('log')

    return ax

def plot_data(
        time: np.ndarray, signal: np.ndarray, 
        rhoa: np.ndarray, ax=None,
        signal_title: Optional[str] = None, rhoa_title: Optional[str] = None, 
        signal_label: Optional[str] = None, rhoa_label: Optional[str] = None,
        noise_signal: Optional[np.ndarray] = None, noise_rhoa: Optional[np.ndarray] = None, 
        ylimits_signal: Optional[tuple] = None, ylimits_rhoa: Optional[tuple] = None, 
        xlimits: Optional[tuple] = None, log_rhoa: bool = False, 
        show_sub0_label: bool = False, res2con: bool = False, fontweight: Optional[str] = None,
        sub0_color_signal: str = 'k', sub0_color_rhoa: str = 'k', legend=True,
        **kwargs
        ) -> np.ndarray:
    """
    Method to plot the signal response and apparent resistivity with a predefined style.

    Parameters
    ----------
    time : np.ndarray
        The time values.
    signal : np.ndarray
        The signal values.
    rhoa : np.ndarray
        The apparent resistivity values.
    ax : array-like object, optional
        Use this if you want to plot to existing axes.
        Must contain two matplotlib.axes.Axes objects.
        If set to None a new figure and ax is created.
        The default is None.
    signal_title: str, optional
        The title of the ax with the signal subplot. If set to None a default title is set. The default is None.
    rhoa_title: str, optional
        The title of the ax with the rhoa subplot. If set to None a default title is set. The default is None.
    signal_label: str, optional
        Sets a label to the signal plot. The default is None (No label).
    rhoa_label: str, optional
        Sets a label to the rhoa plot. The default is None (No label).
    noise_signal: np.ndarray. optionall
        The noise values of the signal. If set to None it is not plotted. The default is None.
    noise_rhoa: np.ndarray, optional
        The noise values of the apparent resistivity. If set to None it is not plotted. The default is None.
    ylimits_signal : tuple or None, optional
        Y-axis limits for the signal subplot. The default is None (No limit is applied).
    ylimits_rhoa : tuple or None, optional
        Y-axis limits for the rhoa subplot. The default is None (No limit is applied).
    xlimits : tuple or None, optional
        X-axis limits. The default is None (No limit is applied).
    log_rhoa: boolean, optional
        If the y-axis of the rhoa plot should be set to a log-scale. The default is False.
    show_sub0_label : bool, optional
        Add a label to the sub zero values. The default is True.
    res2con: boolean, optional
        If the resistivity should be converted to conductivity (Ohm*m to mS/m). The default is False.
    fontweight: str, optional
        Fontweight of the title. The default is chosen by matplotlib.
    sub0_color_signal : str, optional
        Markeredgecolor for the sub zero marker in the signal plot. The default is 'k'.
    sub0_color_rhoa : str, optional
        Markeredgecolor for the sub zero marker in the rhoa plot. The default is 'k'.
    **kwargs : dict
        Keyword arguments for the plotting functions of both signal and rhoa plots (loglog() and semilogx()).
    
    Returns
    -------
    np.ndarray
    Returns either the provided or the created axes in a numpy array.
    """
    if ax is None:
        fig, ax = plt.subplots(
            nrows=1, ncols=2,
            figsize=(14, 7), constrained_layout=True
            )
    else:
        ax = np.asarray(ax)

    _ = plot_signal(
        time=time, signal=signal, 
        label=signal_label, noise=noise_signal, 
        ax=ax[0], xlimits=xlimits, 
        ylimits=ylimits_signal, show_sub0_label=show_sub0_label, 
        sub0col=sub0_color_signal, **kwargs
        )

    _ = plot_rhoa(
        time=time, rhoa=rhoa, 
        label=rhoa_label, noise=noise_rhoa, 
        ax=ax[1], log_yaxis=log_rhoa, 
        xlimits=xlimits, ylimits=ylimits_rhoa, 
        res2con=res2con, show_sub0_label=show_sub0_label, 
        sub0col=sub0_color_rhoa, **kwargs
        )
    
    if res2con:
        param = 'Conductivity'
    else:
        param = 'Resistivity'

    ax[0].grid(True, which="both", alpha=.3)
    ax[1].grid(True, which="both", alpha=.3)

    if signal_title is None:
        signal_title = 'Response Signal'
    ax[0].set_title(signal_title, fontsize=14, fontweight=fontweight)
    if rhoa_title is None:
        rhoa_title = f'Apparent {param}'
    ax[1].set_title(rhoa_title, fontsize=14, fontweight=fontweight)

    ax[1].yaxis.tick_right()
    ax[1].yaxis.set_label_position("right")
    
    if legend:
        if ax[0].get_legend_handles_labels()[1]:
                ax[0].legend(loc='lower left')
        if ax[1].get_legend_handles_labels()[1]:
                ax[1].legend(loc='lower left')

    return ax
    
def plot_model(
        model: np.ndarray, ax: Optional[matplotlib.axes.Axes] = None, 
        add_bottom: Union[float, int] = 0, title: Optional[str] = None,
        xlimits: Optional[tuple] = None, ylimits: Optional[tuple] = None,
        res2con: bool = False, fontweight: Optional[str] = None, **kwargs
        ) -> matplotlib.axes.Axes:
    """
    Method to plot the subsurface (resistivity) model with a predefined style.
    It plots a 1D model of the subsurface. 
    The provided model must have two dimensions: 
    The depth and the value of the plotted subsurface property. 

    The function was created to plot the resistivity but it can be used to plot other properties as well.
    If something other than the resistivity is plotted, make sure to set res2con to False (or leave it at the default value).

    Parameters
    ----------
    model: np.ndarray
        The model of the subsurface.
    ax : matplotlib.axes.Axes object, optional
        Use this if you want to plot to an existing ax.
        If set to None a new figure and ax is created.
        The default is None.
    add_bottom: float or int, optional
        By how much meters the last layer should be expanded for the plotting. The default is 0.
    title: str, optional
        Add a title to the plot. The default is None (No title added).
    xlimits : tuple or None, optional
        X-axis limits. The default is None (No limit is applied).
    ylimits : tuple or None, optional
        Y-axis limits. The default is None (No limit is applied).
    res2con: boolean, optional
        If the resistivity should be converted to conductivity (Ohm*m to mS/m). The default is False.
    fontweight: str, optional
        Fontweight of the title. The default is chosen by matplotlib.
    **kwargs : dict
        Keyword arguments for the plot() function.
    
    Returns
    -------
    matplotlib.axes.Axes
    Returns either the provided or the created ax.
    """
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1,
                                figsize=(3, 8), constrained_layout=True)

    if model[0, 0] != 0.0:
        dpths = np.r_[0, np.cumsum(model[:-1, 0])]
        res = model[:, 1]
        model = np.column_stack((dpths.repeat(2, 0)[1:],
                                    res.repeat(2, 0)[:-1]))

    res = np.r_[model[:, 1], model[-1, 1]]
    dpth = np.r_[model[:, 0], model[-1, 0] + add_bottom]

    if res2con:
        param = 1000 / res
        xlabel = r'$\sigma$ (mS/m)'
        title_default = 'Conductivity'
    else:
        param = res
        xlabel = r'$\rho$ ($\Omega$m)'
        title_default = 'Resistivity'
    
    ax.plot(param, dpth, **kwargs)
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
        
    ax.set_ylabel('Depth (m)')
    ax.set_xlabel(xlabel)

    if title is None:
        title = title_default
    ax.set_title(title, fontsize=14, fontweight=fontweight)

    ax.grid(True, which="both", alpha=.3)
    
    if xlimits is not None:
        ax.set_xlim(xlimits)
    if ylimits is not None:
        ax.set_ylim(ylimits)
    
    return ax