#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 15:41:08 2019

@author: dhigley

Run X-ray Maxwell-Bloch simulations for Gaussian pulse
"""

import numpy as np
import matplotlib.pyplot as plt

import model_multilevel
import xbloch2019

HBAR = 6.582E-1    # eV*fs
FIELD_1E15 = 8.68E10   # Electric field strength in V/m that corresponds to
# 10^15 W/cm^2

def simulate_and_plot_gauss_series():
    strengths = np.logspace(-2, 0.5, 5)
    sim_results = []
    for strength in strengths:
        sim_result = run_gauss_sim(strength)
        sim_results.append(sim_result)
    f, axs = plt.subplots(2, 1, sharex=True)
    for i, sim_result in enumerate(sim_results):
        phot_result = sim_result.phot_result
        axs[0].plot(phot_result['phots'], np.abs(phot_result['E_in'])/np.sqrt(strengths[i]))
        axs[1].plot(phot_result['phots'], np.imag(phot_result['polarization']*np.conj(phot_result['E_in']))/(strengths[i]))
    plt.figure()
    stim_strengths = []
    for i, sim_result in enumerate(sim_results):
        stim_strength = sim_result.stim_efficiency
        stim_strengths.append(stim_strength)
    plt.plot(strengths, stim_strengths)
    plot_density_matrix(sim_results[-1])

def run_gauss_sim(strength=0.1E2, duration=0.33, times=np.linspace(-1000, 1000, 100E3)):
    t0 = times[0]
    system = model_multilevel.create_three_level_system(t0)
    E_in = FIELD_1E15*np.sqrt(strength)*gauss(times, 0, sigma=duration)
    sim_result = xbloch2019.run_sim(times, E_in, system)
    return sim_result

def gauss(x, t0, sigma):
    gaussian = np.exp((-1/2)*((x-t0)/sigma)**2)
    return gaussian

def plot_density_matrix(system):
    """Plot history of density matrix elements
    """
    rho_panda = system.density_panda
    rho_panda.plot(subplots=True)