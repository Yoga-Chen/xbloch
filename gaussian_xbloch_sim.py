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

def simulate_gauss_series():
    strengths = np.logspace(-2, 0.5, 5)
    sim_results = []
    for strength in strengths:
        sim_result = run_gauss_sim(strength)
        sim_results.append(sim_result)
    return sim_results

def plot_stim_efficiency(sim_results):
    strengths = []
    stim_efficiencies = []
    for i, sim_result in enumerate(sim_results):
        strengths.append(np.amax(sim_result.history['field_envelope']))
        stim_efficiency = sim_result.stim_efficiency
        stim_efficiencies.append(stim_efficiency)
    plt.figure()
    plt.plot(strengths, stim_efficiencies)
    plt.xlabel('Peak Intensity (W/cm$^2$)')
    plt.ylabel('Inelastic Stimulated Scattering Efficiency')

def plot_sim_spectra(sim_results):
    """Goal: plot simulated incident and transmitted X-ray spectra
    """
    strengths = np.logspace(-2, 0.5, 5)
    f, axs = plt.subplots(3, 1, sharex=True)
    phot0 = sim_results[0].phot_result
    linear_abs = (np.abs(phot0['E_out'])**2-np.abs(phot0['E_in'])**2)/strengths[0]
    abs_region = (phot0['phots'] > -1) & (phot0['phots'] < 1)
    abs_strength = -1*np.trapz(linear_abs[abs_region])
    incident_spectrum = np.abs(phot0['E_in'])
    norm_incident_spectrum = incident_spectrum/np.amax(incident_spectrum)
    #axs[0].plot(phot0['phots'], norm_incident_spectrum)
    stim_efficiencies = []
    for i, sim_result in enumerate(sim_results):
        phot_result = sim_result.phot_result
        strength = np.amax(sim_result.history['field_envelope'])
        transmitted_intensity_change = np.imag(phot_result['polarization']*np.conj(phot_result['E_in']))/strength
        #axs[1].plot(phot_result['phots'], transmitted_intensity_change/strength)
        axs[0].plot(phot_result['phots'], np.abs(phot_result['E_in'])**2/(strengths[i]))
        axs[1].plot(phot_result['phots'], np.abs(phot_result['E_out'])**2/(strengths[i]))
        spec_difference = (np.abs(phot_result['E_out'])**2-np.abs(phot_result['E_in'])**2)/(strengths[i])
        axs[2].plot(phot_result['phots'], spec_difference)
        stim_region = (phot_result['phots'] > 0.5) & (phot_result['phots'] < 1.5)
        change_from_linear = spec_difference-linear_abs
        stim_strength = np.trapz(change_from_linear[stim_region])
        stim_efficiency = stim_strength/abs_strength
        stim_efficiencies.append(stim_efficiency)
    axs[1].set_xlim((-5, 5))
    plt.figure()
    plt.plot(phot0['phots'], linear_abs)
    plt.figure()
    plt.plot(strengths, stim_efficiencies)
    plt.xlabel('Peak Intensity ($10^{15)$ W/cm$^2$)')
    plt.ylabel('
    
def calculate_stim_efficiency(sim_result):
    stim_region = (sim_result['phots'] > 0.5) & (sim_result['phots'] < 1.5)
    abs_region = (sim_result['phots'] > -0.5) & (sim_result['phots'] < 0.5)
    abs_change = np.imag(sim_result['polarization']*np.conj(sim_result['E_in']))
    stim_strength = np.trapz(abs_change[stim_region])
    abs_strength = -1*np.trapz(abs_change[abs_region])
    stim_efficiency = stim_strength/abs_strength
    return stim_efficiency

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
        axs[1].plot(phot_result['phots'], np.abs(phot_result['E_out'])/np.sqrt(strengths[i]))
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