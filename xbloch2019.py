#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 13:31:51 2019

@author: dhigley

Code for keeping track of the dynamics of an X-ray pulse resonantly propagating
through a sample. The dynamics are calculated with the Maxwell-Bloch
equations.
"""

import numpy as np
import pandas as pd
import copy

# Physical constants
HBAR = 6.582E-1    # eV*fs
HBAR_EVOLVE = 1.055E-19   # J*fs
EPSILON_0 = 8.854188E-12    # C/(Vm)
C = 3E8*1E-15    #(m/fs)
DENSITY = 90.9*1E27      # (atoms/m^3)

def run_sim(times, E_in, system):
    delta_ts = np.diff(times)
    for ind_minus_1, delta_t in enumerate(delta_ts):
        system.evolve(delta_t, E_in[ind_minus_1])
    system.phot_result = get_phot_result(system)
    system.density_panda = density_to_panda(system)
    system.stim_efficiency = calculate_stim_efficiency(system.phot_result)
    return system

def process_sim_result(sim_result):
    """Calculate some derived quantities from simulation result
    """
    pass

def density_to_panda(system):
    """Convert recorded history of density matrix to pandas dataframe
    """
    history = system.history
    rho_dict = {}
    for state1 in np.arange(len(system.valence)):
        for state2 in np.arange(len(system.valence)):
            if state1 >= state2:
                rho_str = f'rho({state1}, {state2})'
                rho_dict[rho_str] = [i[state1, state2] for i in history['rho']]
    rho_panda = pd.DataFrame(rho_dict)
    return rho_panda

def calculate_stim_efficiency(sim_result):
    stim_region = (sim_result['phots'] > 0.5) & (sim_result['phots'] < 1.5)
    abs_region = (sim_result['phots'] > -0.5) & (sim_result['phots'] < 0.5)
    abs_change = np.imag(sim_result['polarization']*np.conj(sim_result['E_in']))
    stim_strength = np.trapz(abs_change[stim_region])
    abs_strength = -1*np.trapz(abs_change[abs_region])
    stim_efficiency = stim_strength/abs_strength
    return stim_efficiency

def get_phot_result(system):
    history = system.history
    phot_polarization = convert_time_to_phot(history['t'], history['polarization'])
    phot_E = convert_time_to_phot(history['t'], history['field_envelope'])
    k = system.omega_l/C
    thickness = 2E-9     # thickness in m
    phot_E_out = phot_E['phot_y']-thickness*phot_polarization['phot_y']*1j*k/(2*EPSILON_0)
    phot_results = {'phots': phot_polarization['phot'],
                    'polarization': phot_polarization['phot_y'],
                    'E_out': phot_E_out,
                    'E_in': phot_E['phot_y']}
    return phot_results

def convert_time_to_phot(t, y):
    """Convert input time (t) and amplitude (y) to photon energy domain
    """
    sample_rate = t[1]-t[0]
    freqs = np.fft.fftshift(np.fft.fftfreq(len(t), sample_rate))
    phot = HBAR*2*np.pi*freqs
    phot_y = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(y)))
    return {'phot': phot,
            'phot_y': phot_y}

class MultilevelBloch:
    """
    Calculate evolution of density matrix elements coupled to field.
    
    Attributes
    ----------
    H_0: array
        Material Hamiltonian without an applied field. Units of eV.
    mu: array
        Dipole operator.
    relaxation: array
        Phenomenological relaxation parameters. Units of 1/fs
    omega_l: float
        Central angular frequency of applied field.
    t: float
        Current time in simulation
    valence: array, boolean
        True for indicies where the state with the same index is a valence
        state.
    valene_states: array
        List of valence state indicies
    core_states: array
        List of core state indicies
    omega_ij: array
        Angular transition frequencies
    rho: array
        Density matrix
    polarization: complex float
        Current polarization. Units of C*m
    history: dictionary
        Saved values at previous times
    """
    
    def __init__(self, H_0, mu, relaxation, omega_l, t_0, valence):
        self.H_0 = H_0
        self.mu = mu
        self.relaxation = relaxation
        self.omega_l = omega_l
        self.t = t_0
        self.valence = valence
        states = np.arange(len(self.H_0))
        self.valence_states = states[self.valence]
        self.core_states = states[~np.array(self.valence)]
        self.omega_ij = np.zeros_like(H_0)
        for i in np.arange(len(H_0)):
            for j in np.arange(len(H_0)):
                self.omega_ij[i, j] = (H_0[i, i]-self.H_0[j, j])/HBAR
        self.rho = np.zeros_like(H_0, dtype=complex)
        self.rho[0, 0] = 1
        self.polarization = self._calculate_polarization()
        self.history = {'t': [self.t],
                        'field_envelope': [0],
                        'rho': [self.rho],
                        'polarization': [self.polarization]}
        
    
    def evolve(self, delta_t, field_env):
        R_phase = np.exp(1j*self.t*(self.omega_ij-self.omega_l))
        R = field_env*self.mu*R_phase/(HBAR_EVOLVE)
        new_rho = copy.copy(self.rho)
        new_rho = self._evolve_valence_valence(delta_t, field_env, new_rho, R)
        new_rho = self._evolve_valence_core(delta_t, field_env, new_rho, R)
        new_rho = self._evolve_core_core(delta_t, field_env, new_rho, R)
        new_rho = self._relax(delta_t, new_rho)
        self.rho = new_rho
        self.t += delta_t
        self.polarization = self._calculate_polarization()
        self._update_history(field_env)
                
    def _evolve_valence_valence(self, delta_t, field_env, new_rho, R):
        for v in self.valence_states:
            for v2 in self.valence_states:
                drho_dt = 0
                for c in self.core_states:
                    drho_dt += -1j*np.conj(R[c, v])*self.rho[c, v2]
                for c in self.core_states:
                    current_term = -1*R[c, v2]*self.rho[v, c]
                    drho_dt += -1j*current_term
                drho = drho_dt*delta_t
                new_rho[v, v2] += drho
        return new_rho
                
    def _evolve_valence_core(self, delta_t, field_env, new_rho, R):
        for v in self.valence_states:
            for c in self.core_states:
                drho_dt = 0
                for c2 in self.core_states:
                    drho_dt += -1j*np.conj(R[c2, v])*self.rho[c2, c]
                for v2 in self.valence_states:
                    drho_dt += 1j*np.conj(R[c, v2])*self.rho[v, v2]
                drho = drho_dt*delta_t
                new_rho[v, c] += drho
                new_rho[c, v] += np.conj(drho)
        return new_rho
    
    def _evolve_core_core(self, delta_t, field_env, new_rho, R):
        for c in self.core_states:
            for c2 in self.core_states:
                drho_dt = 0
                for v in self.valence_states:
                    drho_dt += -1j*R[c, v]*self.rho[v, c2]
                for v in self.valence_states:
                    drho_dt += 1j*np.conj(R[c2, v])*self.rho[c, v]
                drho = drho_dt*delta_t
                new_rho[c, c2] += drho
        return new_rho
    
    def _relax(self, delta_t, new_rho):
        relaxation = self.relaxation*self.rho
        drho = relaxation*delta_t
        new_rho += drho
        return new_rho
    
    def _update_history(self, field_env):
        self.history['t'].append(self.t)
        self.history['field_envelope'].append(field_env)
        self.history['rho'].append(self.rho)
        self.history['polarization'].append(self.polarization)
        
    def _calculate_polarization(self):
        pol = 0
        for i in self.valence_states:
            for j in self.core_states:
                pol += DENSITY*self.mu[i, j]*self.rho[j, i]*np.exp(-1j*self.omega_ij[j, i]*self.t)*np.exp(1j*self.omega_l*self.t)
        return pol