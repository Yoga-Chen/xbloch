#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 15:30:22 2019

@author: dhigley

Functions for initializing model multilevel systems
"""

import numpy as np

import xbloch2019

# Physical constants
HBAR = 6.582E-1    # eV*fs
# Model material parameters:
DIPOLE = (4.07E-12+0j)*1.602E-19    # Dipole matrix element to use
Co_L3_BROAD = 0.43    # Natural lifetime of Co 2p_{3/2} core hole

def create_two_level_system(t0):
    """Return two-level system
    """
    H_0 = np.zeros((2, 2))
    H_0[0, 0] = 0
    H_0[1, 1] = 778
    mu = np.zeros((2, 2), dtype=np.complex)
    mu[0, 1] = DIPOLE
    mu = (mu+np.transpose(mu))/2
    relaxation = np.zeros((2, 2))
    relaxation[1, 1] = -1*Co_L3_BROAD/HBAR
    relaxation[0, 1] = -0.5*Co_L3_BROAD/HBAR
    relaxation[1, 0] = -0.5*Co_L3_BROAD/HBAR
    omega_field = 778/HBAR
    valence = [True, False]
    system = xbloch2019.MultilevelBloch(H_0, mu, relaxation, omega_field, t0, valence)
    return system

def create_three_level_system(t0):
    """Return three-level system
    """
    H_0 = np.zeros((3, 3))
    H_0[0, 0] = 0
    H_0[1, 1] = 1
    H_0[2, 2] = 778
    mu = np.zeros((3, 3), dtype=np.complex)
    mu[0, 2] = DIPOLE
    mu[2, 1] = DIPOLE*np.sqrt(3)
    mu = (mu+np.transpose(mu))
    relaxation = np.zeros((3, 3))
    relaxation[2, 2] = -1*Co_L3_BROAD/HBAR
    relaxation[0, 2] = -0.5*Co_L3_BROAD/HBAR
    relaxation[2, 0] = -0.5*Co_L3_BROAD/HBAR
    relaxation[2, 1] = -0.5*Co_L3_BROAD/HBAR
    relaxation[1, 2] = -0.5*Co_L3_BROAD/HBAR
    omega_field = 778/HBAR
    valence = [True, True, False]
    system = xbloch2019.MultilevelBloch(H_0, mu, relaxation, omega_field, t0, valence)
    return system