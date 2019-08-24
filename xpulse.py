"""Code for generating X-ray pulse profiles (particularly SASE)
"""

import numpy as np
import matplotlib.pyplot as plt

HBAR = 1
RMS_TO_FWHM = 1

def sase(pulse_duration=5, E0=707, bw=0.005, N_electrons=10**4):
    """Generate SASE pulse
    """
    # Make a frequency sample vector
    omega0 = E0/HBAR       # Central frequency
    domega = 0.25*np.pi/pulse_duration    # Frequency sampling
    omega = np.arange(omega0*(1-2*bw), omega0*(1+2*bw), domega)
    # Make energy sample vector
    energy = omega*HBAR
    # Get random arrival time of electrons
    tk = pulse_duration*np.random.random_sample(N_electrons)
    # Make electron beam current in frequency domain
    J_of_omega = [np.sum(np.exp(1j*omegai*tk)) for omegai in omega]
    # Calculate amplifier bandpass
    sig_A = omega0*bw/RMS_TO_FWHM
    amp_bandpass = np.exp(-(omega-omega0)**2/(2*sig_A**2))
    # Multiply electron beam current by amplifier bandpass to get
    # electric field in the frequency domain
    E_of_omega = np.sqrt(amp_bandpass)*J_of_omega
    # Inverse transform to get electric field in the time domain
    E_of_t = np.fft.ifft(E_of_omega)
    # Get intensities in time and frequency domains
    I_of_omega = np.abs(E_of_omega**2)
    I_of_t = np.abs(E_of_t**2)
    pulse = {'energy': energy,
             'I_of_omega': I_of_omega,
             'envelope': amp_bandpass}
    return pulse

def gauss(t, t0=0, sigma=0.05):
    """Generate Gaussian pulse
    """
    return np.exp(-1*(t-t0)**2/(2*sigma**2))
