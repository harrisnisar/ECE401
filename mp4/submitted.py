import os, h5py, wave, struct
import numpy as  np

###############################################################################
# TODO: here are the functions that you need to write
def todo_lpf_even(nsamps, omegaL):
    '''
    Input: 
    nsamps = length of the filter (an even number)
    omegaL = cutoff frequency, in radians/sample
    Output: 
    h = impulse respose of a LPF with those parameters, 
    computed by multiplying a Hamming window times a delayed ideal LPF.
    '''
    h = np.zeros(nsamps)
    midpoint = (nsamps - 1) / 2
    win = np.hamming(nsamps)
    
    for n in range(nsamps):
        h[n] = win[n] * (omegaL/np.pi) * np.sinc(omegaL * (n-midpoint) / np.pi)

    return(h)
    
def todo_lpf_odd(nsamps, omegaL):
    '''
    Input: 
    nsamps = length of the filter (an odd number)
    omegaL = cutoff frequency, in radians/sample
    Output: 
    h = impulse respose of a LPF with those parameters, 
    computed by multiplying a Hamming window times a delayed ideal LPF.
    '''
    h = np.zeros(nsamps)
    midpoint = (nsamps - 1) / 2
    win = np.hamming(nsamps)
    
    for n in range(nsamps):
        h[n] = win[n] * (omegaL/np.pi) * np.sinc(omegaL * (n-midpoint) / np.pi)

    return(h)

def todo_h_theta(sfreq, filterlength):
    '''
    Input:
    sfreq (scalar): sampling frequency, samples per second
    filterlength (scalar): length of the filter, in seconds
    Output:
    h_theta: impulse response of a lowpass filter that retains the theta and delta bands
     (all bands below 7Hz), computed using todo_lpf_even or todo_lpf_odd,
    depending on whether filterlength turns out to be an odd or even number of samples.
    '''
    nsamps = int(filterlength*sfreq)
    h = np.zeros(nsamps)
    omega_c = (7 / sfreq) * 2 * np.pi 
    h_theta = todo_lpf_even(nsamps, omega_c)
    return(h_theta)

def todo_convolve(data, h):
    '''
    Input:
    data (nchannels,nsamps): nchannels different time-domain signals, each nsamps long
    h (filterlength): impulse response of the filter
    Output
    filtered (nchannels,nsamps+filterlength-1): results of filtering each row of data by h
    '''
    (nchannels,nsamps) = data.shape
    filterlength = len(h)
    filtered = np.zeros((nchannels,nsamps+filterlength-1))
    for channel in range(nchannels):
        filtered[channel,:] = np.convolve(data[channel,:],h)
    return(filtered)

def todo_h_alpha(sfreq, filterlength):
    '''
    Input:
    sfreq (scalar): sampling frequency, samples per second
    filterlength (scalar): length of the filter, in samples
    Output:
    h_alpha: impulse response of a bandpass filter that retains only the alpha bands
     (7-15Hz), computed by wisely making use of either todo_lpf_even or todo_lpf_odd,
    depending on whether filterlength turns out to be an odd or even number of samples.
    '''
    nsamps = int(filterlength*sfreq)
    h = np.zeros(nsamps)
    omega_2 = (15 / sfreq) * 2 * np.pi 
    omega_1 = (7 / sfreq) * 2 * np.pi 
    h = todo_lpf_even(nsamps, omega_2) - todo_lpf_odd(nsamps, omega_1)
    return(h)

def todo_h_beta(sfreq, filterlength):
    '''
    Input:
    sfreq (scalar): sampling frequency, samples per second
    filterlength (scalar): length of the filter, in samples
    Output:
    h_beta: impulse response of a bandpass filter that retains only the beta bands
     (15-30H),  computed by wisely making use of either todo_lpf_even or todo_lpf_odd,
    depending on whether filterlength turns out to be an odd or even number of samples.
    '''
    nsamps = int(filterlength*sfreq)
    h = np.zeros(nsamps)
    omega_2 = (30 / sfreq) * 2 * np.pi 
    omega_1 = (15 / sfreq) * 2 * np.pi 
    h = todo_lpf_even(nsamps, omega_2) - todo_lpf_odd(nsamps, omega_1)
    return(h)

def todo_h_gamma(sfreq, filterlength):
    '''
    Input:
    sfreq (scalar): sampling frequency, samples per second
    filterlength (scalar): length of the filter, in samples
    Output:
    h_theta: impulse response of a highpass filter that retains only the gamma bands
     (above 30Hz),  computed by wisely making use of either todo_lpf_even or todo_lpf_odd,
    depending on whether filterlength turns out to be an odd or even number of samples.
    The following information was not covered in lecture:
    If you need an even number of samples, construct a bandpassfilter insted of a highpass
    filter, but with an upper cutoff frequency of omegaL=pi.
    '''
    nsamps = int(filterlength*sfreq)
    h = np.zeros(nsamps)
    omega_2 = np.pi
    omega_1 = (30 / sfreq) * 2 * np.pi 
    h = todo_lpf_even(nsamps, omega_2) - todo_lpf_odd(nsamps, omega_1)
    return(h)

