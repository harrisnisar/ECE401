'''
If you finish this module, you can submit it for extra credit.
'''
import numpy as np

def fourier_series_coefficients(x, F0, Fs, N):
    '''
    phasors = fourier_series_coefficients(x, F0, Fs, N)
    
    x (numpy array) - the input signal
    F0 (scalar) - the fundamental frequency, in Hertz
    Fs (scalar) - sampling frequency, in samples/second
    N (scalar) - number of harmonics to measure
    phasors (length-N array) - Fourier series coefficients
    
    This should compute only the positive-frequency Fourier series coefficients
    (for k=1 through k=N).  Assume that the first sample of the input signal, x[0],
    is at time 0; the second sample, x[1], is at time 1/Fs, and so on.
    
    Instead of averaging over one period, you should average over the 
    whole signal length.  So you should multiply x by the complex 
    conjugate of each harmonic, then add over the whole signal length,
    then divide by the length of the signal.
    
    𝑋𝑘=1𝐿∑𝑛=0𝐿−1𝑥[𝑛]𝑒−𝑗2𝜋𝑘𝑛𝐹0/𝐹𝑠
    Xk = (1/L) * sum{from n=0 to n=L-1}(x[n] * exp(-j * 2pi * k * n * F0 / Fs))
    '''
    # make sure input is an array
    x = np.array(x)
    
    # length (L) of our signal
    L = int(len(x))
    
    # array of sample indicies from 0 to L-1
    t = np.arange(0, L) 

    # empty list of phasors (to return)
    phasors = []
    
    # for all the harmonics we want
    for k in range(N):
        # particular harmonic over t
        harmonic = np.exp(-1j * 2 * np.pi * (k+1) * t * F0 / Fs)
 
        # multiply each component of x with each component of harmonic and add
        phasor = x @ harmonic
        
        # divide by L
        phasor = phasor / L
        
        # save this particular harmonic's phasor
        phasors.append(phasor)
    
    # make sure we return a numpy array
    return np.array(phasors)
    
    