import numpy as np

def todo_spectrum(x):
    '''
    Input:
    x (nsamps) - a waveform
    Output:
    spec (nsamps) - magnitude spectrum at nsamps frequencies between 0 and fs, not including fs.
    '''
    return np.abs(np.fft.fft(x))

def todo_findpeak(spec, fs, flo, fhi):
    '''
    Input:
    spec (nsamps) - magnitude spectrum at nsamps frequencies between 0 and fs, not including fs.
    fs (scalar)  - sampling frequency
    flo (scalar) -  low end of the frequency range to search
    fhi (scalar) - high end of the frequency range to search
    Output:
    fpeak (scalar) - frequency of the highest-magnitude spectral sample, in Hertz
    '''
    nsamps = len(spec)
    klo = int(flo*nsamps/fs)
    khi = int(fhi*nsamps/fs)
    peak = np.argmax(spec[klo:khi])+klo
    fpeak = peak* fs /nsamps
    return fpeak

def todo_zeros(freqs, fs):
    '''
    Input:
    freqs (nfreqs) - an array of nfreqs Hertz freqeuncies
    fs (scalar) - sampling frequency, in Hertz
    Output:
    z (2,nfreqs) - an array of complex zeros on the unit circle, in complex conjugate pairs
    '''
    nfreqs = len(freqs)
    omega = 2 * np.pi * np.array(freqs) / fs
    z1 = np.exp(1j*omega)
    z2 = np.exp(-1j*omega)
    z = np.stack((z1,z2))
    return z

def todo_poles(z,BW,fs):
    '''
    Input: 
    z (2,nfreqs) - an array of complex zeros on the unit circle, in complex conjugate pairs
    BW (scalar) - desired bandwidth, in Hertz
    fs (scalar) - sampling frequency, in Hertz
    Output:
    p (2,nfreqs) - an array of complex poles with bandwidth BW, in complex conjugate pairs
    '''
    _, nfreqs = z.shape
    B = 2 * np.pi * BW / fs
    a = np.exp(-B/2)
    p = a * z
    return p

def todo_coefficients(z):
    '''
    Input: 
    z (2,nfreqs) - an array of complex roots, in complex conjugate pairs
    Output:
    b (3,nfreqs) - an array of second-order polynomial coefficients, one per complex root pair
    '''
    _, nfreqs = z.shape
    b = np.ones((3,nfreqs))
    b[1,:] = -np.real(z[0,:] + z[1,:])
    b[2,:] = np.real(z[0,:]*z[1,:])
    return b
    
def todo_freqresponse(a, b, nsamps):
    '''
    Input: 
    a (3) - feedback coefficients.  You may assume a[0]=1.
    b (3) - feedforward coefficients.  You may assume b[0]=1.
    nsamps (scalar) - number of samples of the frequency response to compute
    Output: 
    omega (nsamps) - frequencies linearly spaced between 0 and 2pi, not including 2pi.
    H (nsamps) - B(e^{jw})/A(e^{jw}) evaluated at the frequencies in the vector omega.
    '''
    omega = np.linspace(0, 2*np.pi, nsamps, endpoint = False)
    z = np.exp(1j*omega)
    z_inv = 1/z
    A = 1 + a[1]*z_inv + a[2]*z_inv**2
    B = b[0] + b[1]*z_inv + b[2] * z_inv**2
    H = B/A
    return omega, H
    
def todo_filter(x, a, b):
    '''
    Input: 
    a (3) - feedback coefficients.  You may assume a[0]=1.
    b (3) - feedforward coefficients.  You may assume b[0]=1.
    x (nsamps) - input waveform
    Output: 
    y (nsamps) - output after being filtered using B(z)/A(z)
      Assume that x[n]==0 for n<0.
      Do not generate samples of y[n] for n >= nsamps.
    '''
    nsamps = len(x)
    y = np.zeros(nsamps)
    for n in range(nsamps):
        x_n = x[n]
        if(n-1<0):
            x_nminus1 = 0
            y_nminus1 = 0
        else:
            x_nminus1 = x[n-1]
            y_nminus1 = y[n-1]
            
        if(n-2<0):
            x_nminus2 = 0
            y_nminus2 = 0
        else:
            x_nminus2 = x[n-2]
            y_nminus2 = y[n-2]
        
        y_n = x_n + b[1]*x_nminus1 + b[2]*x_nminus2 - a[1]*y_nminus1-a[2]*y_nminus2
        y[n] = y_n
    return y
        

