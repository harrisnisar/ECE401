'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np

def sinusoid(frequency, phasor, duration, samplerate):
    '''
    timeaxis, signal = sinusoid(frequency, phasor, duration, samplerate)
    Generate a sinusoid.

    frequency (real scalar) - frequency of the sinusoid, in Hertz
    phasor (complex scalar) - magnitude times e^{j phase}
    duration (real scalar) - duration, in seconds
    samplerate (real scalar) - sampling rate, in samples/second
    timeaxis (array) - sample times, from 0 to duration, including endpoints
    signal (array) - the generated sinusoid, length = int(duration*samplerate+1)
    
    𝑥[𝑛]=ℜ{𝑧 * 𝑒xp(𝑗2𝜋𝑛𝑓/𝐹𝑠)}
    '''
    # how many samples we want to get 
    num_samples = samplerate * duration+1
    
    # an array that goes from 0 to num_sumples
    n = np.arange(num_samples)
    
    # timeaxis is just n / samplerate, converts from samples to time
    timeaxis = n / samplerate
    
    # 𝑥[𝑛]=ℜ{𝑧 * 𝑒xp(𝑗2𝜋𝑛𝑓/𝐹𝑠)}
    x_n = np.real(phasor * np.exp(1j * (2*np.pi) * n * frequency / samplerate))
    
    return timeaxis, x_n
    
def compute_aliasing(frequencies, phasors, samplerates):
    '''
    aliased_freqs, aliased_phasors = compute_aliasing(frequencies, phasors, samplerates)
    Find the frequency and phasor of sinusoid aliases.  All arguments should have same length.

    frequencies (real array) - frequencies of the sinusoids, in Hertz
    phasors (complex array) - magnitudes times e^{j phases}
    samplerates (real array) - sampling rates, in samples/second
    aliased_freqs (real array)  - frequencies at which sinusoids seems to occur, in Hertz
    aliased_phasors (complex array) - phasors with which sinusoids seems to occur
    
    ALIASING RULES:
    If  𝑓mod𝐹𝑠<(𝐹𝑠−𝑓)mod𝐹𝑠 , 
        the aliased frequency is 𝑓𝑎=𝑓mod𝐹𝑠
        and the phasor of the aliased sinusoid is the same as the phasor of the true sinusoid: 𝑎=𝑧

    If  𝑓mod𝐹𝑠>(𝐹𝑠−𝑓)mod𝐹𝑠 , 
        the aliased frequency is 𝑓𝑎=(𝐹𝑠−𝑓)mod𝐹𝑠
        and the phasor of the aliased sinusoid is the complex conjugate the phasor of the true sinusoid: 𝑧𝑎=𝑧∗
    '''
    # init the things aliased frequency and aliased phasors that we will return
    aliased_freqs = []
    aliased_phasors = []
    
    # for all the frequencies in that we are calculating aliasing
    for i, f in enumerate(frequencies):
        # the sample rate of that frequency
        Fs = samplerates[i]
        
        # ALIASING RULES FROM DOC STRING:
        if np.mod(f, Fs) < np.mod(Fs-f, Fs):
           aliased_freqs.append(np.mod(f, Fs))
           aliased_phasors.append(phasors[i])
        if np.mod(f, Fs) > np.mod(Fs-f, Fs):
           aliased_freqs.append(np.mod(Fs-f, Fs))
           aliased_phasors.append(np.conj(phasors[i]))
    
    # return after converting our lists to arrays                         
    return np.array(aliased_freqs), np.array(aliased_phasors)

def fourier_analysis(signal, number_of_coefficients):
    '''
    coefficients = fourier_analysis(signal, number_of_coefficients)
    Find the Fourier series coefficients using the discrete-time Fourier analysis formula.

    signal (array of length N_0) = one period of the signal
    number_of_coefficients (scalar) = number of coefficients to compute, starting with X_0
    coefficients (array of length=number_of_coefficients) = X_0 through X_{number_of_coefficients-1}
    
    𝑋𝑘=(1/𝑁0)sum{𝑛=0 to n=𝑁0−1}:(𝑥[𝑛] * exp(−𝑗 * (2𝜋) * 𝑘 * 𝑛 / 𝑁0))
    '''
    # signal
    x = np.array(signal)
    
    # N0 is the length of our signal
    N0 = int(len(x))
    
    # array of sample indicies from 0 to L-1
    n = np.arange(0, N0) 
   
    # empty list of phasors (to return)
    coefficients = []

    # for all the harmonics we want
    for k in range(number_of_coefficients):
        # particular harmonic over t
        Xk = (1/N0) * np.sum(x*np.exp(-1j * (2 * np.pi * k * n) / N0))
        coefficients.append(Xk)

    # make sure we return a numpy array
    return np.array(coefficients)
    

def interpolate(lowrate_signal, T, kernel_timeaxis, kernel):
    '''
    highrate_signal = interpolate(lowrate_signal, T, kernel_timeaxis, kernel)
    Use lowrate-to-highrate conversion to simulate discrete-to-continuous conversion.

    lowrate_signal (length-N array) - the lowrate signal
    T (scalar) - ratio of highrate/lowrate, i.e., number of output samples per input sample
    kernel_timeaxis (array) - sample times of the kernel, at the highrate
    kernel (array) - the interpolation kernel.  length(kernel)==length(kernel_timeaxis).
    highrate_signal (length-N*T array) - the highrate signal
    
    Note: in order to keep the output to only N*T samples, use modulo arithmetic for the 
    interpolation, e.g.,
    highrate_signal[np.mod(kernel_timeaxis+n*T, N*T)] += kernel * lowrate_signal[n]
    '''
    # N is the length of the low rate signal
    N = len(lowrate_signal)
    
    # length of the high rate signal is N * T
    len_highrate_signal = N * T
    
    # highrate signal starts as an array with length of high rate signal containing 0's 
    highrate_signal = np.zeros(len_highrate_signal)
    
    # for every point
    for n in range(N):
        # create an array that we will add to highrate signal
        add_to_highrate_signal = np.zeros(len_highrate_signal)
        
        # index is (𝑡+𝑛*𝑇) % (𝑁*𝑇)
        index = np.mod(kernel_timeaxis+n*T, N*T)
        
        # set for the index the kernel * lowrate signal at n
        add_to_highrate_signal[index] = kernel * lowrate_signal[n]
        
        # add to highrate_signal
        highrate_signal = highrate_signal + add_to_highrate_signal
    
    return highrate_signal

def rectangle(T):
    '''
    timeaxis, h = rectangle(T)
    Return a rectangle function of length T.

    T (scalar) - length, in samples
    timeaxis (length-T array) - sample indices, from 0 to T-1, corresponding to h
    h (length-T array) - the rectangle function
    '''
    
    timeaxis = np.arange(0,T)
    timeaxis_len = len(timeaxis)
    h = np.ones(timeaxis_len)
    return timeaxis, h

def triangle(T):
    '''
    timeaxis, h = triangle(T)
    Return a triangle function of length 2*T-1.

    T (scalar) - length of each side of the triangle, in samples
    timeaxis (array, length 2*T-1) - sample indices, from -(T-1) through (T-1)
    h (array, length 2*T-1) - the triangle function, 1 - abs(timeaxis)/T
    
    if −𝑇+1≤𝑛≤𝑇−1:
        h[n] = 1−|𝑡|/𝑇 
    else:
        h[n] = 0
    '''
    lower_lim = -T + 1
    upper_lim = T - 1
    
    timeaxis = np.arange(lower_lim, upper_lim + 1)
    
    h = 1-np.absolute(timeaxis)/T
    
    return timeaxis, h

def spline(T):
    '''
    timeaxis, h = spline(T)
    Return a continuous spline interpolator with continuous first deriviative.

    T (scalar) - the upsampling factor
    timeaxis (array, length 4*T-1) - sample indices, from -(2*T-1) through (2*T-1)
    h (array, length 4*T-1) - the cubic spline interpolation kernel
    '''
    
    timeaxis_len = 4 * T - 1
    timeaxis = np.arange(-(2*T-1), (2*T))
    h = []
    
    for t in timeaxis:
        if(np.absolute(t) <= T):
            to_append = 1 - (3/2) * np.power((np.absolute(t)/T),2) + (1/2) * np.power((np.absolute(t)/T),3)
            h.append(to_append)
        elif(T <= np.absolute(t) <= 2*T):
            to_append = (-3/2) * np.power((np.absolute(t)-2*T)/T,2) * ((np.absolute(t)-T)/T)
            h.append(to_append)
        else:
            h.append(0)
    h = np.array(h)
    
    return timeaxis, h

def sinc(T, D):
    '''
    timeaxis, h = sinc(T, D)
    Return D samples from the center of h(t)=sin(pi*t/T) / (pi*t/T).

    T (scalar) - the upsampling factor
    D (scalar) - the duration of the returned kernel; always an odd number.
    timeaxis (array, length D) - sample indices, from -(D-1)/2 through (D-1)/2
    h (array, length 4*T-1) - the sinc interpolation kernel
    '''
    lower_lim = -(D-1)/2
    upper_lim = ((D-1)/2) + 1
    timeaxis = np.arange(lower_lim, upper_lim)
    h = []

    for t in timeaxis:
        if(t==0):
            h.append(1)
        else:
            to_append = (np.sin(np.pi*t/T)) / ((np.pi * t)/T)
            h.append(to_append)
    h = np.array(h)
  
    return timeaxis, h


