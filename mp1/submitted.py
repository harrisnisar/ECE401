'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np

def level_to_amplitude(levels):
    '''
    amplitudes = level_to_amplitudes(levels)

    levels - any array or list of levels, in dB
    amplitudes - a numpy array or list, of the same length, containing amplitudes
    
    level=20log10(amplitudes)
    amplitudes=10^(level/20)
    '''
    # ensure we can handle list input or array input
    levels = np.array(levels)
    
    # amplitudes=10^(level/20)
    return np.power(10,levels/20)
    

def create_phasors(amplitudes, phases):
    '''
    phasors = create_phasors(amplitudes, phases)
    
    amplitudes - an array or list of amplitudes
    phases - an array or list of the same length, containing phase angles, in radians
    phasors - the resulting phasors = amplitude*exp(j*phases)
    '''
    # ensure we can handle list input or array input
    amplitudes = np.array(amplitudes)
    phases = np.array(phases)
    
    # phasors = amplitude * exp(j*phases)
    return amplitudes * np.exp(1j*phases)

def synthesize_note(z, F0, Fs, d):
    '''
    x = synthesize_note(z, F0, Fs, d)
    
    z (array of length N) - an array or list of phasors, giving the amplitude and phase of each harmonic
    F0 (scalar) - the fundamental frequency, in Hertz
    Fs (scalar) - the sampling frequency, in samples/second
    d (scalar) - the duration, in seconds
    x (array of length Fs*d) - the synthesized signal
    
    This function creates one harmonic for each element of z, then adds them together to generate x.
    
    𝑥(𝑡) = sum from 0<=k<=N-1: Real(𝑧[𝑘]*𝑒xp(𝑗*2𝜋*(𝑘+1)*𝐹0*𝑡) 
    '''
    # calculate the number of samples for our harmonics using Fs and d
    num_samples = Fs * d

    # create a vector t of num_samples points from 0 to d 
    # t = np.linspace(0.0, d, num=int(num_samples), endpoint=False)
    t = np.arange(0, int(num_samples)) / Fs
    
    # list of harmonics sampled at t
    harmonics = [z[k] * np.exp(1j * 2 * np.pi * (k+1) * F0 * t) for k in range(len(z))]
      
    # convert harmonics list to an array
    harmonics = np.array(harmonics)
    
    # sum the harmonics
    x = np.sum(np.real(harmonics), axis=0)
    
    # drop a dimension to meet the function requirement
    return np.squeeze(x)

import note2f0
def names_to_fundamentals(names):
    '''
    F0 = names_to_fundamentals(names)
    
    names - a list of names of notes, e.g., ['D#4','G#4','F4','G4','F4']
    fundamentals - a list of the corresponding fundamental frequencies, in Hertz, e.g., [311.13, 415.3, 349.23, 392.0, 349.23]
    
    print(note2f0.F0['C4'],note2f0.F0['A4'],note2f0.F0['C5'])
    '''
    # return a new array where each element is note corresponding to its fundamental freq ([Hz])
    return [note2f0.F0[name] for name in names]
    
def synthesize_song(fundamentals, beats, bpm, Fs, phasors):
    '''
    song = synthesize_song(notes, beats, bpm, Fs, phasors)
    
    fundamentals (array) - fundamental frequencies of the notes to be played
    beats (array) - duration of each note, in beats, e.g., [1,3/4,1/4,1,1]
    bpm (scalar) - number of beats per minute
    Fs (scalar) - number of samples per second
    phasors (list or array) - amplitude and phase of each harmonic
    song (numpy array)  - the returned song
    
    This function should first use beats and bpm to figure out the durations of the notes, in seconds.
    Then, for each note, it should:
        (1) call synthesize_note, to synthesize the note
        (2) call np.hanning, to create a hanning window, and multiply it by the note
        (3) call np.concatenate to concatenate the note  onto the song
    '''
    
    # convert beats per min to beats per sec
    #bps = bpm / 60
    
    # calculate duration of each note from beats array and bps
    durations = np.array(beats) * 60 / bpm
    
    # init return value
    song = None
    
    # for every requested frequency
    for i in range(len(fundamentals)):
        # grab the current note's frequency and duration
        fundamental = fundamentals[i]
        duration = durations[i]
        
        # call synthesize_note, to synthesize the note
        note = synthesize_note(phasors, fundamental, Fs, duration)
        
        # call np.hanning, to create a hanning window, and multiply it by the note
        note = note * np.hanning(len(note))
     
        # if song hasn't started (ie first note) set the song to the first generated note
        if i == 0:
            song = note
        # if the song has started (ie not first note), concatenate the generated note to the song
        else:
            song = np.concatenate((song, note))
    
    return song
