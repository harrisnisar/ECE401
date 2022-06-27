import unittest, h5py, submitted
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np

# TestSequence
class TestStep(unittest.TestCase):
    def setUp(self):
        self.h5 = h5py.File('solutions.hdf5','r')
        self.fs = self.h5['fs'][0]

    @weight(7.15)
    def test_spectrum(self):
        spec=submitted.todo_spectrum(self.h5['x'])
        e = np.sum(np.abs(spec-self.h5['spec']))/np.sum(np.abs(self.h5['spec']))
        self.assertTrue(e < 0.04, 'todo_spec wrong by more than 4% (visible case)')

    @weight(7.15)
    def test_findpeak(self):
        noise1 = submitted.todo_findpeak(self.h5['spec'], self.fs, 0, 100)
        noise2 = submitted.todo_findpeak(self.h5['spec'], self.fs, noise1+1, 100)
        noise3 = submitted.todo_findpeak(self.h5['spec'], self.fs, 100, 110)
        noise4 = submitted.todo_findpeak(self.h5['spec'], self.fs, 100, 150)
        freqs = np.array([noise1,noise2,noise3,noise4])    
        e = np.sum(np.abs(freqs-self.h5['freqs']))/np.sum(np.abs(self.h5['freqs']))
        self.assertTrue(e < 0.04, 'todo_findpeak wrong by more than 4% (visible case)')

    @weight(7.14)
    def test_zeros(self):
        z= submitted.todo_zeros(self.h5['freqs'], self.fs)
        e = np.sum(np.abs(z-self.h5['z']))/np.sum(np.abs(self.h5['z']))
        self.assertTrue(e < 0.04, 'todo_zeros wrong by more than 4% (visible case)')

    @weight(7.14)
    def test_poles(self):
        p= submitted.todo_poles(self.h5['z'],20,self.fs)
        e = np.sum(np.abs(p-self.h5['p']))/np.sum(np.abs(self.h5['p']))
        self.assertTrue(e < 0.04, 'todo_poles wrong by more than 4% (visible case)')

    @weight(7.14)
    def test_coefficients(self):
        a = submitted.todo_coefficients(self.h5['p'])
        b = submitted.todo_coefficients(self.h5['z'])
        e = np.sum(np.abs(a-self.h5['a']))/np.sum(np.abs(self.h5['a']))
        self.assertTrue(e < 0.04, 'todo_coefficients: "a" wrong by more than 4% (visible case)')
        e = np.sum(np.abs(b-self.h5['b']))/np.sum(np.abs(self.h5['b']))
        self.assertTrue(e < 0.04, 'todo_coefficients: "b" wrong by more than 4% (visible case)')

    @weight(7.14)
    def test_freqresponse(self):
        nsamps = len(self.h5['x'])
        H = np.ones(nsamps,dtype='complex')
        for k in range(len(self.h5['freqs'])):
            omega, Hnew = submitted.todo_freqresponse(self.h5['a'][:,k], self.h5['b'][:,k], nsamps)
            H *= Hnew
        e = np.sum(np.abs(H-self.h5['H']))/np.sum(np.abs(self.h5['H']))
        e2 = np.sum(np.abs(H-self.h5['H_wrong']))/np.sum(np.abs(self.h5['H_wrong']))
        self.assertTrue((e<0.04) or (e2<0.04), 'todo_freqresponse by more than 4% (visible case)')

    @weight(7.14)
    def test_filter(self):
        y = self.h5['x'][:]
        for k in range(len(self.h5['freqs'])):
            y = submitted.todo_filter(y, self.h5['a'][:,k], self.h5['b'][:,k])
        e = np.sum(np.abs(y-self.h5['y']))/np.sum(np.abs(self.h5['y']))
        self.assertTrue(e < 0.04, 'todo_filter gammawaves by more than 4% (visible case)')

