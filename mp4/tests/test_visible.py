import unittest, h5py, submitted
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np

# TestSequence
class TestStep(unittest.TestCase):
    def setUp(self):
        self.h5 = h5py.File('solutions.hdf5','r')
        with h5py.File('data.hdf5','r') as f:
            self.sfreq = f['sfreq'][0]
            self.onesecond = f['onesecond'][:]
        self.filterlength = 1
        self.evenlength = 200
        self.oddlength = 201
        self.omegaL = 0.785398163397448

    @weight(7.15)
    def test_lpf_even(self):
        h_even = submitted.todo_lpf_even(self.evenlength, self.omegaL)
        e = np.sum(np.abs(h_even-self.h5['h_even']))/np.sum(np.abs(self.h5['h_even']))
        self.assertTrue(e < 0.04, 'todo_lpf_even wrong by more than 4% (visible case)')

    @weight(7.15)
    def test_lpf_odd(self):
        h_odd = submitted.todo_lpf_odd(self.oddlength, self.omegaL)
        e = np.sum(np.abs(h_odd-self.h5['h_odd']))/np.sum(np.abs(self.h5['h_odd']))
        self.assertTrue(e < 0.04, 'todo_lpf_odd wrong by more than 4% (visible case)')

    @weight(7.14)
    def test_h_theta(self):
        h_theta = submitted.todo_h_theta(self.sfreq, self.filterlength)
        e = np.sum(np.abs(h_theta-self.h5['h_theta']))/np.sum(np.abs(self.h5['h_theta']))
        self.assertTrue(e < 0.04, 'todo_h_theta wrong by more than 4% (visible case)')

    @weight(7.14)
    def test_convolve(self):
        h_theta = self.h5['h_theta']
        thetawaves = submitted.todo_convolve(self.onesecond, h_theta)
        e = np.sum(np.abs(thetawaves-self.h5['thetawaves']))/np.sum(np.abs(self.h5['thetawaves']))
        self.assertTrue(e < 0.04, 'todo_convolve wrong by more than 4% (visible case)')

    @weight(7.14)
    def test_h_alpha(self):
        h_alpha = submitted.todo_h_alpha(self.sfreq, self.filterlength)
        alphawaves = submitted.todo_convolve(self.onesecond, h_alpha)
        e = np.sum(np.abs(h_alpha-self.h5['h_alpha']))/np.sum(np.abs(self.h5['h_alpha']))
        self.assertTrue(e < 0.04, 'todo_h_alpha wrong by more than 4% (visible case)')
        e = np.sum(np.abs(alphawaves-self.h5['alphawaves']))/np.sum(np.abs(self.h5['alphawaves']))
        self.assertTrue(e < 0.04, 'todo_h_alpha alphawaves by more than 4% (visible case)')

    @weight(7.14)
    def test_h_beta(self):
        h_beta = submitted.todo_h_beta(self.sfreq, self.filterlength)
        betawaves = submitted.todo_convolve(self.onesecond, h_beta)
        e = np.sum(np.abs(h_beta-self.h5['h_beta']))/np.sum(np.abs(self.h5['h_beta']))
        self.assertTrue(e < 0.04, 'todo_h_beta wrong by more than 4% (visible case)')
        e = np.sum(np.abs(betawaves-self.h5['betawaves']))/np.sum(np.abs(self.h5['betawaves']))
        self.assertTrue(e < 0.04, 'todo_h_beta betawaves by more than 4% (visible case)')

    @weight(7.14)
    def test_h_gamma(self):
        h_gamma = submitted.todo_h_gamma(self.sfreq, self.filterlength)
        gammawaves = submitted.todo_convolve(self.onesecond, h_gamma)
        e = np.sum(np.abs(h_gamma-self.h5['h_gamma']))/np.sum(np.abs(self.h5['h_gamma']))
        self.assertTrue(e < 0.04, 'todo_h_gamma wrong by more than 4% (visible case)')
        e = np.sum(np.abs(gammawaves-self.h5['gammawaves']))/np.sum(np.abs(self.h5['gammawaves']))
        self.assertTrue(e < 0.04, 'todo_h_gamma gammawaves by more than 4% (visible case)')

