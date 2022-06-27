import unittest, h5py, extra, note2f0
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np

# TestSequence
class TestStep(unittest.TestCase):
    @weight(5)
    def test_extra(self):
        with h5py.File('extra_solutions.hdf5','r') as f:
            F0 = f['F0'][0]
            Fs = f['Fs'][0]
            N = f['N'][0]
            phasors = extra.fourier_series_coefficients(f['x'][:], F0, Fs, N)
            for n in range(len(phasors)):
                self.assertAlmostEqual(
                    f['phasors'][n],
                    phasors[n],
                    msg= "\n*** fourier_series_coefficients({},{},{},{}) harmonic %d should be {}".format(
                        'extra_solutions.hdf5[x]', F0, Fs, N, n, f['phasors']
                    )
                )

