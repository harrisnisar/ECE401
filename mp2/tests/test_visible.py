import unittest, h5py, submitted
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np

# TestSequence
class TestStep(unittest.TestCase):
    @weight(6.25)
    def test_sinusoid(self):
        with h5py.File('solutions.hdf5','r') as f:
            timeaxis, signal = submitted.sinusoid(f['frequency'][0], f['phasor'][0],
                                                  f['duration'][0], f['samplerate'][0])
            self.assertAlmostEqual(
                np.average(np.abs(f['timeaxis']-timeaxis)), 0, places=3,
                msg= "\n*** sinusoid({},{},{},{}): timeaxis is incorrect".format(
                    f['frequency'][0], f['phasor'][0], f['duration'][0], f['samplerate'][0]
                )
            )
            self.assertAlmostEqual(
                np.average(np.abs(f['signal']-signal)), 0, places=3,
                msg= "\n*** sinusoid({},{},{},{}): signal is incorrect".format(
                    f['frequency'][0], f['phasor'][0], f['duration'][0], f['samplerate'][0]
                )
            )


    @weight(6.25)
    def test_compute_aliasing(self):
        with h5py.File('solutions.hdf5','r') as f:
            aliased_freqs,  aliased_phasors = submitted.compute_aliasing(f['frequencies'][:],
                                                                         f['phasors'][:],
                                                                         f['samplerates'][:])
            for n in range(len(aliased_freqs)):
                self.assertAlmostEqual(
                    aliased_freqs[n], f['aliased_freqs'][n], places=2,
                    msg= "\n*** compute_aliasing({},{},{}): aliased freq should be {}".format(
                        f['frequencies'][n], f['phasors'][n], f['samplerates'][n], f['aliased_freqs'][n]
                    )
                )
                self.assertAlmostEqual(
                    aliased_phasors[n], f['aliased_phasors'][n], places=2,
                    msg= "\n*** compute_aliasing({},{},{}): aliased phasor should be {}".format(
                        f['frequencies'][n], f['phasors'][n], f['samplerates'][n], f['aliased_phasors'][n]
                    )
                )

    @weight(6.25)
    def test_fourier_analysis(self):
        with h5py.File('solutions.hdf5','r') as f:
            coefficients = submitted.fourier_analysis(f['signal'][:], f['number_of_coefficients'][0])
            for n in range(f['number_of_coefficients'][0]):
                self.assertAlmostEqual(
                    coefficients[n], f['coefficients'][n], places=2,
                    msg= "\n*** fourier_anlaysis() coefficient {} should be {}".format(
                        n, f['coefficients'][n]
                    )
                )

    @weight(6.25)
    def test_triangle(self):
        with h5py.File('solutions.hdf5','r') as f:
            t_timeaxis, triangle = submitted.triangle(f['T'][0])
            self.assertAlmostEqual(
                np.average(np.abs(t_timeaxis-f['t_timeaxis'])), 0, places=3,
                msg="\n*** triangle({}) timeaxis is wrong by more than 0.001".format(f['T'][0])
            )
            self.assertAlmostEqual(
                np.average(np.abs(triangle-f['triangle'])), 0, places=3,
                msg="\n*** triangle({}) is wrong by more than 0.001".format(f['T'][0])
            )

    @weight(6.25)
    def test_rectangle(self):
        with h5py.File('solutions.hdf5','r') as f:
            r_timeaxis, rectangle = submitted.rectangle(f['T'][0])
            self.assertAlmostEqual(
                np.average(np.abs(r_timeaxis-f['r_timeaxis'])), 0, places=3,
                msg="\n*** rectangle({}) timeaxis is wrong by more than 0.001".format(f['T'][0])
            )
            self.assertAlmostEqual(
                np.average(np.abs(rectangle-f['rectangle'])), 0, places=3,
                msg="\n*** rectangle({}) is wrong by more than 0.001".format(f['T'][0])
            )

    @weight(6.25)
    def test_spline(self):
        with h5py.File('solutions.hdf5','r') as f:
            s_timeaxis, spline = submitted.spline(f['T'][0])
            self.assertAlmostEqual(
                np.average(np.abs(s_timeaxis-f['s_timeaxis'])), 0, places=3,
                msg="\n*** spline({}) timeaxis is wrong by more than 0.001".format(f['T'][0])
            )
            self.assertAlmostEqual(
                np.average(np.abs(spline-f['spline'])), 0, places=3,
                msg="\n*** spline({}) is wrong by more than 0.001".format(f['T'][0])
            )

    @weight(6.25)
    def test_sinc(self):
        with h5py.File('solutions.hdf5','r') as f:
            sinc_timeaxis, sinc = submitted.sinc(f['T'][0], f['samplerate'][0]-1)
            self.assertAlmostEqual(
                np.average(np.abs(sinc_timeaxis-f['sinc_timeaxis'])), 0, places=3,
                msg="\n*** sinc({},{}) timeaxis is wrong by more than 0.001".format(
                    f['T'][0], f['samplerate'][0]-1)
            )
            self.assertAlmostEqual(
                np.average(np.abs(sinc-f['sinc'])), 0, places=3,
                msg="\n*** sinc({},{}) is wrong by more than 0.001".format(
                    f['T'][0], f['samplerate'][0]-1)
            )

    @weight(6.25)
    def test_interpolate(self):
        with h5py.File('solutions.hdf5','r') as f:
            highrate_signal = submitted.interpolate(f['lowrate_signal'][:], f['T'][0],
                                                    f['t_timeaxis'][:], f['triangle'][:])
            self.assertAlmostEqual(
                np.average(np.abs(highrate_signal-f['highrate_signal'])), 0, places=3,
                msg="\n*** interpolate: highrate_signal is wrong by more than 0.001".format(f['T'][0])
            )
