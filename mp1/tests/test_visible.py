import unittest, h5py, submitted
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np

# TestSequence
class TestStep(unittest.TestCase):
    @weight(10)
    def test_level_to_amplitude(self):
        with h5py.File('solutions.hdf5','r') as f:
            amplitudes = submitted.level_to_amplitude(f['levels'])
            for n in range(len(amplitudes)):
                self.assertAlmostEqual(
                    f['amplitudes'][n],
                    amplitudes[n],
                    msg= "\n*** level_to_amplitude({}) should produce {}".format(
                        f['levels'][n], f['amplitudes'][n]
                    )
                )

    @weight(10)
    def test_create_phasors(self):
        with h5py.File('solutions.hdf5','r') as f:
            phasors = submitted.create_phasors(f['amplitudes'], f['phases'])
            for n in range(len(phasors)):
                self.assertAlmostEqual(
                    f['z'][n],
                    phasors[n],
                    msg= "\n*** create_phasors({},{}) should produce {}".format(
                        f['amplitudes'][n], f['phases'][n], f['z'][n]
                    )
                )

    @weight(10)
    def test_synthesize_note(self):
        with h5py.File('solutions.hdf5','r') as f:
            note = submitted.synthesize_note(
                f['z'],
                f['F0'][0],
                f['Fs'][0],
                f['d'][0]
            )
            for n in [ np.random.randint(0,len(note)) for i in range(10) ]:
                self.assertAlmostEqual(
                    f['note'][n],
                    note[n],
                    msg= "\n*** synthesize_note({}, {}, {}, {}), sample {}, should be {}".format(
                        f['z'][:], f['F0'][0], f['Fs'][0],
                        f['d'][0], n, f['note'][n]
                    )
                )

    @weight(10)
    def test_names_to_fundamentals(self):
        import note2f0
        names = list(note2f0.F0.keys())
        ref = list(note2f0.F0.values())
        hyp = submitted.names_to_fundamentals(names)
        for n in [ np.random.randint(0,len(ref)) for i in range(10) ]:
            self.assertAlmostEqual(
                ref[n],
                hyp[n],
                msg="\n*** names_to_fundamentals(['{}']) should produce {}".format(
                    names[n], ref[n]
                )
            )

    @weight(10)
    def test_synthesize_song(self):
        with h5py.File('solutions.hdf5','r') as f:
            song = submitted.synthesize_song(
                f['fundamentals'],
                f['beats'],
                f['bpm'][0],
                f['Fs'][0],
                f['z'])
            for n in [ np.random.randint(0,len(song)) for i in range(10) ]:
                self.assertAlmostEqual(
                    f['song'][n],
                    song[n],
                    msg="\n*** synthesize_song({},{},{},{},{}), sample {}, should be {}".format(
                        f['fundamentals'][:], f['beats'][:], f['bpm'][0],
                        f['Fs'][0], f['z'][:], n, f['song'][n]
                    )
                )
                
