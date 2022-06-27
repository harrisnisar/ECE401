import unittest, h5py, submitted
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np

# TestSequence
class TestStep(unittest.TestCase):
    def setUp(self):
        self.h5 = h5py.File('solutions.hdf5','r')
        self.downsampling_factor = 2
        self.min1 = 16
        self.max1 = 19
        self.min2 = 29
        self.max2 = 32

    @weight(12.5)
    def test_mri(self):
        with h5py.File('data.hdf5','r')  as f:
            mri_dft = f['mri_dft'][:]
        N1,N2 = mri_dft.shape
        mri = submitted.downsample_and_shift_dft2(mri_dft, self.downsampling_factor, N1//4, N2//4)
        e = np.sum(np.abs(mri-self.h5['mri']))/np.sum(np.abs(self.h5['mri']))
        self.assertTrue(e < 0.04, 'downsample_and_shift_dft2 wrong by more than 4% (visible case)')

    @weight(12.5)
    def test_cleaned_image(self):
        with h5py.File('data.hdf5','r')  as f:
            noisy_image = f['noisy_image'][:]
        cleaned_image = submitted.dft_filter(noisy_image, self.min1, self.max1, self.min2, self.max2)
        e = np.sum(np.abs(cleaned_image-self.h5['cleaned_image']))/np.sum(np.abs(self.h5['cleaned_image']))
        self.assertTrue(e < 0.04, 'dft_filter wrong by more than 4% (visible case)')

    @weight(12.5)
    def test_mirrored_image(self):
        with h5py.File('data.hdf5','r')  as f:
            noisy_image = f['noisy_image'][:]
        mirrored_image = submitted.create_mirrored_image(noisy_image)
        e = np.sum(np.abs(mirrored_image-self.h5['mirrored_image']))/np.sum(np.abs(self.h5['mirrored_image']))
        self.assertTrue(e < 0.04, 'create_mirrored_image wrong by more than 4% (visible case)')

    @weight(12.5)
    def test_transitioned_image(self):
        with h5py.File('data.hdf5','r')  as f:
            noisy_image = f['noisy_image'][:]
        transitioned_image=submitted.transitioned_filter(noisy_image,self.min1,self.max1,
                                                         self.min2,self.max2)
        e = np.sum(np.abs(transitioned_image-self.h5['transitioned_image']))/np.sum(np.abs(self.h5['transitioned_image']))
        self.assertTrue(e < 0.04, 'transitioned_filter wrong by more than 4% (visible case)')

