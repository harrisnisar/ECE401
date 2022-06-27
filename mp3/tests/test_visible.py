import unittest, h5py, submitted
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np
from PIL import Image

# TestSequence
class TestStep(unittest.TestCase):
    @weight(6.25)
    def test_rectangular_filter(self):
        solutions=h5py.File('solutions.hdf5','r')
        (n_rect,h_rect) = submitted.todo_rectangular_filter()
        e = np.sum(np.abs(h_rect-solutions['h_rect']))/np.sum(np.abs(solutions['h_rect']))
        self.assertTrue(e < 0.04, 'todo_rectangular_filter wrong by more than 4% (visible case)')
    @weight(6.25)
    def test_convolve_rows(self):
        solutions=h5py.File('solutions.hdf5','r')
        original = np.asarray(Image.open('image.jpg')).astype('float64')
        h_rect = solutions['h_rect']
        smoothed_rows = submitted.todo_convolve_rows(original, h_rect)
        e = np.sum(np.abs(smoothed_rows-solutions['smoothed_rows']))/np.sum(np.abs(solutions['smoothed_rows']))
        self.assertTrue(e < 0.04, 'todo_convolve_rows wrong by more than 4% (visible case)')
    @weight(6.25)
    def test_convolve_columns(self):
        solutions=h5py.File('solutions.hdf5','r')
        smoothed_rows = solutions['smoothed_rows']
        h_rect = solutions['h_rect']
        smoothed_image = submitted.todo_convolve_columns(smoothed_rows, h_rect)
        e = np.sum(np.abs(smoothed_image-solutions['smoothed_image']))/np.sum(np.abs(solutions['smoothed_image']))
        self.assertTrue(e < 0.04, 'todo_convolve_columns wrong by more than 4% (visible case)')
    @weight(6.25)
    def test_backward_difference(self):
        solutions=h5py.File('solutions.hdf5','r')
        (n_diff,h_diff) = submitted.todo_backward_difference()
        e = np.sum(np.abs(h_diff-solutions['h_diff']))/np.sum(np.abs(solutions['h_diff']))
        self.assertTrue(e < 0.04, 'todo_backward_difference wrong by more than 4% (visible case)')
    @weight(6.25)
    def test_gaussian_smoother(self):
        solutions=h5py.File('solutions.hdf5','r')
        (n_gauss,h_gauss) = submitted.todo_gaussian_smoother()
        e = np.sum(np.abs(h_gauss-solutions['h_gauss']))/np.sum(np.abs(solutions['h_gauss']))
        self.assertTrue(e < 0.04, 'todo_gaussian_smoother wrong by more than 4% (visible case)')
    @weight(6.25)
    def test_difference_of_gaussians(self):
        solutions=h5py.File('solutions.hdf5','r')
        (n_dog,h_dog) = submitted.todo_difference_of_gaussians()
        e = np.sum(np.abs(h_dog-solutions['h_dog']))/np.sum(np.abs(solutions['h_dog']))
        self.assertTrue(e < 0.04, 'todo_difference_of_gaussians wrong by more than 4% (visible case)')
    @weight(6.25)
    def test_normalize_colors(self):
        solutions=h5py.File('solutions.hdf5','r')
        original = np.asarray(Image.open('image.jpg')).astype('float64')
        h_dog = solutions['h_dog']
        tmp = submitted.todo_convolve_rows(original, h_dog)
        hgrad = submitted.todo_normalize_colors(np.abs(tmp))
        e = np.sum(np.abs(hgrad-solutions['hgrad']))/np.sum(np.abs(solutions['hgrad']))
        self.assertTrue(e < 0.04, 'todo_normalize_colors wrong by more than 4% (visible case)')
    @weight(6.25)
    def test_gradient_magnitude(self):
        solutions=h5py.File('solutions.hdf5','r')
        vgrad = solutions['vgrad']
        hgrad = solutions['hgrad']
        tmp = submitted.todo_gradient_magnitude(hgrad, vgrad)
        gradient_magnitude=submitted.todo_normalize_colors(tmp)
        e = np.sum(np.abs(gradient_magnitude-solutions['gradient_magnitude']))/np.sum(np.abs(solutions['gradient_magnitude']))
        self.assertTrue(e < 0.04, 'todo_gradient_magnitude wrong by more than 4% (visible case)')
