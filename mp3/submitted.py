import os, h5py
from PIL import Image
import numpy as  np

###############################################################################
# These are utility functions to generate gaussian and unit step signals
def gaussian(n, mu, var):
    '''
    Computes samples of a Gaussian, with mean=mu and variance=var,
    at the sample values given in the array n.
    '''
    p = (1/np.sqrt(2*np.pi*var))*np.exp(-0.5*(n-mu)*(n-mu)/var)
    return(p)

def unit_step():
    '''
    Generate 31 samples of a unit step function; return (n_step, x_step).
    '''
    M = 15
    n_step = np.linspace(-M,M,2*M+1,endpoint=True)
    x_step = np.concatenate((np.zeros(M), np.ones(M+1)))
    return(n_step,x_step)

###############################################################################
# TODO: here are the functions that you need to write
def todo_rectangular_filter():
    '''
    Create 11 samples of a 7-sample rectangular filter,
    h[n]=1/7 for -3 <= n <= 3, h[n]=0 otherwise.
    Return 11 samples of both n_rect and h_rect.
    '''
    n_rect = np.array([-5,-4,-3,-2,-1,0,1,2,3,4,5])
    len_n_rect = len(n_rect)
    h_rect = np.zeros(len_n_rect)
    for n in range(len_n_rect):
        if n_rect[n] <= 3 and n_rect[n] >= -3:
            h_rect[n] = 1/7

    return(n_rect,h_rect)

def todo_convolve_rows(X, h):
    '''
    Convolve Y=X*h along the rows.
    Use mode='same', so that the output, Y, has the same size as X.
    '''
    (nrows,ncols,ncolors) = X.shape
    Y = np.zeros((nrows,ncols,ncolors))
    for row in range(nrows):
        for color in range(ncolors):
            Y[row,:,color] = np.convolve(X[row,:,color], h, mode='same')
    return(Y)

def todo_convolve_columns(X, h):
    '''
    Convolve Y=X*h along the columns.
    Use mode='same', so that the output, Y, has the same size as X.
    '''
    (nrows,ncols,ncolors) = X.shape
    Y = np.zeros((nrows,ncols,ncolors))
    for col in range(ncols):
        for color in range(ncolors):
            Y[:,col,color] = np.convolve(X[:,col,color], h, mode='same')
    return(Y)

def todo_backward_difference():
    '''
    Create 11 samples of a backward difference filter, h[n],
    such that the convolution y=x*h gives, as output, y[n]=x[n]-x[n-1].
    Note that you'll need to set h[n]=0 for most n; only a couple of
    samples will be nonzero.
    Return the n_array and the h_array.
    '''
    n_diff = np.array([-5,-4,-3,-2,-1,0,1,2,3,4,5])
    len_n_diff = len(n_diff)

    h_diff = np.zeros(len_n_diff)
    for n in range(len_n_diff):
        if n_diff[n] == 0:
            h_diff[n] = 1
        elif n_diff[n] == 1:
            h_diff[n] = -1
    return(n_diff,h_diff)

def todo_gaussian_smoother():
    '''
    Create 11 samples of a unit-normal Gaussian filter.
    You might want to call the function gaussian(), which is defined at the
    top of this file.
    Return the n_array and the h_array.
    '''
    n_gauss = np.array([-5,-4,-3,-2,-1,0,1,2,3,4,5])
    h_gauss = gaussian(n_gauss, 0, 1)
    return(n_gauss,h_gauss)

def todo_difference_of_gaussians():
    '''
    Create 11 samples of a difference-of-Gaussians filter, h[n] is the
    difference between a unit-variance Gaussian centered at mu=0,
    minus a unit-variance Gaussian centered at mu=1.
    Return the n_array and the h_array.
    '''
    n_dog = np.array([-5,-4,-3,-2,-1,0,1,2,3,4,5])
    gaus_1 = h_gauss = gaussian(n_dog, 0, 1)
    gaus_2 = h_gauss = gaussian(n_dog, 1, 1)
    h_dog = gaus_1 - gaus_2
    return(n_dog,h_dog)

def todo_normalize_colors(X):
    '''
    Normalize the color planes of the image, so that
    each color plane has a maximum value of 1.0, and a minimum value of 0.0.
    '''
    (nrows,ncols,ncolors) = X.shape
    Y = np.zeros((nrows,ncols,ncolors))
    for color in range(ncolors):
        xmax = np.amax(X[:,:,color])
        xmin = np.amin(X[:,:,color])
        Y[:,:,color] = (X[:,:,color] - xmin) / (xmax-xmin)
    return(Y)

def todo_gradient_magnitude(GH, GV):
    '''
    Given the horizontal gradient GH and vertical gradient GV,
    compute and return the gradient magnitude GM=sqrt(GH**2 + GV**2)
    '''
    GH = np.array(GH)
    GV = np.array(GV)
    GM = np.sqrt(np.square(GH) + np.square((GV)))
    return(GM)

