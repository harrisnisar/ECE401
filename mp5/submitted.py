import os, h5py, wave, struct
import numpy as  np

###############################################################################
# TODO: here are the functions that you need to write
def downsample_and_shift_dft2(oversampled_dft, downsampling_factor, row_shift, col_shift):
    '''
    Input: 
      oversampled_dft [M1,M2] - a 2d array containing the oversampled DFT of a grayscale image
      downsampling_factor (scalar) - the factor by which the DFT image is oversampled
      row_shift (scalar)  - the number of rows that the image should be shifted
      col_shift (scalar) - the number of columns that the image should be shifted
    Output: 
      image [M1/downsampling_factor, M2/downsampling_factor] - the real part of the inverse DFT
      of the valid frequency samples, shifted by the specified numbers of rows and columns.
    '''
    downsampled_mri_dft = oversampled_dft[::downsampling_factor, ::downsampling_factor].copy()
    N1, N2 = downsampled_mri_dft.shape

    for k1 in range(N1):
        if k1 < N1/downsampling_factor:
            downsampled_mri_dft[k1,:] *= np.exp(-2j*np.pi*k1*row_shift/N1)
        else:
            downsampled_mri_dft[k1,:] *= np.exp(-2j*np.pi*(k1-N1)*row_shift/N1)
    for k2 in range(N2):
        if k2 < N2/downsampling_factor:
            downsampled_mri_dft[:,k2] *= np.exp(-2j*np.pi*k2*col_shift/N2)
        else:
            downsampled_mri_dft[:,k2] *= np.exp(-2j*np.pi*(k2-N2)*col_shift/N2)
    
    return np.real(np.fft.ifft2(downsampled_mri_dft))
    
def dft_filter(noisy_image, min1, max1, min2, max2):
    '''
    Input: 
      noisy_image [N1,N2] - an image with narrowband noises
      min1, max1 (scalars) - zero out all rows of the DFT min1 <= k1 < max1, likewise  for N1-k1
      min2, max2 (scalars) - zero out all cols  of the DFT min2 <= k2 < max2, likewise for N2-k2
    Outut:
      cleaned_image [N1,N2] - image with the corrupted bands removed.
      Be sure to take the real part of the inverse DFT, and then truncate
      so that 0 <= cleaned_image[n1,n2,color] <= 1 for all n1,n2,color.
    '''
    noisy_image_dft = np.fft.fft2(noisy_image, axes=(0,1))
    N1, N2, N3 = noisy_image_dft.shape
 
    # zero out rows
    for k1 in range(N1):
        if (min1 <= k1 < max1) or (min1 <= N1-k1 < max1):
            noisy_image_dft[k1,:,:] = 0
 
    # zero out cols       
    for k2 in range(N2):
        if (min2 <= k2 < max2) or (min2 <= N2-k2 < max2):
            noisy_image_dft[:,k2,:] = 0
   
    # ifft
    image_to_return = np.real(np.fft.ifft2(noisy_image_dft, axes=(0,1)))
   
    # truncate
    I,J,K = image_to_return.shape
    for i in range(I):
        for j in range(J):
            for k in range(K):
                if(image_to_return[i,j,k]<0):
                    image_to_return[i,j,k] = 0
                elif(image_to_return[i,j,k]>1):
                    image_to_return[i,j,k] = 1
                    
    return image_to_return

def create_mirrored_image(image):
    '''
    Input:
      image [N1,N2,N3] - an original image that you  want to filter.
    Output:
      mirrored_image [2*N1, 2*N2, N3] - an image containing mirrored copies of the original,
      left-to-right and top-to-bottom, so that there is no sudden change of color at any of the
      original edges of the image.
    '''
    top = np.hstack((image,np.fliplr(image))).copy()
    bottom = np.flipud(top).copy()
    return np.vstack((top,bottom)).copy()
    
def transitioned_filter(noisy_image, min1, max1, min2, max2):
    '''
    Input: 
      noisy_image [N1,N2] - an image with narrowband noises
      min1, max1 (scalars) - zero out all rows of the DFT min1 <= k1 < max1, likewise  for N1-k1
      min2, max2 (scalars) - zero out all cols  of the DFT min2 <= k2 < max2, likewise for N2-k2
    Outut:
      cleaned_image [N1,N2] - image with the corrupted bands removed.
      Be sure to take the real part of the inverse DFT, and then truncate
      so that 0 <= cleaned_image[n1,n2,color] <= 1 for all n1,n2,color.

    Transition band:
      the bands k1=min1-1, k1=max1, k2=min2-1, and k2=max2 should be set to half of their
      original values, 0.5*X[k1,k2].
    '''
    noisy_image_dft = np.fft.fft2(noisy_image, axes=(0,1))
    N1, N2, N3 = noisy_image_dft.shape
 
    # zero out rows
    for k1 in range(N1):
        if (min1 <= k1 < max1) or (min1 <= N1-k1 < max1):
            noisy_image_dft[k1,:,:] = 0
        if(k1==min1-1):
            noisy_image_dft[k1,:,:] = noisy_image_dft[k1,:,:] / 2
        if(k1==max1):
            noisy_image_dft[k1,:,:] = noisy_image_dft[k1,:,:] / 2
 
    # zero out cols       
    for k2 in range(N2):
        if (min2 <= k2 < max2) or (min2 <= N2-k2 < max2):
            noisy_image_dft[:,k2,:] = 0
        if(k2==min2-1):
            noisy_image_dft[:,k2,:] = noisy_image_dft[:,k2,:] / 2
        if(k2==max2):
            noisy_image_dft[:,k2,:] = noisy_image_dft[:,k2,:] / 2
   
    # ifft
    image_to_return = np.real(np.fft.ifft2(noisy_image_dft, axes=(0,1)))
   
    # truncate
    I,J,K = image_to_return.shape
    for i in range(I):
        for j in range(J):
            for k in range(K):
                if(image_to_return[i,j,k]<0):
                    image_to_return[i,j,k] = 0
                elif(image_to_return[i,j,k]>1):
                    image_to_return[i,j,k] = 1
                    
    return image_to_return
