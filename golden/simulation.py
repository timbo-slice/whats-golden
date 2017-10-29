"""The code is based on the method presented by D.S. He, Z.Y. Li and J. Yuan in
 "Kinematic HAADF-STEM image simulation of small nanoparticles", 2015"""

# core

# 3rd party
import numpy as np
from scipy.ndimage.filters import gaussian_filter

# local
from golden.config import *
from golden.config import FFT_ZOOM_FACTOR, PIXELS


def create_image(X, pixs, depths_resolution, depths_offset, alpha):  # also give optional boundaries
    H_, edges = np.histogramdd(X, bins=[pixs - 2 * EXTRA_SIDE_PIXELS, pixs - 2 * EXTRA_SIDE_PIXELS, depths_resolution], normed=False)
    H = np.zeros((pixs, pixs, depths_resolution))
    H_filtered = np.zeros((pixs, pixs, depths_resolution))
    H[EXTRA_SIDE_PIXELS:-EXTRA_SIDE_PIXELS, EXTRA_SIDE_PIXELS:-EXTRA_SIDE_PIXELS, :] = H_
    for i in range(depths_resolution):
        d = np.abs(depths_offset + edges[2][i + 1])
        # print(d)
        # sigma = np.tan(alpha) * d
        factor = 1 - d * INTENSITY_DECAY
        if factor < 0:
            print('WARNING: Intensity negative')
        H_filtered[:, :, i] = H[:, :, i] * factor
        H_filtered[:, :, i] = gaussian_filter(H_filtered[:, :, i], SIGMA)

    im = np.sum(H_filtered, axis=2)
    return im


def custom_fft_abs(im):
    fftim = np.fft.fft2(im - np.mean(im))
    fftim = np.absolute(fftim)
    fftim = np.log(fftim + 1)
    fftim = np.fft.fftshift(fftim)
    return fftim


def custom_fft_angle(im):
    fftim = np.fft.fft2(im - np.mean(im))
    fftim = np.angle(fftim) + np.pi
    # fftim = np.mod(fftim, np.pi)
    return np.fft.fftshift(fftim)


def crop_fft(fftim):
    crop_ind = int(FFT_ZOOM_FACTOR * PIXELS)
    return fftim[crop_ind:-crop_ind,
                 crop_ind:-crop_ind]