"""The code is based on the method presented by D.S. He, Z.Y. Li and J. Yuan in
 "Kinematic HAADF-STEM image simulation of small nanoparticles", 2015"""

# core

# 3rd party
import numpy as np
from scipy.ndimage.filters import gaussian_filter

# local
from golden.config import *


logger = logging.getLogger('logger')


def create_image(X):
    # TODO also give optional x y boundaries
    h_, edges = np.histogramdd(X, bins=[PIXELS - 2 * EXTRA_SIDE_PIXELS, PIXELS - 2 * EXTRA_SIDE_PIXELS, DEPTH_RESOLUTION], normed=False)
    h = np.zeros((PIXELS, PIXELS, DEPTH_RESOLUTION))
    h_filt = np.zeros((PIXELS, PIXELS, DEPTH_RESOLUTION))
    h[EXTRA_SIDE_PIXELS:-EXTRA_SIDE_PIXELS, EXTRA_SIDE_PIXELS:-EXTRA_SIDE_PIXELS, :] = h_
    min_d = np.min(X[:, 2]) + DEPTH_OFFSET  # Needs to be positive
    logger.debug('minimal distance between camera and specimen: %f' % min_d)
    if min_d <= 0:
        logger.warning('WARNING: distance between camera and closest atom negative')

    for i in range(DEPTH_RESOLUTION):
        d = np.abs(DEPTH_OFFSET + edges[2][i + 1])
        # sigma = np.tan(alpha) * d
        factor = 1 - d * INTENSITY_DECAY
        if factor < 0:
            logger.warning('WARNING: Intensity negative')
        h_filt[:, :, i] = h[:, :, i] * factor
        h_filt[:, :, i] = gaussian_filter(h_filt[:, :, i], SIGMA)

    logger.debug('final sum: %f' % np.sum(h_filt))
    im = np.sum(h_filt, axis=2)
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
    return np.fft.fftshift(fftim)


def crop_fft(fftim):
    crop_ind = int(FFT_ZOOM_FACTOR * PIXELS)
    return fftim[crop_ind:-crop_ind,
                 crop_ind:-crop_ind]