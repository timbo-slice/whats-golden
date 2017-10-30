# core
import os
from pdb import set_trace  # imported here so it an be used throughout the module without explicit import
import logging
import sys

BASEPATH = os.path.join(os.path.split(os.path.abspath(__file__))[0], '..')  # basepath, i.e. basepath/golden/config.py
CORDSPATH = os.path.join(BASEPATH, 'data/in/cords/')

PICSPATH = os.path.join(BASEPATH, 'data/out/pics2/')

if not os.path.exists(PICSPATH):
    os.makedirs(PICSPATH)

PIXELS = 512  # PIXELS x PIXELS image
DEPTH_RESOLUTION = 25  # number of histogram bins in z-direction
INTENSITY_DECAY = .015  # intensity is computed as 1 - intensity_decay * z-distance
SIGMA = 5.  # the std of the gaussian convolution in pixels TODO change to absolute value in angstrom
DEPTH_OFFSET = 25  # depth_offset = 0 means camera at the center of the specimen
EXTRA_SIDE_PIXELS = 25  # Pixels added to the side to account for smearing of atoms at the edges of the image
FFT_ZOOM_FACTOR = .38  # the fft images are restricted to the middle (low freq) part.


# declares the logger, sets the level, and sends output to stdout
logger = logging.getLogger('logger')
logger.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

"""
CRITICAL 	50
ERROR 	40
WARNING 	30
INFO 	20
DEBUG 	10
NOTSET 	0
"""