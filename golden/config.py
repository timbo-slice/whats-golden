# core
import os
from pdb import set_trace  # imported here so it an be used throughout the module without explicit import

BASEPATH = os.path.join(os.path.split(os.path.abspath(__file__))[0], '..')
CORDSPATH = os.path.join(BASEPATH, 'data/in/cords/')

PICSPATH = os.path.join(BASEPATH, 'data/out/pics1/')

print(BASEPATH)
print(CORDSPATH)
print(PICSPATH)
if not os.path.exists(PICSPATH):
    os.makedirs(PICSPATH)

PIXELS = 512  # PIXELS x PIXELS image
DEPTH_RESOLUTION = 25  # number of histogram bins in z-direction
INTENSITY_DECAY = .015  # intensity is computed as 1 - intensity_decay * z-distance
SIGMA = 5.
DEPTH_OFFSET = 20  # 0 means detector is in the middle of the specimen (unrealistic)
EXTRA_SIDE_PIXELS = 25  # Pixels added to the side to account for smearing of atoms at the edges of the image
FFT_ZOOM_FACTOR = .38  # the fft images are restricted to the middle (low freq) part.