
# core
import os
from pdb import set_trace

FOLDERPATH = '/home/asdmueller/Desktop/mst/data/cords/'
FILENAMES = ['Au790_R2.0Rec69_ang.xyz', 'Au871_fcc.xyz',
             'Au887_N11_M7_K5_decaedro_Marks.xyz', 'Au923_ICO.xyz', 'Au967-cluster-hcp.xyz']
FILENAME = FILENAMES[0]

DATAPATH = os.path.join(FOLDERPATH, FILENAME)

PIXELS = 512  # PIXELS x PIXELS image
DEPTH_RESOLUTION = 25  # number of histogram bins in z-direction
INTENSITY_DECAY = .015  # intensity is computed as 1 - intensity_decay * z-distance
SIGMA = 5.
DEPTH_OFFSET = 20  # 0 means detector is in the middle of the specimen (unrealistic)
EXTRA_SIDE_PIXELS = 25  # Pixels added to the side to account for smearing of atoms at the edges of the image
FFT_ZOOM_FACTOR = .38  # the fft images are restricted to the middle (low freq) part.