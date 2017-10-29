# core
import itertools as it
from subprocess import call

# 3rd party
import matplotlib.pyplot as plt
import numpy as np

# local
import golden.rotations as rotations
from golden.config import *
from golden.simulation import custom_fft_abs, custom_fft_angle, crop_fft, create_image


def load_data(data_path):
    return np.loadtxt(data_path, skiprows=2, usecols=(1, 2, 3))


def bright_to_dark(im):
    im = -im
    im -= np.min(im)
    return im


def single_show(rx, ry, rz):
    X = load_data(DATAPATH)
    X = rotations.rotate(X, rx, ry, rz)
    im = create_image(X, PIXELS, DEPTH_RESOLUTION, DEPTH_OFFSET)
    plt.imshow(im, cmap='Greys')
    plt.show()


def save_series(folder, name, func):
    # creates a movie in folder with name
    # func is the function which creates the images
    ffolder = os.path.join(folder, name)
    filen = os.path.join(ffolder, name)
    os.mkdir(ffolder)
    X = load_data(DATAPATH)

    func(X, filen)

    callname = '/usr/bin/ffmpeg -r %f -i %s_%%d.png %s.avi' % (20, filen, filen)
    call(callname.split(' '))


def series1d(X, filen):
    rots = np.linspace(0, 2 * np.pi, 500, endpoint=False)
    for i in range(len(rots)):
        print(i)
        X_ = rotations.rotate(X, 0, rots[i], 0)
        im = create_image(X_, PIXELS, DEPTH_RESOLUTION, DEPTH_OFFSET)
        plt.imshow(im, cmap='Greys')
        plt.title('rot')
        plt.savefig('%s_%i' % (filen, i))
        plt.cla()


def create_2d_rot_plot(DATAPATH, rotations_per_axis=2, mode='normal'):
    # mode = 'normal', 'fftabs', 'fftangle'

    if mode == 'fftabs' or mode == 'fftangle':
        pxs = PIXELS - (2 * int(FFT_ZOOM_FACTOR * PIXELS))
    elif mode == 'normal':
        pxs = PIXELS
    else:
        raise AttributeError()

    X = load_data(DATAPATH)
    X -= - np.mean(X, axis=0)
    rotsx = np.linspace(0, .5 * np.pi, rotations_per_axis)
    rotsy = np.linspace(0, .5 * np.pi, rotations_per_axis)
    rots = list(it.product(rotsx, rotsy))
    indices = list(it.product(range(rotations_per_axis), range(rotations_per_axis)))
    final = np.zeros((rotations_per_axis * pxs, rotations_per_axis * pxs))
    for i in range(len(rots)):
        print(i)
        # print(indices[i][0], indices[i][1])
        # print(rots[i][0], rots[i][1])
        rotx, roty = rots[i][0], rots[i][1]
        indx, indy = indices[i][0], indices[i][1]
        X_ = rotations.rotate(X, rotx, roty, 0)
        im = create_image(X_, PIXELS, DEPTH_RESOLUTION, DEPTH_OFFSET, SCATTER_ALPHA)
        if mode == 'fftabs':
            fftim = crop_fft(custom_fft_abs(im))
            final[indx * pxs:(indx + 1) * pxs,indy * pxs:(indy + 1) * pxs] = fftim
        elif mode == 'fftangle':
            fftim = crop_fft(custom_fft_angle(im))
            final[indx * pxs:(indx + 1) * pxs,indy * pxs:(indy + 1) * pxs] = fftim
        elif mode == 'normal':
            final[indx * pxs:(indx + 1) * pxs,indy * pxs:(indy + 1) * pxs] = im
    final = bright_to_dark(final)
    plt.imshow(final, cmap='Greys')
    plt.yticks(pxs/2. + np.linspace(0, rotations_per_axis - 1, rotations_per_axis) * pxs, np.degrees(rotsx))
    plt.xticks(pxs/2. + np.linspace(0, rotations_per_axis - 1, rotations_per_axis) * pxs, np.degrees(rotsy))
    plt.xlabel('rotation around axis y'); plt.ylabel('rotation around axis x')
    plt.title(os.path.split(DATAPATH)[1]); plt.savefig(os.path.split(DATAPATH)[1] + '_' + mode + '.png', dpi=800)
    # plt.show()
    plt.cla()


def im_fft(DATAPATH):
    X = load_data(DATAPATH)
    im = create_image(X, PIXELS, DEPTH_RESOLUTION, DEPTH_OFFSET, SCATTER_ALPHA)
    im = bright_to_dark(im)
    fftim = custom_fft_abs(im)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.imshow(im, cmap='Greys')
    ax1.set_title('image')
    ax2.imshow(crop_fft(fftim),
               cmap='Greys')
    ax2.set_title('fft: log(abs(fft) + 1), zoomed into center')
    plt.savefig('fft_' + FILENAME + '.jpg')
    plt.cla()


def series3d(X, filen):
    rots = rotations.all_rots(3)
    for i in range(len(rots)):
        print(i)
        X_ = rotations.rotate(X, rots[i][0], rots[i][1], rots[i][2])
        im = create_image(X_, PIXELS, DEPTH_RESOLUTION, DEPTH_OFFSET, SCATTER_ALPHA)
        plt.imshow(im, cmap='Greys')
        plt.title('rot')
        plt.savefig('%s_%i' % (filen, i))
        plt.cla()


if __name__ == '__main__':
    for FILENAME in FILENAMES:
        DATAPATH = os.path.join(FOLDERPATH, FILENAME)
        create_2d_rot_plot(DATAPATH, rotations_per_axis=3, mode='fftangle')
    # im_fft()

    # single_show(np.radians(0), np.radians(0), np.radians(0))
    # single_show(np.radians(3), np.radians(0), np.radians(0))
    # single_show(np.radians(6), np.radians(0), np.radians(0))
