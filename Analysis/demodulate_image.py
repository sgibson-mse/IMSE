#External Imports
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


#Internal imports
from Analysis.read_binary import load_binary
from Analysis.peak_find import indexes

def prepare_image(filename):

    images = load_binary(filename, FLC=True)

    image = images[:, :, 1].astype(np.float64)
    image = image / np.max(image)

    return image

def take_slice(image, pixel):

    image_slice = image[int(pixel), :]  # slice image at the center pixel

    return image_slice

#make the window

def make_window(image_slice, n_pixels):

    fft_column = np.fft.fftshift(np.fft.fft(image_slice))

    # find the peaks of the carrier frequency at /pm f and DC
    peaks = indexes(abs(fft_column), thres=0.2)

    fringe_peaks = indexes(abs(image_slice), thres=0.7, min_dist=10)

    fringe_width = int((fringe_peaks[1] - fringe_peaks[0]) / 2)

    window = np.zeros((n_pixels))

    #Design filter around the positive and negative carrier frequency

    window[peaks[0] - fringe_width: peaks[0] + fringe_width + 1] = signal.hann(fringe_width*2+1)
    window[peaks[-1] - fringe_width: peaks[-1] + fringe_width + 1 ] = signal.hann(fringe_width*2+1)

    return window, fringe_width, peaks, fft_column

def filter_image(window, fft_column):

    # just take half the window, and the positive frequency for filtering.
    window = window[int(len(window) / 2):]

    half_fftcolumn = fft_column[int(len(fft_column) / 2):]

    filtered_column = window * half_fftcolumn

    carrier_frequency = np.where(np.max(abs(filtered_column)))

    ifft_column = np.fft.ifft(filtered_column)

    intensity = 2 * abs(ifft_column)
    new_pixels = np.arange(0, ny, 2)

    phase = np.angle(ifft_column)

    return intensity, phase




#Extract the images from the binary file
filename = str(os.getcwd()) + '/sam_8.dat'

nx = 1280
ny = 1080
n_frames = 38
pixels = np.arange(0,ny,1)
image = prepare_image(filename)




def plot_image(image):

    plt.figure()
    plt.title('Ne $\lambda$ = 600nm, Calibration Image $\phi_{45^{\circ}}$')
    plt.imshow(image.T)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Normalised Intensity', rotation=90)
    plt.xlabel('x pixels')
    plt.ylabel('y pixels')
    plt.show()

    return

def plot_spectrogram(image):

    plt.figure()
    plt.plot(image[int(len(image)/2),:])
    plt.show()

    return

def plot_1dpowerspectrum(image_fft):
    plt.figure()
    plt.plot(abs(image_fft))
    plt.xlabel('y pixel')
    plt.ylabel('Intensity (arb units)')
    plt.title('Power spectrum - Ne $\lambda$ = 600nm $\phi_{45^{\circ}}$')
    plt.show()

    return

def plot_2dpowerspectrum(total_image_fft):

    plt.figure()
    plt.imshow(np.log10(abs(total_image_fft)))
    plt.xlabel('x pixels')
    plt.ylabel('y pixels')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Intensity', rotation=90)
    plt.show()

    return

def plot_peaks(image_slice, fringe_peaks):

    plt.figure()
    plt.plot(image_slice)
    plt.plot(fringe_peaks, image_slice[fringe_peaks], 'x', color='red')
    plt.show()

    return

def plot_output():
    plt.figure()
    plt.plot(abs(half_fftcolumn)/np.max(abs(half_fftcolumn)), label='data')
    plt.plot(window, label='window')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(phase*(180./np.pi))
    plt.show()

    plt.figure()
    plt.plot(image_slice, label='Original')
    plt.plot(new_pixels, intensity, label='Filtered')
    plt.legend()
    plt.show()

    return

