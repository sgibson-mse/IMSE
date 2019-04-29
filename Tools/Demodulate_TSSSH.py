import numpy as np
import pandas as pd
from IMSE.Tools.Plotting.graph_format import plot_format

import matplotlib.pyplot as plt

plot_format()

def load_image(filename):
    image_file = pd.HDFStore(filename)
    image = image_file['/a']
    return image

def hanning_window(image):
    h = np.hamming(len(image))
    b = np.ones((np.shape(image)))
    return (b.T*h).T

def apply_hanning(image):
    window = hanning_window(image)
    return window*image

def fft_2D(image):
    return np.fft.fft2(image)

def get_carrier_frequency(image):

    image_fft = fft_2D(image)
    shift_image = np.fft.fftshift(image_fft) #shift so DC frequency is in the center.

    # plt.figure()
    # plt.imshow(np.log10(abs(shift_image)))
    # plt.colorbar()
    # plt.clim(10,12.5)
    # plt.show()

    return shift_image

def box_filter(image, x_size, y_size, centre):

    mask = np.zeros((np.shape(image)))

    mask[centre[1] - int(((y_size/2)-0.5)): centre[1] + int((y_size/2)+0.5), centre[0]- int((x_size/2)-0.5):centre[0] + int((x_size/2)+0.5)] = 1.

    return mask

def filter_image(mask, shift_image):

    filtered_image = shift_image*mask

    plt.figure()
    plt.imshow(np.log10(abs(filtered_image)))
    plt.show()

    ifft_image = np.fft.ifft2(filtered_image)
    phase = np.arctan2(ifft_image.imag,ifft_image.real)
    amplitude = abs(ifft_image) # I0*contrast/2

    return phase, amplitude

def phase_mod(x,y):
    return x - np.floor((x+y/2.)/y)*y

def demodulate_image(image):

    window_image = apply_hanning(image)

    shift_image = get_carrier_frequency(window_image)

    mask = box_filter(image, x_size=int(len(image)), y_size=100, centre=[int(len(image)/2),433])
    dc_mask = box_filter(image, x_size=int(len(image)), y_size=100, centre=[int(len(image)/2), int(len(image)/2)])
    phase, amplitude = filter_image(mask, shift_image)

    dc_phase, dc_amplitude = filter_image(dc_mask, shift_image)

    contrast = (2*amplitude)/(dc_amplitude)

    return phase, contrast, dc_amplitude

# Get the synthetic images

def demodulate_images(image_1, image_2):

    #Demodulate images

    phase_45, contrast_45, dc_amplitude_45 = demodulate_image(image_1)
    phase_90, contrast_90, dc_amplitude_90 = demodulate_image(image_2)

    #Calculate polarisation angle
    polarisation_angle = phase_45 - phase_90

    polarisation = phase_mod(polarisation_angle, 2*np.pi)/4.

    return polarisation
