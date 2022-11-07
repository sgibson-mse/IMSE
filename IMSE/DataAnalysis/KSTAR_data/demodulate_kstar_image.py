import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters

def fft_2D(image):
    return np.fft.fft2(image, axes=(0,1))

def get_carrier_frequency(image):

    image_fft = fft_2D(image)
    shift_image = np.fft.fftshift(image_fft) #shift so DC frequency is in the center.

    return shift_image

def box_filter(image, x_size, y_size, centre):

    mask = np.zeros((np.shape(image)))

    mask[centre[1] - int((y_size/2)-0.5): centre[1] + int((y_size/2)+0.5), centre[0] - int((x_size/2)-0.5):centre[0] + int((x_size/2)+0.5)] = 1.

    return mask

def find_maxima(image, neighborhood_size, threshold):
    data_max = filters.maximum_filter(image, neighborhood_size)
    maxima = (image == data_max)
    data_min = filters.minimum_filter(image, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    x, y = [], []
    for dy, dx in slices:
        x_center = (dx.start + dx.stop - 1) / 2
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1) / 2
        y.append(y_center)
    return x,y

def filter_image(mask, shift_image):

    filtered_image = shift_image*mask

    ifft_image = np.fft.ifft2(filtered_image)

    phase = np.arctan2(ifft_image.imag,ifft_image.real)
    amplitude = abs(ifft_image)

    return phase, amplitude

def phase_mod(x,y):
    return x - np.floor((x+y/2.)/y)*y

def unwrap(phase_array):

    y_pix, x_pix = np.shape(phase_array)
    phase_contour = -np.unwrap(phase_array[int(np.round(y_pix / 2)), :])

    # sequentially unwrap image columns:
    phase_uw_col = np.zeros_like(phase_array)

    for i in range(0, x_pix):
        phase_uw_col[:, i] = np.unwrap(phase_array[:, i])

    phase_contour = phase_contour + phase_uw_col[int(np.round(y_pix / 2)), :]
    phase_0 = np.tile(phase_contour, [y_pix, 1])
    phase_uw = phase_uw_col - phase_0

    # wrap image centre into [-pi, +pi] (assumed projection of optical axis onto detector)
    y_centre_idx = np.round((np.size(phase_uw, 0) - 1) / 2).astype(np.int)
    x_centre_idx = np.round((np.size(phase_uw, 1) - 1) / 2).astype(np.int)
    phase_uw_centre = phase_uw[y_centre_idx, x_centre_idx]

    if phase_uw_centre > 0:
        while abs(phase_uw_centre) > np.pi:
            phase_uw -= 2*np.pi
            phase_uw_centre = phase_uw[y_centre_idx, x_centre_idx]
    else:
        while abs(phase_uw_centre) > np.pi:
            phase_uw += 2*np.pi
            phase_uw_centre = phase_uw[y_centre_idx, x_centre_idx]

    return phase_uw

def demodulate_image(image):

    shift_image = get_carrier_frequency(image)

    peak_x, peak_y = find_maxima(abs(shift_image), neighborhood_size=50, threshold = 1*10**7)

    mask = box_filter(image, x_size=int(len(image[-1])), y_size=50, centre=[int(peak_x[1]), int(peak_y[0])])
    dc_mask = box_filter(image, x_size=int(len(image[-1])), y_size=50, centre=[int(peak_x[1]), int(peak_y[1])])

    phase, amplitude = filter_image(mask, shift_image)

    phase = unwrap(phase)

    dc_phase, dc_amplitude = filter_image(dc_mask, shift_image)

    dc_phase = unwrap(phase)

    contrast = (2*amplitude)/(dc_amplitude)

    return phase, contrast