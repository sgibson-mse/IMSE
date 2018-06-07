import numpy as np
import matplotlib.pyplot as plt

from Model.synthetic_images import calculate_synthetic_image

def center_points(image):
    shape = np.shape(image)
    x_center = int(shape[0]/2)
    y_center = int(shape[1]/2)
    return x_center, y_center

def fft_2D(image):
    return np.fft.fft2(image, axes=(0,1))

def get_carrier_frequency(image):

    image_fft = fft_2D(image)
    shift_image = np.fft.fftshift(image_fft) #shift so DC frequency is in the center.

    return shift_image

def box_filter(image, x_size, y_size, centre):

    mask = np.zeros((np.shape(image)))

    mask[centre[1]:centre[1] + y_size, centre[0]:centre[0] + x_size] = 1.

    return mask

def filter_image(mask, shift_image):

    filtered_image = shift_image*mask

    plt.figure()
    plt.imshow(abs(np.log10(shift_image)))
    plt.colorbar()
    plt.imshow(abs(filtered_image), alpha=0.3)
    plt.show()

    ifft_image = np.fft.ifft2(filtered_image)

    phase = np.arctan2(ifft_image.imag, ifft_image.real)

    plt.figure()
    plt.imshow(phase*(180./np.pi))
    plt.colorbar()
    plt.show()

    return phase

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
    mask = box_filter(image, x_size=110, y_size=40, centre=[460,450])

    phase = filter_image(mask, shift_image)
    unwrapped_phase = unwrap(phase)

    plt.figure()
    plt.imshow(unwrapped_phase*(180./np.pi))
    plt.show()

    return unwrapped_phase

image_1 = calculate_synthetic_image(FLC_state=1)
phase_45 = demodulate_image(image_1)
image_2 = calculate_synthetic_image(FLC_state=2)
phase_90 = demodulate_image(image_2)

polarisation_angle = phase_45 - phase_90

# plt.figure()
# plt.imshow(polarisation_angle*(180./np.pi))
# plt.gca().invert_yaxis()
# plt.colorbar()
# plt.show()