import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_image(filename):
    image_file = pd.HDFStore(filename)
    image = image_file['/a']
    return image

def hanning_window(image):
    h = np.hanning(len(image))
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

    return shift_image

def box_filter(image, x_size, y_size, centre):

    mask = np.zeros((np.shape(image)))

    mask[centre[1] - int(((y_size/2)-0.5)): centre[1] + int((y_size/2)+0.5), centre[0]- int((x_size/2)-0.5):centre[0] + int((x_size/2)+0.5)] = 1.

    return mask

def filter_image(mask, shift_image):

    filtered_image = shift_image*mask

    plt.figure()
    plt.title('filtered_image')
    plt.imshow(np.log10(abs(filtered_image)))
    plt.colorbar()
    plt.clim(8,12)
    plt.show()

    ifft_image = np.fft.ifft2(filtered_image)
    phase = np.arctan2(ifft_image.imag,ifft_image.real)

    return phase

def phase_mod(x,y):
    return x - np.floor((x+y/2.)/y)*y

def demodulate_image(image):

    window_image = apply_hanning(image)

    shift_image = get_carrier_frequency(window_image)

    mask = box_filter(image, x_size=int(len(image)), y_size=70, centre=[int(len(image)/2),430]) #box_filter(shift_image, x_size = int(len(image)), y_size=35, centre=[int(len(image)/2),23])

    phase = filter_image(mask, shift_image)

    return phase

image_1 = load_image(filename='image_FLC1.hdf')
image_2 = load_image(filename='image_FLC2.hdf')

phase_45 = demodulate_image(image_1)
phase_90 = demodulate_image(image_2)

polarisation_angle = phase_45 - phase_90

polarisation_mod = phase_mod(polarisation_angle, 2*np.pi)/4.

plt.figure()
plt.imshow(polarisation_mod*(180./np.pi))
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.colorbar()
plt.show()

plt.figure()
plt.plot(polarisation_mod[int(len(image_1)/2),:]*(180./np.pi))
plt.show()