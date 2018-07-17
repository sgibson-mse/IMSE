import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp2d, interp1d
from Model.graph_format import plot_format

plot_format()

from Model.load_msesim_output import load_msesim_spectrum

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

    return shift_image

def box_filter(image, x_size, y_size, centre):

    mask = np.zeros((np.shape(image)))

    mask[centre[1] - int(((y_size/2)-0.5)): centre[1] + int((y_size/2)+0.5), centre[0]- int((x_size/2)-0.5):centre[0] + int((x_size/2)+0.5)] = 1.

    return mask

def filter_image(mask, shift_image):

    filtered_image = shift_image*mask

    ifft_image = np.fft.ifft2(filtered_image)
    phase = np.arctan2(ifft_image.imag,ifft_image.real)
    amplitude = abs(ifft_image) # I0*contrast/2

    return phase, amplitude

def phase_mod(x,y):
    return x - np.floor((x+y/2.)/y)*y

def demodulate_image(image):

    window_image = apply_hanning(image)

    shift_image = get_carrier_frequency(window_image)

    mask = box_filter(image, x_size=int(len(image)), y_size=80, centre=[int(len(image)/2),430]) #box_filter(shift_image, x_size = int(len(image)), y_size=35, centre=[int(len(image)/2),23])
    dc_mask = box_filter(image, x_size=int(len(image)), y_size=80, centre=[int(len(image)/2), int(len(image)/2)])

    phase, amplitude = filter_image(mask, shift_image)

    dc_phase, dc_amplitude = filter_image(dc_mask, shift_image)

    contrast = (2*amplitude)/(dc_amplitude)

    return phase, contrast, dc_amplitude

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

def msesim_profiles():
    # Load input from msesim
    msesim = load_msesim_spectrum()
    stokes = msesim["total_stokes"]

    msesim_S2 = np.sum(stokes[:, 2, :], axis=1)
    msesim_S1 = np.sum(stokes[:, 1, :], axis=1)

    msesim_gamma = 0.5 * np.arctan(msesim_S2 / msesim_S1)
    msesim_gamma = msesim_gamma.reshape(21, 21)

    nx = 1024
    ny= 1024
    pixel_size = 20*10**-6

    x = np.arange(+0.5-int(nx/2), int(nx/2), 1)*pixel_size
    y = np.arange(+0.5-int(ny/2), int(ny/2), 1)*pixel_size

    x_small = x[::50]
    y_small = y[::50]

    gamma_interp = interp2d(x_small, y_small, msesim_gamma, kind='cubic')

    gamma = gamma_interp(x,y)

    #find positions in major radius
    xyz0 = msesim["central_coordinates"]
    c_xyz = msesim["collection_lens_coordinates"]

    xyz_grid = xyz0.reshape(21,21,3)

    R_vector = msesim["resolution_vector(R)"]
    R_grid = R_vector.reshape(21,21,7)
    R_vals = R_grid[:,:,2]

    R_interp = interp2d(x_small, y_small, R_vals)
    R = R_interp(x,y)

    return gamma, R

# Get the synthetic images

def demodulate_fw_images():

    image_1_fw = load_image(filename='image_FLC1_field_widened.hdf')
    image_2_fw = load_image(filename='image_FLC2_field_widened.hdf')

    #Demodulate images

    phase_45, contrast_45, dc_amplitude_45 = demodulate_image(image_1_fw)
    phase_90, contrast_90, dc_amplitude_90 = demodulate_image(image_2_fw)

    #Calculate polarisation angle
    polarisation_angle = phase_45 - phase_90

    polarisation_fw = phase_mod(polarisation_angle, 2*np.pi)/4.

    return image_1_fw, image_2_fw, polarisation_fw

def demodulate_nfw_images():

    image_1 = load_image(filename='image_FLC1_quintic.hdf')
    image_2 = load_image(filename='image_FLC2_quintic.hdf')

    #Demodulate images

    phase_45, contrast_45,dc_amplitude_45 = demodulate_image(image_1)
    phase_90, contrast_90, dc_amplitude_90 = demodulate_image(image_2)

    #Calculate polarisation angle
    polarisation_angle = phase_45 - phase_90

    polarisation = phase_mod(polarisation_angle, 2*np.pi)/4.

    return image_1, image_2, polarisation

# gamma, R = msesim_profiles()
# image_1_fw, image_2_fw, polarisation_fw = demodulate_fw_images()
# image_1, image_2, polarisation = demodulate_nfw_images()
#
#
# plt.figure()
# plt.title('No field widening')
# plt.subplot(211)
# plt.imshow(polarisation*(180./np.pi))
# plt.gca().invert_yaxis()
# plt.gca().invert_xaxis()
# plt.colorbar()
#
# plt.subplot(212)
# plt.imshow(polarisation_fw*(180./np.pi))
# plt.gca().invert_yaxis()
# plt.gca().invert_xaxis()
# plt.colorbar()
# plt.show()
#
# plt.figure()
# plt.plot(R[int(len(image_1)/2),:], -1*polarisation[int(len(image_1)/2),:]*(180./np.pi), label='demodulated, no field widening')
# plt.plot(R[int(len(image_1_fw)/2),:], -1*polarisation_fw[int(len(image_1_fw)/2),:]*(180./np.pi), label='demodulated, field widening')
# plt.plot(R[int(len(image_1)/2),:], gamma[int(len(image_1)/2),:]*(180./np.pi), '--', label='msesim output')
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(image_2.iloc[500,:].values, color='black', label='No field widening')
# plt.plot(image_1.iloc[500,:].values, color='red', label='Field widening')
# plt.xlabel('X pixel')
# plt.ylabel('Intensity [ph/s]')
# plt.legend(loc=1, prop={'size': 20})
# plt.show()

#
# plt.figure()
# plt.imshow(contrast_45)
# plt.gca().invert_xaxis()
# plt.colorbar()
# plt.clim(0.1,0.5)
# plt.show()
