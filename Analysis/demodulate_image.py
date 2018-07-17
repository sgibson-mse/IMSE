#External Imports
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

#Internal imports
from Analysis.read_binary import load_binary
from Analysis.peak_find import indexes

def prepare_image(filename, frames):

    images = load_binary(filename, FLC=True)
    image = images[:, :, frames].astype(np.float64)
    image = image / np.max(image)

    return image

def take_slice(image, pixel):

    image_slice = image[int(pixel),:]# slice image at the center pixel

    return image_slice

#make the window

def make_window(image_slice, ny):

    fft_column = np.fft.fftshift(np.fft.fft(image_slice))

    # find the peaks of the carrier frequency at /pm f and DC
    peaks = indexes(abs(fft_column), thres=0.2)

    fringe_peaks = indexes(abs(image_slice), thres=0.7, min_dist=10)

    try:
        fringe_width = int((fringe_peaks[1] - fringe_peaks[0]) / 2)
    except IndexError:
        fringe_width = 4

    window = np.zeros((ny))

    #Design filter around the positive or negative carrier frequency

    #window[peaks[0] - fringe_width: peaks[0] + fringe_width + 1] = signal.hann(fringe_width*2+1)
    window[peaks[-1] - fringe_width: peaks[-1] + fringe_width + 1 ] = signal.hann(fringe_width*2+1)

    return window, fringe_width, peaks, fft_column

def filter_image(window, fft_column):

    filtered_column = window * fft_column

    carrier_frequency = np.where(np.max(abs(filtered_column)))

    ifft_column = np.fft.ifft(filtered_column)

    intensity = 2 * abs(ifft_column) #amplitude of the carrier frequency

    #phase = np.arctan2(ifft_column.imag, ifft_column.real)
    phase = np.angle(ifft_column)

    return intensity, phase

# from Pycis - import the module when I can clone the git.

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

#Extract the images from the binary file
filename = str(os.getcwd()) + '/sam_8.dat'

nx = 1280
ny = 1080
n_frames = 38
n_states = 2
frames = np.arange(0,38,1)
pixels = np.arange(0,ny,1)

Intensity = np.zeros((nx, ny, n_frames))
Phase = np.zeros((nx, ny, n_frames))
filtered_image = np.zeros((nx, ny, n_frames))
Theta = np.zeros((nx,ny,n_frames))

for n in frames:
    image = prepare_image(filename, frames[n])

    for i in range(len(pixels)):
        image_slice = take_slice(image, pixels[i])
        window, fringe_width, peaks, fft_column = make_window(image_slice, ny)
        intensity_slice, phase_slice = filter_image(window, fft_column)
        Intensity[i,:,n] = intensity_slice
        Phase[i,:,n] = phase_slice

phase_unwrapped = np.zeros((nx,ny,n_frames))
phase_differences = []

for n in range(n_frames):
    phase_unwrapped[:,:,n] = unwrap(Phase[:,:,n])

for n in range(n_frames-1):
    phase_difference = phase_unwrapped[:,:,n+1] - phase_unwrapped[:,:,n]
    phase_differences.append(phase_difference)

phase_diff = np.asarray(phase_differences)
phase_diff = phase_diff[::2,:,:]

center_y = np.round((ny - 1) / 2).astype(np.int)
center_x = np.round((nx - 1) / 2).astype(np.int)

central_offsets = phase_diff[:,center_y, center_x]*(180./np.pi)

rotary_stage_angles = np.arange(0,190,10)

# plt.figure()
# plt.plot(rotary_stage_angles, central_offsets/4.)
# plt.show()
#
# plt.figure()
# plt.imshow(phase_diff[0,:,:]*(180/np.pi))
# plt.colorbar()
# plt.clim(-12,+12)
# plt.show()

def plot_image(image):

    plt.figure()
    plt.title('Ne $\lambda$ = 600nm, Calibration Image $\phi_{45^{\circ}}$')
    plt.imshow(image.T, cmap='gray')
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

def plot_output(half_fftcolumn, window, phase, image_slice, new_pixels, intensity):

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



#find the phase at the center of the column
#find difference between first pixel and the center column
#find it as an integer of pi [-pi, 0, pi]
#then add or subtract this from the value
#repeat for each row

# #Unwrap the phase
#
# phase_array = Phase[:,:,0]
#
# unwrapped_phase_array = np.zeros((nx,ny))
#
# for row in range(nx):
#
#     phase_slice = phase_array[row,:]
#
#     center_x = int((len(phase_array[row,:])-1)/2)
#
#     center_phase = phase_slice[center_x]
#
#     #center_y = int((len(phase_array[:,0])-1)/2)
#
#     for i in range(len(phase_slice)):
#         #find phase difference as integer of pi
#         phase_diff = int((phase_slice[i] - center_phase) / np.pi)
#
#         if phase_diff == 0:
#             phase_slice[i] = phase_slice[i]
#         if phase_diff == 1:
#             phase_slice[i] += np.pi
#         if phase_diff == -1:
#             phase_slice[i] -= np.pi
#
#         unwrapped_phase_array[row,i] = phase_slice[i]
#
# plt.figure()
# plt.imshow(unwrapped_phase_array)
# plt.show()

