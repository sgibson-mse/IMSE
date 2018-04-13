#External Imports
import os
import numpy as np
import matplotlib.pyplot as plt

#Internal imports
from Analysis.read_binary import load_binary

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

def plot_1dpowerspectrum(image_fft):
    plt.figure()
    plt.plot(abs(image_fft)**2)
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

#Extract the images from the binary file
filename = str(os.getcwd()) + '/sam_8.dat'

nx = 1280
ny = 1080
n_frames = 38

images = load_binary(filename, FLC=True)

image = images[:,:,1].astype(np.float64)
image = image/np.max(image)

image_slice = image[nx/2,:] #slice image at the center pixel

image_fft = np.fft.fft(image_slice)

total_image_fft = np.fft.fft2(image)
total_image_fft = np.fft.fftshift(total_image_fft)






