import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Tools.Plotting.graph_format import plot_format
from Model.Observer import Camera
from Model.Constants import Constants
from Tools.demodulate_TSH_synthetic_image import demodulate_image, phase_mod, msesim_profiles

camera=Camera(photron=True)

cb = plot_format()

def load_image(filename):
    image_file = pd.HDFStore(filename)
    image = image_file['/a']
    return image

def image_shot_noise(image):
    # Set a seed so that we can reproducibly generate the same random numbers each time and initialize a RandomState instance with it.
    #  Using this RandomState instance, call the poisson method with a mean defined by the number of photons in the image
    shot_noise = np.random.poisson(lam=image, size=np.shape(image))

    return shot_noise

def calc_electrons_in(image):
    # Calculate number of photoelectrons
    return np.round(camera.quantum_efficiency * image)

def dark_noise(image):
    # Simulate read noise of camera by sampling from a normal distribution with the standard deviation equal to the dark noise of the camera.

    #expecting a poisson distribution around the dark noise - dark noise also has shot noise.
    electrons_dark_noise = np.round(np.random.normal(loc=camera.dark_noise, scale=np.sqrt(camera.dark_noise), size=image.shape))

    return electrons_dark_noise

def digitize_image(image, exposure_time):
    #Input range on ADC = +/- 5V, 12 Bit ADC. Photron sa-4 cam has conversion gain sensitivity of 11.6e/count. So each bin is 11.6 electrons.
    #Max count 4095 = 4095 * 11.6 = 47502

    image = image*exposure_time #1ms exposure time

    electrons_dark_noise = dark_noise(image)

    #image shot noise
    photons_shot_noise = image_shot_noise(image)
    electrons_shot_noise = calc_electrons_in(photons_shot_noise)

    final_image = electrons_shot_noise + electrons_dark_noise

    # Convert to ADU
    sensitivity = 1/11.6
    max_adu     = np.int(2**12 - 1)
    adu         = (final_image * sensitivity).astype(int) # Convert to discrete numbers
    adu[(adu > max_adu)] = max_adu # models pixel saturation

    digitized_image = adu

    # plt.imshow(digitized_image)
    # plt.colorbar()
    # plt.show()

    return digitized_image

def digitize_reference_image(image):

    # Convert to ADU
    sensitivity = 1/11.6
    max_adu     = np.int(2**12 - 1)
    adu         = (image * sensitivity).astype(int) # Convert to discrete numbers
    #adu[adu > max_adu] = 12 # models pixel saturation

    digitized_image = adu

    return digitized_image

def demodulate_images(digitized_image1, digitized_image2):

    phase_45, contrast_45, dc_amplitude_45 = demodulate_image(digitized_image1)
    phase_90, contrast_90, dc_amplitude_90 = demodulate_image(digitized_image2)
    polarisation_angle = phase_45 - phase_90
    poisson_gamma = -1 * phase_mod(polarisation_angle, 2 * np.pi) / 4.

    return poisson_gamma

# plt.figure(1)
# plt.subplot(221)
# plt.imshow(image)
# plt.colorbar()

# plt.figure()
# plt.imshow(noise_image)
# plt.xlabel('X pixel')
# plt.ylabel('Y pixel')
# plt.colorbar()
# plt.show()
#
# plt.figure(2)
# plt.hist(image.ravel(), bins=np.arange(np.min(image), np.max(image),1000))
# plt.xlabel('Number of photons per pixel')
# plt.ylabel('Frequency')
