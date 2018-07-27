import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from Model.demodulate_TSH_synthetic_image import demodulate_nfw_images, demodulate_image, phase_mod, msesim_profiles
from Model.graph_format import plot_format
from Model.Constants import Constants

constants = Constants()
gamma, R = msesim_profiles(npx=32)
plot_format()

def load_image(filename):
    image_file = pd.HDFStore(filename)
    image = image_file['/a']
    return image

def digitize_image(image):
    #Input range on ADC = +/- 5V, 12 Bit ADC. Photron sa-4 cam has conversion gain of 11.6e/count. So each bin is 11.6 electrons.
    #Max count 4095 = 4095 * 11.6 = 47502
    photons = np.arange(0,47513.6,11.6)
    image = np.array(image)
    digitized_image = np.digitize(image, bins=photons)
    return digitized_image

def sample_distribution(image):
    lam = np.var(image)
    return np.random.poisson(lam=lam, size=(np.shape(image)))

def add_noise(image):
    shot_noise = sample_distribution(image)
    noisy_image = image + shot_noise
    return noisy_image

def demod_noisy_image(image_1, image_2):

    poisson_gamma = []

    for i in range(1000):

        noisy_image1 = add_noise(image_1)
        noisy_image2 = add_noise(image_2)

        phase_45, contrast_45, dc_amplitude_45 = demodulate_image(noisy_image1)
        phase_90, contrast_90, dc_amplitude_90 = demodulate_image(noisy_image2)

        polarisation_angle = phase_45 - phase_90

        poisson_gamma.append(-1 * phase_mod(polarisation_angle, 2 * np.pi) / 4.)

    poisson_gamma = np.array([poisson_gamma])[0,:,:,:]

    mean = np.average(poisson_gamma, axis=0)
    std = np.std(poisson_gamma, axis=0)

    return mean, std, poisson_gamma

filename1 = '/home/sgibson/PycharmProjects/IMSE/Model/synthetic_image1_32x32.hdf'
filename2 = '/home/sgibson/PycharmProjects/IMSE/Model/synthetic_image2_32x32.hdf'
image1 = load_image(filename1)
image2 = load_image(filename2)

digitized_image1 = digitize_image(image1*0.6*0.5*0.5*0.5*10**-3)
digitized_image2 = digitize_image(image2*0.6*0.5*0.5*0.5*10**-3)

mean, std, poisson_gamma = demod_noisy_image(digitized_image1, digitized_image2)

# plt.figure(1)
# plt.subplot(211)
# plt.imshow(digitized_image1, cmap='gray')
# plt.gca().invert_xaxis()
# plt.colorbar()
#
# plt.subplot(212)
# plt.imshow(digitized_image2, cmap='gray')
# plt.gca().invert_xaxis()
# plt.colorbar()
# plt.show()

plt.figure(2)
plt.plot(R, gamma[512,:]*(180./np.pi), '--', color='red', label='msesim')
plt.plot(R[:-1], mean[512,:]*(180./np.pi), alpha=0.7)
plt.errorbar(R[:-1], mean[512,:]*(180./np.pi), yerr=std[512,:]*(180./np.pi))
plt.xlabel('Major Radius (m)')
plt.ylabel('Mean $\mu$ polarisation angle (degrees)')
plt.show()

# plt.figure(3)
# plt.plot(R[:-1], std[512,:]*(180./np.pi))
# plt.xlabel('Major Radius (m)')
# plt.ylabel('Standard deviation $\sigma$')
# plt.show()