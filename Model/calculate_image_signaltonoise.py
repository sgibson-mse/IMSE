import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Model.demodulate_TSH_synthetic_image import demodulate_image, phase_mod, msesim_profiles
from Model.graph_format import plot_format
from Model.Constants import Constants

constants = Constants()
plot_format()
gamma, R = msesim_profiles(npx=32)

def load_image(filename):
    image_file = pd.HDFStore(filename)
    image = image_file['/a']
    return image

def sample_distribution(image):
    lam = np.var(image)
    return np.random.poisson(lam=lam, size=(np.shape(image)))

def normal_distribution(image):
    data = np.loadtxt('/home/sam/Desktop/msesim/beam/beamripple_txt.txt')
    voltage = data[:,0]
    probability = data[:,1]
    mu, sigma = np.average(voltage), np.std(voltage)  # mean and standard deviation
    beam_modulation = np.random.normal(mu, sigma, size=(np.shape(image)))
    velocity_ripple = np.sqrt(2*beam_modulation/constants.mass_n)
    lambda_ripple = velocity_ripple/constants.c
    print(lambda_ripple)
    return beam_modulation

def add_noise(image):
    shot_noise = sample_distribution(image.values)
    noisy_image = image.values + shot_noise
    return noisy_image

def calculate_SN(I0, contrast):
    #standard deviation in 1ms, 14 pixels per fringe
    sigma = 2/(contrast*np.sqrt(I0*14))
    return sigma

def SN_nopoisson(image_FLC1, image_FLC2):

    phase_45, contrast_45, dc_amplitude_45 = demodulate_image(image_FLC1)

    phase_90, contrast_90, dc_amplitude_90 = demodulate_image(image_FLC2)

    sigma_flc1 = calculate_SN(dc_amplitude_45, contrast_45)

    sigma_flc2 = calculate_SN(dc_amplitude_90, contrast_90)

    sigma_total = np.sqrt(sigma_flc1**2 + sigma_flc2**2)/4.

    #Calculate polarisation angle
    polarisation_angle = phase_45 - phase_90

    polarisation_mod = -1*phase_mod(polarisation_angle, 2*np.pi)/4.

    return polarisation_mod, sigma_total

def poisson_noise(image_FLC1, image_FLC2):

    poisson_gamma = []

    for i in range(100):

        noisy_image1 = add_noise(image_FLC1)
        noisy_image2 = add_noise(image_FLC2)

        phase_45, contrast_45, dc_amplitude_45 = demodulate_image(noisy_image1)
        phase_90, contrast_90, dc_amplitude_90 = demodulate_image(noisy_image2)

        polarisation_angle = phase_45 - phase_90

        poisson_gamma.append(-1 * phase_mod(polarisation_angle, 2 * np.pi) / 4.)

    poisson_gamma = np.array([poisson_gamma])[0,:,:,:]

    mean = np.average(poisson_gamma, axis=0)
    std = np.std(poisson_gamma, axis=0)

    return mean, std, poisson_gamma

# image_FLC1 = load_image(filename='/home/sgibson/PycharmProjects/IMSE/Model/synthetic_image1_32x32.hdf')
# image_FLC2 = load_image(filename='/home/sgibson/PycharmProjects/IMSE/Model/synthetic_image2_32x32.hdf')
# mean, std, poisson_gamma = poisson_noise(image_FLC1, image_FLC2)
#
# plt.figure()
# plt.plot(R, gamma[512,:]*(180./np.pi), '--', color='red', label='msesim')
# plt.plot(R[:-1], mean[512,:]*(180./np.pi), alpha=0.7)
# plt.legend()
# plt.show()

