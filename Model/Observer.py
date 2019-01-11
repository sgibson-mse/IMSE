import numpy as np
import matplotlib.pyplot as plt
from Tools.Plotting.graph_format import plot_format

plot_format()

"""
Camera specifications for the Photron SA-4 and the PCO Edge cameras, two candidates for the imaging system. 
Calculate the total signal to noise ratio including the read, dark and shot noise as measured for the two cameras.
"""

class Camera(object):

    def __init__(self, name):

        self.name = name

        self.px = None
        self.py = None
        self.pixel_size = None

        self.sensor_size_x = None
        self.sensor_size_y = None

        self.dark_noise = None
        self.dark_noise_error = None

        self.read_noise = None
        self.fill_factor = None

        self.quantum_efficiency = None
        self.fullwell_capacity = None
        self.gain = None
        self.integration_time = None

        self.n_photons = None

        if self.name == 'photron-sa4':
            self.photron_specs(self.name)
        elif self.name == 'pco-edge':
            self.pco_specs(self.name)
        else:
            print('Camera properties are not currently implemented, choose photron-sa4 or pco-edge!')

    def observe(self, emission, exposure, ideal):

        """
        Observe emission using the camera, for a given exposure time. Sample noise using the camera specs if ideal = False.
        :param emission: Incoming emission in photons/s
        :param exposure: Exposure time in s
        :param ideal: (Bool) - True - Sample a noise distribution characterized by camera specs and apply to the image.
        :return: An image output from the camera.
        """

        total_intensity = np.sum(emission, axis=2)
        n_photons = total_intensity * exposure

        if ideal == True:
            print('Ideal image - no shot noise added!')
            image = n_photons
            return image

        elif ideal== False:
            print('Generating noisy image!')
            image = self.sample_noise(n_photons)
            return image

    def sample_noise(self, n_photons):

        collected_photons = n_photons*self.fill_factor
        dark_electrons = self.calculate_dark_noise(collected_photons)
        shot_photons = self.calculate_shot_noise(collected_photons)
        shot_electrons = self.count_electrons(shot_photons)

        return shot_electrons + dark_electrons

    def calculate_shot_noise(self, n_photons):

        """
        Sample a poission distribution to simulate photon counting statistics for the given camera.

        :param n_photons: Number of incident photons on the camera sensor
        :return: Number of photons due to shot noise
        """

        return np.random.poisson(lam=n_photons, size=n_photons.shape)

    def calculate_dark_noise(self, n_photons):

        """
        Simulate dark of camera by sampling from a normal distribution with the standard deviation equal to the dark noise of the camera.

        :param n_photons: Number of incident photons on the camera sensor
        """

        return np.round(np.random.normal(loc=self.dark_noise, scale=np.sqrt(self.dark_noise), size=n_photons.shape))

    def count_electrons(self, n_photons):

        """
        Find number of photons actually converted to electrons given the quantum efficiency of the camera.
        :param n_photons:
        :return:
        """

        return np.round(self.quantum_efficiency * n_photons)

    def digitize(self, n_photons):

        """
        Digitize the image for a given camera sensor, and simulate pixel saturation.

        :param n_photons: Number of photons on the sensor
        :return: Digitized image (Array - px, py)
        """

        adu = (n_photons * self.sensitivity).astype(int)  # Convert to discrete numbers
        adu[(adu > self.ADU_max)] = self.ADU_max  # models pixel saturation

        digitized_image = adu

        return digitized_image

    def photron_specs(self, name):

        self.name = name
        self.px = 1024
        self.py = 1024
        self.pixel_size = 20*10**-6 #m

        self.sensor_size_x = 20.48*10**-3 #m
        self.sensor_size_y = 20.48*10**-3 #m

        self.dark_noise = 3.55 #electrons
        self.dark_noise_error = 0.02

        self.read_noise = 41.2
        self.fill_factor = 0.52

        self.sensitivity = 1 / 11.6
        self.ADU_max = np.int(2 ** 12 - 1)

        self.fullwell_capacity = 45000
        self.quantum_efficiency = 0.4
        self.gain = 0.0862
        self.integration_time = np.arange(1*10**-3, 20.5*10**-3, 0.5*10**-3)

        self.x = np.linspace(-1 * self.sensor_size_x / 2, 1 * self.sensor_size_x / 2, self.px)
        self.y = np.linspace(-1 * self.sensor_size_y / 2, 1 * self.sensor_size_y / 2, self.py)

    def pco_specs(self, name):

        self.name = name
        self.px = 2560
        self.py = 2160

        self.sensor_size_x = 16.64*10**-3
        self.sensor_size_y = 14.04*10**-3

        self.dark_noise = 2 #electrons
        self.read_noise = 2.2 #electrons

        self.fill_factor = 1 # how much of the pixel is actually available for light collection
        self.fullwell_capacity = 30000 #electrons
        self.pixel_size = 6.5*10**-6

        self.gain = 0.46 #electrons/count
        self.quantum_efficiency = 0.54
        self.integration_time = np.arange(5*10**-3, 20.5*10**-3, 0.5*10**-3)

        self.n_photons = 4*10**7

        self.x = np.linspace(-1 * self.sensor_size_x / 2, 1 * self.sensor_size_x / 2, self.px)
        self.y = np.linspace(-1 * self.sensor_size_y / 2, 1 * self.sensor_size_y / 2, self.py)

    def plot(self, image):
        xx, yy = np.meshgrid(self.x, self.y)
        plt.figure()
        plt.pcolormesh(xx, yy, image)
        plt.colorbar()
        plt.show()

    # def calculate_intensity(self):
    #     self.intensity = self.n_photons * self.fill_factor * self.quantum_efficiency * self.integration_time * self.gain
    #     return self.intensity
    #
    # def calculate_shotnoise(self):
    #     self.shot_noise = self.n_photons * self.integration_time * self.quantum_efficiency * self.gain**2
    #     return self.shot_noise
    #
    # def calculate_darknoise(self):
    #     self.dark = self.dark_noise * self.integration_time
    #     return self.dark
    #
    # def calculate_SN(self):
    #     self.SN = self.intensity / np.sqrt(self.read_noise ** 2 + self.dark + self.shot_noise)
    #     return self.SN
    #
    # def plot_SN(self):
    #
    #     plt.figure()
    #     plt.plot(self.integration_time*1000, self.SN, '--', label=str(self.name))
    #     plt.xlim(0,10)
    #     plt.xlabel('Integration time [ms]')
    #     plt.ylabel('Signal to Noise')
    #     plt.legend()
    #     plt.show()

class AvalanchePhotoDiode:

    def __init__(self, name):
        self.name = name
        self.sensor_size = None
        self.quantum_efficiency = None
        #etc.
