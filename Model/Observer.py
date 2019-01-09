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
        if self.name == 'pco-edge':
            self.pco_specs(self.name)
        # else:
        #     print('Properties for the camera you specified are not available! Please choose either photron-sa4 or pco-edge.')

    def observe(self, light):
        return np.sum(light, axis=2)

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

        self.fullwell_capacity = 45000
        self.quantum_efficiency = 0.4
        self.gain = 0.0862
        self.integration_time = np.arange(1*10**-3, 20.5*10**-3, 0.5*10**-3)

        self.n_photons = 4*10**7

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

    def calculate_intensity(self):
        self.intensity = self.n_photons * self.fill_factor * self.quantum_efficiency * self.integration_time * self.gain
        return self.intensity

    def calculate_shotnoise(self):
        self.shot_noise = self.n_photons * self.integration_time * self.quantum_efficiency * self.gain**2
        return self.shot_noise

    def calculate_darknoise(self):
        self.dark = self.dark_noise * self.integration_time
        return self.dark

    def calculate_SN(self):
        self.SN = self.intensity / np.sqrt(self.read_noise ** 2 + self.dark + self.shot_noise)
        return self.SN

    def plot_SN(self):

        plt.figure()
        plt.plot(self.integration_time*1000, self.SN, '--', label=str(self.name))
        plt.xlim(0,10)
        plt.xlabel('Integration time [ms]')
        plt.ylabel('Signal to Noise')
        plt.legend()
        plt.show()

