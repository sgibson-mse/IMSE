import numpy as np

from Model.Physics_Constants import Constants

class AlphaBBO(object):

    def __init__(self, wavelength, constants):

        self.wavelength = wavelength
        self.angular_frequency = 2 * np.pi * (constants.c / self.wavelength)
        self.sc_ne = [2.3753, 0.012240, -0.016670, -0.01516]
        self.sc_no = [2.7359, 0.018780, -0.018220, -0.01354]

        self.ne = self.refractive_index(sellmeier_coefficients=self.sc_ne)
        self.no = self.refractive_index(sellmeier_coefficients=self.sc_no)

        self.birefringence = self.ne - self.no

        self.kappa = self.frequency_dispersion()

    def refractive_index(self, sellmeier_coefficients):

        return np.sqrt(sellmeier_coefficients[0] + (sellmeier_coefficients[1] / ((self.wavelength ** 2) +
                   sellmeier_coefficients[2])) + (sellmeier_coefficients[3] * (self.wavelength ** 2)))

    def frequency_dispersion(self):
        #Calculate derivatives using central differences method

        d_omega = np.gradient(self.angular_frequency)
        d_birefringence = np.gradient(self.birefringence)

        dB_dw = d_birefringence/d_omega

        self.kappa = 1 + (self.angular_frequency/self.birefringence)*(dB_dw)

        return self.kappa


wavelength = np.arange(659, 661, 0.01)
wavelength *= 10**-3 # wavelength must be in microns for AlphaBBO class

constants = Constants()
bbo = AlphaBBO(wavelength, constants)

print(bbo.kappa)
