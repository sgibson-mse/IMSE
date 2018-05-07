import numpy as np
import matplotlib.pyplot as plt

from Model.Physics_Constants import Constants
constants = Constants()

class AlphaBBO(object):

    def __init__(self, wavelength, thickness, cut_angle):

        self.wavelength = wavelength * 10**6 #convert to micrometers
        self.cut_angle = cut_angle * (np.pi/180.)
        self.alpha = 4.7 * (180./np.pi) #non axial ray incidence angle
        self.L = thickness #um crystal thickness
        self.focal_length = 85*10**3 #micrometers
        self.angular_frequency = 2 * np.pi * (constants.c / self.wavelength)
        self.sc_ne = [2.3753, 0.012240, -0.016670, -0.01516]
        self.sc_no = [2.7359, 0.018780, -0.018220, -0.01354]
        self.beta = np.arange(0,2*np.pi)

        self.ne = self.refractive_index(sellmeier_coefficients=self.sc_ne)
        self.no = self.refractive_index(sellmeier_coefficients=self.sc_no)

        self.birefringence = self.ne - self.no

        self.kappa = self.frequency_dispersion()

        self.phi_0 = self.phase_offset()

        self.phi_shear = self.phase_shear()

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

    def phase_offset(self):
        self.phi_0 = (-1 * 2* np.pi * self.L * self.birefringence * np.cos(self.cut_angle)**2) / (self.wavelength)
        return self.phi_0

    def phase_shear(self):
        self.phi_shear = (2 * np.pi * self.L * self.birefringence * np.sin(2*self.cut_angle))/(2*self.wavelength* ((self.ne + self.no)/2) * self.focal_length)
        return self.phi_shear

    def find_pixel_positions(self):
        self.x = -1 * self.alpha * self.focal_length * np.cos(self.beta)


