import numpy as np
import matplotlib.pyplot as plt

from Model.Physics_Constants import Constants
constants = Constants()

class Crystal(object):

    def __init__(self, wavelength, thickness, cut_angle, name):

        self.wavelength = wavelength * 10**-3 #convert to micrometers
        self.cut_angle = cut_angle * (np.pi/180.)
        #self.alpha = alpha * (180./np.pi) #non axial ray incidence angle
        self.L = thickness #um crystal thickness
        self.focal_length = 85000 #micrometers
        self.angular_frequency = 2 * np.pi * (constants.c / self.wavelength)
        self.beta = np.arange(0,2*np.pi)
        self.name = name

        if self.name == 'lithium_niobate':
            self.sc_ne = [2.9804, 0.02047, 0.5981, 0.0666, 8.9543, 416.08]
            self.sc_no = [2.6734, 0.01764, 1.2290, 0.05914, 12.614, 474.60]

            self.ne = self.refractive_index(sellmeier_coefficients=self.sc_ne, name=self.name)
            self.no = self.refractive_index(sellmeier_coefficients=self.sc_no, name=self.name)

        if self.name == 'alpha_bbo':

            self.sc_ne = [2.3753, 0.012240, -0.016670, -0.01516]
            self.sc_no = [2.7359, 0.018780, -0.018220, -0.01354]

            self.ne = self.refractive_index(sellmeier_coefficients=self.sc_ne, name=self.name)
            self.no = self.refractive_index(sellmeier_coefficients=self.sc_no, name=self.name)

        self.birefringence = self.ne - self.no

        self.kappa = self.frequency_dispersion()

        self.phi_0 = self.phase_offset()

        self.phi_shear = self.phase_shear()

        self.phi_hyperbolic = self.phase_hyperbolic()

    def refractive_index(self, sellmeier_coefficients, name):

        if name == 'lithium_niobate':
            return np.sqrt(1 + (sellmeier_coefficients[0]*self.wavelength**2)/(self.wavelength**2 - sellmeier_coefficients[1]) + (sellmeier_coefficients[2]*self.wavelength**2)/(self.wavelength**2 - sellmeier_coefficients[3]) + (sellmeier_coefficients[4]*self.wavelength**2) / (self.wavelength**2 - sellmeier_coefficients[5]))

        if name == 'alpha_bbo':
            return np.sqrt(sellmeier_coefficients[0] + (sellmeier_coefficients[1] / ((self.wavelength ** 2) +
                                                                              sellmeier_coefficients[2])) + (
                                sellmeier_coefficients[3] * (self.wavelength ** 2)))
        else:
            print('Crystal not yet implemented!')

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
        self.phi_shear = (2 * np.pi * self.L*10**-6 * self.birefringence * np.sin(2*self.cut_angle))/(2*self.wavelength*10**-6 * ((self.ne + self.no)/2) * self.focal_length*10**-6)
        return self.phi_shear

    def phase_hyperbolic(self):
        self.phi_hyperbolic = -2*np.pi * self.L*10**-6 * self.birefringence / ( 4 * ((self.ne + self.no)/2)**2 * self.wavelength*10**-6 * (self.focal_length*10**-6)**2 )
        return self.phi_hyperbolic
