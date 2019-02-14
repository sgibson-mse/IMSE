import numpy as np
import matplotlib.pyplot as plt

from IMSE.Model.Material import Material
from IMSE.Model.Constants import Constants

constants = Constants()

class Crystal(Material):

    def __init__(self, thickness, cut_angle, material_name, orientation, wavelength, alpha, delta):

        self.name = material_name
        self.orientation = orientation * (np.pi/180.) #orientation of the optical axis (ie. vertical = 90 degrees)
        self.cut_angle = cut_angle * (np.pi/180.) #optical axis in radians
        self.L = thickness
        self.angular_frequency = 2 * np.pi * (constants.c / wavelength)

        #Inherent material properties based on the crystal material we want
        self.material_name = material_name

        super().__init__(material_name)

        self.ne = self.refractive_index(self.sc_ne, self.material_name, wavelength)
        self.no = self.refractive_index(self.sc_no, self.material_name, wavelength)

        self.birefringence = self.ne - self.no

        self.phi = self.phase_total(alpha, delta, wavelength)


    def refractive_index(self, sellmeier_coefficients, name, wavelength):

        wl_um = wavelength*10**6

        if name == 'lithium_niobate':
            return np.sqrt(1 + (sellmeier_coefficients[0]*wl_um**2)/(wl_um**2 - sellmeier_coefficients[1]) + (sellmeier_coefficients[2]*wl_um**2)/(wl_um**2 - sellmeier_coefficients[3]) + (sellmeier_coefficients[4]*wl_um**2) / (wl_um**2 - sellmeier_coefficients[5]))

        if name == 'alpha_bbo':
            return np.sqrt(sellmeier_coefficients[0] + (sellmeier_coefficients[1] / (wl_um** 2 - sellmeier_coefficients[2])) - (sellmeier_coefficients[3] * wl_um ** 2))
        else:
            print('Crystal not yet implemented!')

    def frequency_dispersion(self):

        #Calculate derivatives using central differences method

        d_omega = np.gradient(self.angular_frequency)
        d_birefringence = np.gradient(self.birefringence)

        dB_dw = d_birefringence/d_omega

        self.kappa = 1 + (self.angular_frequency/self.birefringence)*(dB_dw)

        return self.kappa

    #total phase shift between ordinary and extraordinary rays due to birefringent material

    def phase_total(self, alpha, delta, wavelength):
        self.phi = ((2*np.pi*self.L)/(wavelength)) * ( (self.no**2 - constants.n_air**2*np.sin(alpha)**2)**0.5 + (constants.n_air*(self.no**2 - self.ne**2)*np.sin(self.cut_angle)*np.cos(self.cut_angle)*np.cos(delta)*np.sin(alpha))/(self.ne**2*np.sin(self.cut_angle)**2 + self.no**2*np.cos(self.cut_angle)**2) + ((-1*self.no * (self.ne**2 * (self.ne**2 * np.sin(self.cut_angle)**2 + (self.no**2 * np.cos(self.cut_angle)**2)) - (self.ne**2 - (self.ne**2 - self.no**2)*np.cos(self.cut_angle)**2 * np.sin(delta)**2)*constants.n_air**2*np.sin(alpha)**2)**0.5)/(self.ne**2*np.sin(self.cut_angle)**2 + self.no**2*np.cos(self.cut_angle)**2)))
        return self.phi

    #individual terms in equation when taken to specific limits

    def phase_offset(self, wavelength):
        self.phi_0 = (-1 * 2 * np.pi * self.L * self.birefringence * np.cos(self.cut_angle) ** 2) / (wavelength)
        return self.phi_0

    def phase_shear(self, wavelength, focal_length):
        self.phi_shear = (2 * np.pi * self.L * self.birefringence * np.sin(2 * self.cut_angle)) / (2 * wavelength * ((self.ne + self.no) / 2) * focal_length) #*self.yy
        return self.phi_shear

    def phase_hyperbolic(self, wavelength, focal_length):
        self.phi_hyperbolic = (-2 * np.pi * self.L * self.birefringence / (4 * ((self.ne + self.no)/2)**2 * wavelength) * (focal_length) ** 2)
        return self.phi_hyperbolic

