import numpy as np
import matplotlib.pyplot as plt

from Model.Constants import Constants
constants = Constants()

class Crystal(object):

    def __init__(self, wavelength, thickness, cut_angle, name, nx, ny, pixel_size, orientation, two_dimensional):

        self.name = name
        self.two_dimensional = two_dimensional
        self.nx = nx
        self.ny = ny
        self.pixel_size = pixel_size
        self.orientation = orientation * (np.pi/180.) #orientation of the optical axis (ie. vertical = 90 degrees)
        self.wavelength = wavelength * 10**6 #convert to micrometers
        self.cut_angle = cut_angle * (np.pi/180.) #optical axis in radians
        self.L = thickness*10**6 #um crystal thickness
        self.focal_length = 0.085 *10**6 #um micrometers
        self.angular_frequency = 2 * np.pi * (constants.c / self.wavelength) #in micrometers

        self.alpha, self.beta = self.non_axial_rays()
        self.delta = self.beta - self.orientation

        if self.name == 'lithium_niobate':
            self.sc_ne = [2.9804, 0.02047, 0.5981, 0.0666, 8.9543, 416.08]
            self.sc_no = [2.6734, 0.01764, 1.2290, 0.05914, 12.614, 474.60]

            self.ne = self.refractive_index(sellmeier_coefficients=self.sc_ne, name=self.name)
            self.no = self.refractive_index(sellmeier_coefficients=self.sc_no, name=self.name)

        if self.name == 'alpha_bbo':
            self.sc_ne = [2.3753, 0.012240, 0.016670, 0.01516]
            self.sc_no = [2.7359, 0.018780, 0.018220, 0.01354]

            self.ne = self.refractive_index(sellmeier_coefficients=self.sc_ne, name=self.name)
            self.no = self.refractive_index(sellmeier_coefficients=self.sc_no, name=self.name)

        self.birefringence = self.ne - self.no

        if self.two_dimensional == True:

            self.phi_total = self.phase_total()
        else:
            self.phi_0 = self.phase_offset()
            self.phi_shear = self.phase_shear()
            self.phi_hyperbolic = self.phase_hyperbolic()

    def refractive_index(self, sellmeier_coefficients, name):

        if name == 'lithium_niobate':
            return np.sqrt(1 + (sellmeier_coefficients[0]*self.wavelength**2)/(self.wavelength**2 - sellmeier_coefficients[1]) + (sellmeier_coefficients[2]*self.wavelength**2)/(self.wavelength**2 - sellmeier_coefficients[3]) + (sellmeier_coefficients[4]*self.wavelength**2) / (self.wavelength**2 - sellmeier_coefficients[5]))

        if name == 'alpha_bbo':
            return np.sqrt(sellmeier_coefficients[0] + (sellmeier_coefficients[1] / (self.wavelength ** 2 - sellmeier_coefficients[2])) - (sellmeier_coefficients[3] * self.wavelength ** 2))
        else:
            print('Crystal not yet implemented!')

    def non_axial_rays(self):

        x = np.arange(-int(self.nx / 2)+0.5, int(self.nx / 2), 1) * self.pixel_size
        y = np.arange(-int(self.ny / 2)+0.5, int(self.ny / 2), 1) * self.pixel_size

        xx, yy = np.meshgrid(x, y)

        self.alpha = np.arctan(np.sqrt(xx**2 + yy**2)/(self.focal_length*10**-6))
        self.beta = np.arctan2(yy,xx)

        return self.alpha, self.beta

    def frequency_dispersion(self):

        #Calculate derivatives using central differences method

        d_omega = np.gradient(self.angular_frequency)
        d_birefringence = np.gradient(self.birefringence)

        dB_dw = d_birefringence/d_omega

        self.kappa = 1 + (self.angular_frequency/self.birefringence)*(dB_dw)

        return self.kappa

    #total phase shift between ordinary and extraordinary rays due to birefringent material

    def phase_total(self):
        self.phi = ((2*np.pi*self.L)/(self.wavelength)) * ( (self.no**2 - constants.n_air**2*np.sin(self.alpha)**2)**0.5 + (constants.n_air*(self.no**2 - self.ne**2)*np.sin(self.cut_angle)*np.cos(self.cut_angle)*np.cos(self.delta)*np.sin(self.alpha))/(self.ne**2*np.sin(self.cut_angle)**2 + self.no**2*np.cos(self.cut_angle)**2) + ((-1*self.no * (self.ne**2 * (self.ne**2 * np.sin(self.cut_angle)**2 + (self.no**2 * np.cos(self.cut_angle)**2)) - (self.ne**2 - (self.ne**2 - self.no**2)*np.cos(self.cut_angle)**2 * np.sin(self.delta)**2)*constants.n_air**2*np.sin(self.alpha)**2)**0.5)/(self.ne**2*np.sin(self.cut_angle)**2 + self.no**2*np.cos(self.cut_angle)**2)))
        return self.phi

    #individual terms in equation when taken to specific limits

    def phase_offset(self):
        self.phi_0 = (-1 * 2* np.pi * self.L * self.birefringence * np.cos(self.cut_angle)**2) / (self.wavelength)
        return self.phi_0

    def phase_shear(self):
        self.phi_shear = (2 * np.pi * self.L*10**-6 * self.birefringence * np.sin(2*self.cut_angle))/(2*self.wavelength*10**-6 * ((self.ne + self.no)/2) * self.focal_length*10**-6)
        return self.phi_shear

    def phase_hyperbolic(self):
        self.phi_hyperbolic = -2*np.pi * self.L*10**-6 * self.birefringence / ( 4 * ((self.ne + self.no)/2)**2 * self.wavelength*10**-6 * (self.focal_length*10**-6)**2 )
        return self.phi_hyperbolic


#EXAMPLE
# nx = 1024
# ny = 1024
# pixel_size = 20*10**-6
# wavelength = 660*10**-9
# crystal_thickness = 16*10**-3
# cut_angle = 45.
# optic_axis = 90. #(ie. the crystal optical axis is vertical)
#
# #Everything must be given in meters, angles in degrees.
#
# delay_plate = Crystal(wavelength=wavelength, thickness=crystal_thickness, cut_angle=cut_angle, name='alpha_bbo', nx=nx, ny=ny, pixel_size=pixel_size, orientation=optic_axis, two_dimensional=False)
