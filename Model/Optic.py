import numpy as np
from Model.Material import Material

from Model.Constants import Constants
constants = Constants()

class Polarizer():

    def __init__(self, theta):

        self.theta = theta*(np.pi/180.)

    def get_phase(self, *args):
        return self.theta

    def matrix(self, theta):

        return np.array([[1, 0, 0, 0],
                        [0, np.cos(2*theta), np.sin(2*theta), 0],
                        [0, -1*np.sin(2*theta), np.cos(2*theta), 0],
                        [0, 0, 0, 1]])

class Crystal(Material):

    def __init__(self, thickness, cut_angle, orientation, material_name):

        #thickness, cut_angle, orientation, material_name
        self.thickness = thickness
        self.cut_angle = cut_angle*(np.pi/180.)
        self.orientation = orientation*(np.pi/180.)

        #Inherent material properties based on the material we want
        self.material_name = material_name

        super().__init__(material_name)

    def calculate_birefringence(self, wavelength):

        self.ne = self.refractive_index(self.sc_ne, self.name, wavelength)
        self.no = self.refractive_index(self.sc_no, self.name, wavelength)

        self.birefringence = self.ne - self.no

    def get_phase(self, alpha, beta, wavelength):

        """
        Calculate the phase delay introduced by a uniaxial waveplate with an arbitrary optical axis orientation.

        Equation from "Phase shift formulas in uniaxial media: an application to waveplates", F. Veiras (2010). Orientation of the crystal is defined in this version of the equation as
        90 degrees = parallel to the y axis of the crystal. Taking limits of this equation (derivation in A. Thorman's ANU thesis 2018)

        Required values:

        wavelength: Wavelength of incident light in metres.

        ne & no: Extraordinary and ordinary refractive indices for a given crystal and wavelength

        beta: Azimuthal angle around the xy plane of the sensor (radians)

        alpha: Incident angle onto the sensor (alpha = 0 is axial) (radians)

        n_air: Refractive index of air

        thickness: Thickness of the waveplate (m)

        cut_angle: Orientation of the optical axis (radians)

        :return: Delay in radians due to the waveplate. Same dimensions as the sensor size.
        """

        t1 = (self.no * np.sqrt(self.ne**2 * (self.ne**2 * np.sin(self.cut_angle)**2 + self.no**2 * np.cos(self.cut_angle)**2 ) - (self.ne**2 - (self.ne**2 - self.no**2)*np.cos(self.cut_angle)**2 * np.sin(beta - self.orientation)**2)*constants.n_air**2 * np.sin(alpha)**2)) / (self.ne**2 *np.sin(self.cut_angle)**2 + self.no**2 * np.cos(self.cut_angle)**2)

        t2 = np.sqrt(self.no**2 - constants.n_air**2*np.sin(alpha)**2)

        t3 = (constants.n_air * (self.no**2 - self.ne**2) * np.sin(self.cut_angle) * np.cos(self.cut_angle) * np.cos(beta + self.orientation) * np.sin(alpha))/(self.ne**2 * np.sin(self.cut_angle)**2 + self.no**2 * np.cos(self.cut_angle)**2)

        self.phi = 2 * np.pi * (self.thickness/wavelength) * (t1 - t2 - t3)

        return self.phi

    def approximate_phase_delay(self, wavelength, x, y, lens):

        """

        :param lens: Class containing information on the focussing lens used between the crystal and camera sensor (focal length in metres is required)
        :return: phi_constant: Constant phase delay offset due to zeroth order term in Veiras' formula (Radians)
                 phi_shear: Linear phase delay from first order term (radians)
                 phi_hyperbolic: Quadratic phase delay from second order term (radians)
        """

        # individual terms in equation when taken to specific limits - For more info read Alex Thorman's ANU thesis (2018)

        self.phi_constant = (2*np.pi / wavelength) * self.thickness * self.birefringence * np.cos(self.cut_angle)**2 * np.ones((np.shape(x)))

        self.phi_shear = ( (-2 * np.pi / wavelength) * self.thickness * self.birefringence * np.sin(2*self.cut_angle) * y) / ( ((self.ne + self.no)/2) * lens.focal_length )

        self.phi_hyperbolic = ((2 * np.pi / wavelength) * self.thickness * self.birefringence) / (4 * ((self.ne + self.no)/2)**2 * (lens.focal_length)**2) * ((3-np.cos(2*self.cut_angle))*(x**2) - (3*np.cos(2*self.cut_angle)-1)*(y**2))

        return self.phi_constant, self.phi_shear, self.phi_hyperbolic

    def matrix(self, phi):

        """Muller matrix that describes how light propagates through a uniaxial crystal."""

        return np.array([[1, 0, 0, 0,],
                         [0, 1, 0, 0,],
                         [0, 0, np.cos(phi), -1*np.sin(phi)],
                         [0, 0, np.sin(phi), np.cos(phi)]])


class Lens():

    def __init__(self, focal_length, diameter, aperture, f):

        self.focal_length = focal_length
        self.diameter = diameter
        self.aperture = aperture
        self.f_number = f



