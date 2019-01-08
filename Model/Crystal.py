import numpy as np
import matplotlib.pyplot as plt

from Model.Constants import Constants
from Model.Camera import Camera
from Model.Lens import Lens

constants = Constants()

class Crystal(object):

    def __init__(self, camera, lens, thickness, cut_angle, orientation, wavelength, name):

        """
        :param camera: Camera class with camera specifications - sensor size, pixel size etc.
        :param lens: Lens class with lens specifications - such as focal length, aperture etc.
        :param thickness: Thickness of the waveplate in metres.
        :param cut_angle: Cut angle of the optical axis in degrees. Number between 0 and 90
        :param orientation: Orientation angle of the crystal. 90 degrees is parallel to the y axis (normally the value to choose)
        :param wavelength: Wavelength of the incident light in metres
        :param name: Name of the crystal type - either alpha_bbo or lithium_niobate - will add more crystals as necessary

        Output: A waveplate with the following derived values:
                ne, no - extraordinary and ordinary refractive indices, calculated using sellmeier coefficients
                birefringence - ne - no
                phi - total phase shift due to the crystal, using the whole veiras formula
                phi_constant - constant phase shift (zero order term)
                phi_shear - linear phase shift (first order term)
                phi_hyperbolic - quadratic phase shift (second order term)
        """

        self.name = name
        self.thickness = thickness
        self.cut_angle = cut_angle*(np.pi/180.)
        self.orientation = orientation*(np.pi/180.)
        self.wavelength = wavelength

        if self.name == 'lithium_niobate':

            self.sc_ne = [2.9804, 0.02047, 0.5981, 0.0666, 8.9543, 416.08]
            self.sc_no = [2.6734, 0.01764, 1.2290, 0.05914, 12.614, 474.60]

            self.ne = self.refractive_index(sellmeier_coefficients=self.sc_ne, name=self.name)
            self.no = self.refractive_index(sellmeier_coefficients=self.sc_no, name=self.name)

        if self.name == 'alpha_bbo':

            self.sc_ne = [2.3753, 0.01224, 0.01667, 0.01516]
            self.sc_no = [2.7359, 0.01878, 0.01822, 0.01354]

            self.ne = self.refractive_index(sellmeier_coefficients=self.sc_ne, name=self.name)

            self.no = self.refractive_index(sellmeier_coefficients=self.sc_no, name=self.name)

            self.ne = np.sqrt(self.ne)
            self.no = np.sqrt(self.no)

        self.birefringence = self.ne - self.no

        x = np.linspace(-camera.sensor_size_x/2, camera.sensor_size_x/2, camera.n_pixels_x)
        y = np.linspace(-camera.sensor_size_y/2, camera.sensor_size_y/2, camera.n_pixels_y)

        self.xx, self.yy = np.meshgrid(x,y)

        self.alpha, self.beta = self.include_non_axial_ray_angles(lens)
        self.delta = self.beta - self.orientation

        self.phi = self.total_phase_delay()

        self.phi_constant, self.phi_shear, self.phi_hyperbolic = self.phase_delay_to_second_order(lens)

    def include_non_axial_ray_angles(self, lens):

        """
        Calculate the variation in angle deviation from axial (alpha)
        Calculate the azimuthal angles around the sensor xy plane (beta)

        :param lens: Lens class as specified above.
        :return: Alpha - Non-axial angle as a function of sensor position (Radians)
                 Beta - Azimuthal angle around the sensor (Radians)
        """

        self.alpha = np.arctan(np.sqrt((self.xx)**2 + (self.yy)**2)/(lens.focal_length))

        self.beta = np.arctan2(self.yy,self.xx) + np.pi

        return self.alpha, self.beta

    def refractive_index(self, sellmeier_coefficients, name):

        """

        :param sellmeier_coefficients: Coefficients required to calculate the refractive indices of the specific crystal material. Taken from A thormans thesis.. there are many others...
        :param name: Crystal type - either alpha_bbo or lithium_niobate
        :return: refractive index of the material
        """

        if name == 'alpha_bbo':
            return sellmeier_coefficients[0] + sellmeier_coefficients[1]/((self.wavelength*10**6)**2 - sellmeier_coefficients[2]) - (sellmeier_coefficients[3]*(self.wavelength*10**6)**2)

        if name == 'lithium_niobate':
            return 1 + (sellmeier_coefficients[0]*(self.wavelength*10**6)**2)/((self.wavelength*10**6)**2 - sellmeier_coefficients[1]) + (sellmeier_coefficients[2]*(self.wavelength*10**6)**2)/((self.wavelength*10**6)**2 - sellmeier_coefficients[3]) + (sellmeier_coefficients[4]*(self.wavelength*10**6)**2) / ((self.wavelength*10**6)**2 - sellmeier_coefficients[5])
        else:
            print('Crystal not yet implemented!')

    def total_phase_delay(self):

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

        t1 = (self.no * np.sqrt(self.ne**2 * (self.ne**2 * np.sin(self.cut_angle)**2 + self.no**2 * np.cos(self.cut_angle)**2 ) - (self.ne**2 - (self.ne**2 - self.no**2)*np.cos(self.cut_angle)**2 * np.sin(self.beta - self.orientation)**2)*constants.n_air**2 * np.sin(self.alpha)**2)) / (self.ne**2 *np.sin(self.cut_angle)**2 + self.no**2 * np.cos(self.cut_angle)**2)

        t2 = np.sqrt(self.no**2 - constants.n_air**2*np.sin(self.alpha)**2)

        t3 = (constants.n_air * (self.no**2 - self.ne**2) * np.sin(self.cut_angle) * np.cos(self.cut_angle) * np.cos(self.beta + self.orientation) * np.sin(self.alpha))/(self.ne**2 * np.sin(self.cut_angle)**2 + self.no**2 * np.cos(self.cut_angle)**2)

        self.phi = 2 * np.pi * (self.thickness/self.wavelength) * (t1 - t2 - t3)

        return self.phi

    def phase_delay_to_second_order(self, lens):

        """

        :param lens: Class containing information on the focussing lens used between the crystal and camera sensor (focal length in metres is required)
        :return: phi_constant: Constant phase delay offset due to zeroth order term in Veiras' formula (Radians)
                 phi_shear: Linear phase delay from first order term (radians)
                 phi_hyperbolic: Quadratic phase delay from second order term (radians)
        """

        # individual terms in equation when taken to specific limits - For more info read Alex Thorman's ANU thesis (2018)

        self.phi_constant = (2*np.pi / self.wavelength) * self.thickness * self.birefringence * np.cos(self.cut_angle)**2 * np.ones((np.shape(self.xx)))

        self.phi_shear = ( (-2 * np.pi / self.wavelength) * self.thickness * self.birefringence * np.sin(2*self.cut_angle) * self.yy) / ( ((self.ne + self.no)/2) * lens.focal_length )

        self.phi_hyperbolic = ((2 * np.pi / self.wavelength) * self.thickness * self.birefringence) / (4 * ((self.ne + self.no)/2)**2 * (lens.focal_length)**2) * ((3-np.cos(2*self.cut_angle))*(self.xx**2) - (3*np.cos(2*self.cut_angle)-1)*(self.yy**2))

        return self.phi_constant, self.phi_shear, self.phi_hyperbolic

#Example how to use

# camera = Camera(photron=False)
# lens = Lens(name='collection lens', focal_length=85*10**-3, diameter=None, aperture=None, f=None)
# crystal = Crystal(camera, lens, thickness=1*10**-3, cut_angle=45, orientation=90, wavelength=660*10**-9, name='alpha_bbo')
#
# levels = np.arange(-50,60,10)
#
# plt.figure()
# cs = plt.contour(crystal.xx, crystal.yy, crystal.phi_shear, levels=levels)
# plt.clabel(cs, inline=1, fontsize=10)
# plt.colorbar()
# plt.show()
