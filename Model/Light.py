from Tools.load_msesim import MSESIM
import numpy as np

class Light():

    def __init__(self, filepath):

        msesim = MSESIM(filepath=str(filepath))

        self.stokes_vector = msesim.stokes_vector
        self.S0 = msesim.S0
        self.S1 = msesim.S1
        self.S2 = msesim.S2
        self.S3 = msesim.S3

        self.wavelength = msesim.wavelength*10**-9

    def project(self, lens, camera):

        """
        :param lens: Lens object containing info about the focussing lens
        :param camera: Camera object, contains information about the camera sensor.
        :return:  Alpha (array): Angle light hits the sensor at, given that the rays exiting the crystal are non-axial
                  Beta (Array): Azimuthal angle around the camera sensor
        """

        #Find out where the light will focus onto the sensor.
        #We need to do this first as the shift between ordinary/extraordinary rays exiting the crystal depends on
        #the incident angle when the optical axis is not parallel to the plate surface.

        self.x, self.y = np.meshgrid(camera.x, camera.y)

        self.alpha = np.arctan(np.sqrt(self.x ** 2 + self.y ** 2) / lens.focal_length)

        self.beta = np.arctan2(self.y, self.x)

        return self.alpha, self.beta

    def interact(self, phi, S0, S1, S2):
        """

        :param phi: Phase delay imposed upon light from the uniaxial crystal
        :param S0: Initial Stokes component of the incident light (Total intensity)
        :param S1: Initial Stokes component of the incident light (Linearly polarised 45)
        :param S2: Initial Stokes component of the incident light (Linearly polarised -45)
        :return: Output Stokes component, S0.
        """
        """Find the output stokes vector after light has interacted with the crystal."""
        return S0 + S1*np.sin(phi) + S2*np.cos(phi)