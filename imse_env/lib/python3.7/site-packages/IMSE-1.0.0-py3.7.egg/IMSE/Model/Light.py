from Tools.load_msesim import MSESIM
import numpy as np

class Light():

    def __init__(self, filepath, dimension, msesim=True):

        if msesim:

            msesim = MSESIM(filepath, dimension)

            self.stokes_vector = msesim.stokes_vector
            self.S0 = msesim.S0
            self.S1 = msesim.S1
            self.S2 = msesim.S2
            self.S3 = msesim.S3
            self.x = msesim.x
            self.y = msesim.y

            self.wavelength = msesim.wavelength*10**-9

        else:

            self.S0 = np.ones((nx, ny))
            self.S1 = 0.5*np.ones((nx, ny))*np.cos(gamma)
            self.S2 = 0.5*np.ones((nx, ny))*np.sin(gamma)
            self.S3 = np.zeroes((nx,ny))
            self.x = np.linspace(0,1,nx)
            self.y = np.linspace(0,1,ny)


    def interact(self, phi, S0, S1, S2, FLC):
        """
        :param phi: Phase delay imposed upon light from the uniaxial crystal
        :param S0: Initial Stokes component of the incident light (Total intensity)
        :param S1: Initial Stokes component of the incident light (Linearly polarised 45)
        :param S2: Initial Stokes component of the incident light (Linearly polarised -45)
        :return: Output Stokes component, S0.
        """
        """Find the output stokes vector after light has interacted with the crystal."""
        if FLC == 45:
            return S0 + S1*np.sin(phi) - S2*np.cos(phi)
        if FLC == 90:
            return S0 +S1*np.sin(phi) + S2*np.cos(phi)
        if FLC == None:
            return #w/e the equation is for the ash system