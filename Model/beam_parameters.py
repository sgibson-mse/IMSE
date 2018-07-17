import numpy as np
from Model.Constants import Constants, Conversions

constants = Constants()
conversion = Conversions()

class Beam(object):

    def __init__(self):
        self.beam_duct = np.array([0.539, -1.926, 0.0]) #xyz position of duct
        self.xi = 85.16 * np.pi/180.  #89.1 # Angle between x-axis and beam axis
        self.delta = 90 * np.pi/180. #Angle between z-axis and beam axis
        self.distance_source2duct = 4.98 # Distance between beam source and beam duct
        self.f_horizontal = 14.00 # Horizontal focus of beam (from source)
        self.f_vertical = 6.00 # Vertical focus of beam (from source)

        #pini stuff left out - probably need to add this in?

        self.hsw = 0.15 # Half sampling width of beam (degrees)
        self.divergence = 0.5 # Beam divergence (degrees)
        self.ionisation_rate = 5*10**-20 #Ionisation rate of beam particles per m (m**2)
        self.emission_rate = 2.5*10**-18 # Emission rate of beam particles per second (photons m**3/s)

        self.energy = 63 * 10 ** 3  # 65kV Beam -full energy component
        self.mass = constants.mass_p  # Assume Hydrogen beam for now
        self.velocity = np.sqrt(2 * constants.charge_e * self.energy / self.mass)

        self.emission_rate = 2.5e-18
        self.emission_intensity = self.emission_rate / (4. * np.pi)

        self.vector = np.array([np.cos(self.xi) * np.sin(self.delta), np.sin(self.xi) * np.sin(self.delta), np.cos(self.delta)])

        self.source_coordinates = self.beam_duct - self.distance_source2duct * self.vector

        self.length = self.beam_duct + self.vector



