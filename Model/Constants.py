from numpy import pi, array, sin, cos

class Constants(object):

    """Any physical constants required throughout the code."""

    def __init__(self):
        self.mass_e = 9.11*10**-31 # Kg Electron mass
        self.mass_p = 1.6725*10**-27 # Kg Proton mass
        self.mass_n = 1.5748*10**-27 # Kg Neutron mass
        self.k      = 1.38*10**-23 # Boltzmann constant
        self.c      = 2.99*10**8   # m/s speed of light
        self.charge_e = 1.6*10**-19 #Coulombs, electron charge
        self.amu = 1.661*10**-27   # Atomic mass unit (Kg)
        self.rydberg = 1.097373*10**7 # Rydberg constant
        self.n_air = 1.0002926 # Refractive index of air
        self.planck = 6.62607*10**-34 # Planck constant
        self.mu_0 = 1.2566370614*10**-6 # Magnetic vacuum permeability
        self.epsilon_0 = 8.85*10**-12 # Electric vacuum permeability


class Conversions(object):

    def __init__(self):
        self.to_nm = 1**10 #convert to nm from m
        self.deg2rad = pi/180.
        self.rad2deg = 180./pi
