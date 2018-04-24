
class Beam(object):

    def __init__(self):
        self.xyz = [0.539, -1.926, 0.0] #xyz position of duct
        self.xi = 85.16 #89.1 # Angle between x-axis and beam axis
        self.delta = 90 #Angle between z-axis and beam axis
        self.distance_source2duct = 4.98 # Distance between beam source and beam duct
        self.f_horizontal = 14.00 # Horizontal focus of beam (from source)
        self.f_vertical = 6.00 # Vertical focus of beam (from source)

        #pini stuff left out - probably need to add this in?

        self.hsw = 0.15 # Half sampling width of beam (degrees)
        self.divergence = 0.5 # Beam divergence (degrees)
        self.ionisation_rate = 5*10**-20 #Ionisation rate of beam particles per m (m**2)
        self.emission_rate = 2.5*10**-18 # Emission rate of beam particles per second (photons m**3/s)

