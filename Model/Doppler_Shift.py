# External imports
import numpy as np

# Internal imports

from Model.Physics_Constants import Constants, Conversions
from Model.Beam_Parameters import Beam
from Model.View import CollectionOptics

beam = Beam()
conversion = Conversions()
constant = Constants() #physics constant
optics = CollectionOptics()

# Define beam parameters (this should be a separate class once we want to implement a comprehensive beam code)

beam_energy = 65*10**3 # 65kV Beam -full energy component
beam_mass = constant.mass_p # Assume Hydrogen beam for now
beam_velocity = np.sqrt(2*constant.charge_e*beam_energy/beam_mass)

xi = beam.xi*conversion.deg2rad
delta = beam.delta*conversion.deg2rad

beam_vector = np.array([np.cos(xi)*np.sin(delta), np.sin(xi)*np.sin(delta), np.cos(delta)])

print(beam_vector)

beam_source_coordinates = beam.xyz - beam.distance_source2duct*beam_vector

print(beam_source_coordinates)

beam_length = beam.xyz - beam_vector

print(beam_length)

rydberg_constant = constant.rydberg/(1 + constant.mass_e/beam_mass)

# Calculate the original transition wavelengths
# Principle quantum numbers associated with the H-alpha transition

n1 = 2.0
n2 = 3.0

# Calculate the wavelength of light emission in vaccuum for a given transition
# Rest wavelength emission in nm in vaccuum

lambda_vac = (1./(rydberg_constant*(1/n1**2 - 1/n2**2)))*conversion.to_nm
lambda_air = lambda_vac/constant.n_air # Wavelength in air






# Calculate Doppler Shifts for each position

lambda_shift = - (lambda_vac/constant.c) * np.dot(beam_velocity * beam_vector, view_vector)




