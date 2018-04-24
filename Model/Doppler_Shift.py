# External imports
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import collections as matcoll

# Internal imports

from Model.Physics_Constants import Constants, Conversions
from Model.Beam_Parameters import Beam
from Model.View import CollectionOptics

beam = Beam()
conversion = Conversions()
constant = Constants() #physics constant
optics = CollectionOptics()

with open('beam.pkl', 'rb') as handle:
    beam_params = pickle.load(handle)

# Define beam parameters (this should be a separate class once we want to implement a comprehensive beam code)

beam_energy = 60*10**3 # 65kV Beam -full energy component
beam_mass = constant.mass_p # Assume Hydrogen beam for now
beam_velocity = np.sqrt(2*constant.charge_e*beam_energy/beam_mass)
beam_emission_rate = 2.5e-18
beam_emission_intensity = beam_emission_rate/(4.*np.pi)

xi = beam.xi*conversion.deg2rad
delta = beam.delta*conversion.deg2rad

beam_vector = np.array([np.cos(xi)*np.sin(delta), np.sin(xi)*np.sin(delta), np.cos(delta)])

beam_source_coordinates = beam.xyz - beam.distance_source2duct*beam_vector

beam_length = beam.xyz + beam_vector

rydberg_constant = constant.rydberg/(1 + constant.mass_e/beam_mass)

# Calculate the original transition wavelengths
# Principle quantum numbers associated with the H-alpha transition

n1 = 2.0
n2 = 3.0
rest_wavelength = 656.1 * 10**-9 #m

# Calculate the wavelength of light emission in vaccuum for a given transition
# Rest wavelength emission in nm in vaccuum

lambda_vac = (1./(rydberg_constant*(1/n1**2 - 1/n2**2)))*conversion.to_nm
lambda_air = lambda_vac/constant.n_air # Wavelength in air

#Read in the xyz co-ordinates of sample points along the beam

yc = beam_params['xc']
zc = beam_params['yc']
xc = beam_params['zc']

# xyz co-ordinates of the collection optics

coll_optics_xyz = [-0.949, -2.228,  0.000]

#Find the emission vectors

xc = np.asarray(xc) - beam_vector[1]
yc = np.asarray(yc) - beam_vector[0]
zc = np.asarray(zc) - beam_vector[2]

emission_vector = np.array([xc,yc,zc])

r = np.sqrt(yc**2 + xc**2 + zc**2)

r_beamsource = np.sqrt(beam_source_coordinates[0]**2 + beam_source_coordinates[1]**2 + beam_source_coordinates[2]**2)

r_at0 = abs(r - r_beamsource - 0).argmin()

r_fromduct = r - r_beamsource # Redefine R such that 0 is the duct, not the source

# Calculate Doppler shifts for each position

lambda_shift = rest_wavelength + (lambda_vac/constant.c) *  beam_velocity * np.dot(beam_vector, emission_vector)

m = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
stark_shift = m * 2.77*10*-7 *(10**6)*(10**-10) # linear_shift * magnetic quantum number * mag(E fld) * Angstroms (V/m)

#Relative weights of the intensities of the transitions given a statistical population
r0 = 0.28
r1 = 0.11
r2 = 0.04
r3 = 0.12
r4 = 0.09

transitions = np.array([-r4, -r3, -r2, r1, r0, r1, -r2, -r3, -r4]) * beam_emission_intensity


psi = emission_vector

I_polarised = np.array([-r4, -r3, -r2, r1, r0, r1, -r2, -r3, -r4]) * beam_emission_intensity * np.sin(psi)**2

I_polarised = np.array([-r4, -r3, -r2, r1*(1+np.cos(psi)**2), r0*(1+np.cos(psi)**2), r1*(1+np.cos(psi)**2), -r2, -r3, -r4])\
              * beam_emission_intensity




wavelength_transitions = []
Intensity_fft = []
I_polarized = []

#E_field_vector = np.cross(beam_velocity*beam_vector, Bfield_vector)

lambda_fromduct = lambda_shift[r_at0:]

for i in range(len(lambda_fromduct)):

    lambda_d = lambda_fromduct[i]

    lambda_transitions = stark_shift*10**-9 + lambda_d
    wavelength_transitions.append(lambda_transitions)

    fft_transitions = np.fft.fft(transitions)

    fft_I = np.sum(fft_transitions)
    Intensity_fft.append(fft_I)

    I_polarised = np.sum(abs(transitions))
    I_polarized.append(I_polarised)

    #I_unpolarised = 2 * np.dot(E_field_vector, emission_vector)/10**6 * transitions

    I_total = I_polarised #+ I_unpolarised

    polarised_fraction = I_polarised/I_total

contrast = abs(np.array([Intensity_fft]))/np.array([I_polarized])
print(contrast)

plt.figure()
markerline, stemlines, baseline = plt.stem(wavelength_transitions[50]*10**9, transitions, linefmt='--', markerfmt=' ')
#markerline, stemlines, baseline = plt.stem(wavelength_transitions[20]*10**9, transitions, linefmt='--', markerfmt=' ')
#markerline, stemlines, baseline = plt.stem(wavelength_transitions[40]*10**9, transitions, linefmt='--', markerfmt=' ')
#markerline, stemlines, baseline = plt.stem(wavelength_transitions[60]*10**9, transitions, linefmt='--', markerfmt=' ')
#markerline, stemlines, baseline = plt.stem(wavelength_transitions[80]*10**9, transitions, linefmt='--', markerfmt=' ')
#markerline, stemlines, baseline = plt.stem(wavelength_transitions[100]*10**9, transitions, linefmt='--', markerfmt=' ')
plt.setp(baseline, color='black', linewidth=2)
plt.setp(stemlines[0:3], color='blue')
plt.setp(stemlines[6:9], color='blue')
plt.setp(stemlines[3:6], color='red')
plt.ylabel('Transition Intensity')
plt.xlabel('Wavelength (m)')
plt.show()

# plt.figure()
# plt.plot(r_fromduct[r_at0:], lambda_shift[r_at0:]*10**9)
# plt.xlabel('Distance along Beam from the Duct (m)')
# plt.ylabel('Doppler Shifted Wavelength (nm)')
# plt.show()



