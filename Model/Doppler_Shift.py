# External imports
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Internal imports

from Model.Physics_Constants import Constants, Conversions
from Model.Beam_Parameters import Beam
from Model.View import CollectionOptics
from Model.bbo_model import AlphaBBO

beam = Beam()
conversion = Conversions()
constant = Constants() #physics constant
optics = CollectionOptics()


def calc_doppler_shift():

    #find the beam vector

    beam_duct = np.array([0.539, -1.926, 0.]) #machine coords of beam duct

    R_duct = np.sqrt(beam_duct[0]**2 + beam_duct[1]**2) #R co-ordinates of beam duct
    phi_duct = np.arctan2(beam_duct[1],beam_duct[0]) #phi coord of beam duct

    beam_source = np.array([0.188, -6.88, 0.]) #machine coords of beam source

    R_source = np.sqrt(beam_source[0]**2 + beam_source[1]**2) #R co-ord beam source
    phi_source = np.arctan2(beam_source[1],beam_source[0]) #phi co-ord beam source

    collection_optics_point = np.array([-0.949, -2.228,  0.000]) #machine coords of collection lens

    #vector between source and duct
    source_to_duct = beam_duct - beam_source
    #normalise it to get the beam unit vector
    beam_axis = source_to_duct / (np.sqrt(source_to_duct[0]**2 + source_to_duct[1]**2 + source_to_duct[2]**2))

    sample_points = []
    emission_vectors = []
    R_to_beam = []

    #take some points along the beam from the duct to the tangency radius

    R_tangency = 0.6736

    distance_to_beam = np.linspace(R_duct, R_tangency, 5000)

    distance_along_beam = np.sqrt(distance_to_beam**2 - R_tangency**2)

    distance_along_beam = distance_along_beam[0] - distance_along_beam

    for i in range(len(distance_along_beam)):
        #points along the beam in xyz machine coords = beam_duct_point + beam_vector*length_along_beam
        sample_points.append(beam_duct + beam_axis*distance_along_beam[i])

        #vector between sample point and collection optics (beam points - collection optics position)
        emission_vector = sample_points[:][i] - collection_optics_point

        #normalise it
        normalised_emission_vector = emission_vector/(np.sqrt(emission_vector[0]**2 + emission_vector[1]**2))

        emission_vectors.append(normalised_emission_vector)

        #find the distance to the beam from each point on the collection lens
        R_to_beam.append(np.sqrt(emission_vector[0]**2 + emission_vector[1]**2))

    n1 = 2.0
    n2 = 3.0
    rest_wavelength = 656.1 * 10**-9 #m

    # Calculate the wavelength of light emission in vaccuum for a given transition
    # Rest wavelength emission in nm in vaccuum
    rydberg_constant = constant.rydberg/(1 + constant.mass_e/beam.mass)
    lambda_vac = (1./(rydberg_constant*(1/n1**2 - 1/n2**2)))

    lambda_air = lambda_vac/constant.n_air # Wavelength in air

    emission_vectors = np.array(emission_vectors)

    lambda_doppler = []

    for i in range(len(emission_vectors)):
        lambda_shift = rest_wavelength + (lambda_vac / constant.c) * beam.velocity * np.dot(beam_axis, emission_vectors[i,:])
        lambda_doppler.append(lambda_shift)

    major_radius = distance_to_beam
    lambda_doppler = np.array(lambda_doppler)

    return major_radius, lambda_doppler, beam_axis, emission_vectors, R_duct, R_tangency, sample_points

def calculate_Efield(sample_points, beam_axis):

    sample_points_r = []
    sample_points_phi = []
    sample_points_z = []

    for i in range(len(sample_points)):
        sample_points_r.append(np.sqrt(sample_points[i][0]**2 + sample_points[i][1]**2))
        sample_points_phi.append(np.arctan2(sample_points[i][1], sample_points[i][0]))
        sample_points_z.append(sample_points[i][2])

    sample_points_cylindrical = np.array([sample_points_r, sample_points_phi, sample_points_z])

    phi_hat = np.array([0, 1, 0])

    Bt = 1/(sample_points_cylindrical[0,:])

    E = []
    E_vector = []

    for i in range(len(sample_points_cylindrical[-1])):

        Bx = np.sin(sample_points_cylindrical[1,i])
        By = np.cos(sample_points_cylindrical[1,i])
        Bz = 0.

        B_xyz = np.array([Bx,By,Bz])

        E_field_vectors = beam.velocity * np.cross(beam_axis, B_xyz)

        E_mag = np.sqrt(E_field_vectors.dot(E_field_vectors))

        E_vector.append(E_field_vectors)
        E.append(E_mag)

    return E, E_vector

def calculate_intensities(E, E_vector, lambda_doppler, emission_vectors):

    r0 = 0.28
    r1 = 0.11
    r2 = 0.04
    r3 = 0.12
    r4 = 0.09

    m = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
    transition_weights = np.array([-r4, -r3, -r2, r1, r0, r1, -r2, -r3, -r4])

    I_polarised = np.zeros((len(transition_weights), len(emission_vectors)))
    I_unpolarised = np.zeros((len(transition_weights), len(emission_vectors)))
    lambda_stark = np.zeros((len(transition_weights), len(emission_vectors)))

    for i in range(len(emission_vectors)):

        stark_shift = m * 2.77 * 10 ** -7 * E[i] * (10 ** -10)

        lambda_stark[:,i] = stark_shift + lambda_doppler[i]

        I_polarised[:,i] = (1 - ((E_vector[:][i].dot(emission_vectors[i,:]))**2 / E_vector[:][i].dot(E_vector[:][i]))) * transition_weights

        I_unpolarised[:,i] = 2 * (E_vector[:][i].dot(emission_vectors[i,:]))**2/ E_vector[:][i].dot(E_vector[:][i]) * abs(transition_weights)

        I_unpolarised[0:3,i] = 0
        I_unpolarised[6:-1,i] = 0

    return I_polarised, I_unpolarised, lambda_stark

def add_delay(I_polarised, I_unpolarised, lambda_stark, major_radius, cut_angle, thickness):

    I_total = abs(I_polarised) + I_unpolarised

    contrast = []

    for i in range(len(I_polarised[-1])):

        wavelength = lambda_stark[:,i]

        bbo = AlphaBBO(wavelength, thickness, cut_angle)

        Intensity_displacer = I_polarised[:,i] * np.exp(1j*bbo.phi_0)

        contrast.append(abs(np.sum(Intensity_displacer))/np.sum(I_total[:,i]))

    print(contrast)

    plt.figure()
    plt.plot(major_radius, contrast)
    plt.xlabel('Major radius (m)')
    plt.ylabel('Contrast')
    plt.show()

    return

thickness = 10000  # um
cut_angle = 45  # degrees

major_radius, lambda_doppler, beam_axis, emission_vectors, R_duct, R_tangency, sample_points  = calc_doppler_shift()
E, E_vector = calculate_Efield(sample_points, beam_axis)
I_polarised, I_unpolarised, lambda_stark = calculate_intensities(E, E_vector, lambda_doppler, emission_vectors)
add_delay(I_polarised, I_unpolarised, lambda_stark, major_radius, cut_angle, thickness)


