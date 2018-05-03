# External imports
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import collections as matcoll

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

    points_along_beam = np.arange(R_duct, 0.705, -0.01)

    for i in range(len(points_along_beam)):
        #points along the beam in xyz machine coords = beam_duct_point + beam_vector*length_along_beam
        sample_points.append(beam_duct + beam_axis*points_along_beam[i])

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

    major_radius = np.flipud(points_along_beam)
    lambda_doppler = np.array(lambda_doppler)

    return major_radius, lambda_doppler, beam_axis, emission_vectors

def calculate_Efield(major_radius, beam_axis):

    Bt = 1/major_radius

    Br = np.zeros((len(major_radius)))
    Bz = np.zeros((len(major_radius)))
    B_field = np.array([Br,Bt,Bz])

    E = []
    E_vector = []

    for i in range(len(major_radius)):
        E_field_vectors = beam.velocity * np.cross(beam_axis, B_field[:,i])

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

        stark_shift = m * 2.77 * 10 ** -7 * E[i] * 10 ** -10

        lambda_stark[:,i] = stark_shift * 10 ** -9 + lambda_doppler[i]

        I_polarised[:,i] = (1 - ((E_vector[:][i].dot(emission_vectors[i,:]))**2 / E_vector[:][i].dot(E_vector[:][i]))) * transition_weights

        I_unpolarised[:,i] = 2 * (E_vector[:][i].dot(emission_vectors[i,:]))**2/ E_vector[:][i].dot(E_vector[:][i]) * abs(transition_weights)

        I_unpolarised[0:3,i] = 0
        I_unpolarised[6:-1,i] = 0

    return I_polarised, I_unpolarised, lambda_stark

def add_delay(I_polarised, I_unpolarised, lambda_stark):

    I_total = abs(I_polarised) + I_unpolarised

    for i in range(len(I_polarised[-1])):

        bbo = AlphaBBO(lambda_stark[:,i], constant)

        Intensity_displacer = abs(I_polarised[:,i]) * np.exp(1j*bbo.phi_0)

        contrast = abs(np.sum(Intensity_displacer))/np.sum(I_total[:,i])

        print(contrast)


    return


def run():
    major_radius, lambda_doppler, beam_axis, emission_vectors = calc_doppler_shift()

    E, E_vector = calculate_Efield(major_radius, beam_axis)
    I_polarised, I_unpolarised, lambda_stark = calculate_intensities(E, E_vector, lambda_doppler, emission_vectors)
    add_delay(I_polarised, I_unpolarised, lambda_stark)
    return

run()


#
# m = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
#
# #Relative weights of the intensities of the transitions given a statistical population
# r0 = 0.28
# r1 = 0.11
# r2 = 0.04
# r3 = 0.12
# r4 = 0.09
#
# transitions = np.array([-r4, -r3, -r2, r1, r0, r1, -r2, -r3, -r4]) #* beam_emission_intensity
#
# lambda_fromduct = lambda_shift[r_at0:]
# emission_vector = emission_vector[:,r_at0:]
#
# wavelength_transitions = []
# I_p = []
# I_up = []
# I_tot = []
#
# #Assume the E field = VxB + Er (Er = 0 for simplicity), B = Bt = 1/R ~ 0.6T on MAST
# Bt = np.array([0, 2.5*1.2, 0])
#
# E_field_vector = beam_velocity * np.cross(beam_vector/np.sqrt(beam_vector.dot(beam_vector)), Bt)
#
# beam_vector_norm = beam_vector/np.sqrt(beam_vector.dot(beam_vector))
# bt_norm = Bt/ np.sqrt(Bt.dot(Bt))
#
# stark_shift = m * 2.77*10**-7 * np.sqrt(E_field_vector.dot(E_field_vector)) * 10**-10 # linear_shift * magnetic quantum number * mag(E fld) * Angstroms (V/m)
#
# for i in range(len(lambda_fromduct)):
#
#     lambda_d = lambda_fromduct[i]
#
#     lambda_transitions = stark_shift*10**-9 + lambda_d
#
#     wavelength_transitions.append(lambda_transitions)
#
#     I_polarised = ( 1 - (E_field_vector.dot(emission_vector[:,i])))**2/E_field_vector.dot(E_field_vector) * transitions
#
#     I_p.append(I_polarised)
#
#     I_unpolarised = 2 * (E_field_vector.dot(emission_vector[:,i]))**2/E_field_vector.dot(E_field_vector) * abs(transitions)
#
#     #I = 0 for m = +/-4, 3, 2
#     I_unpolarised[0:3]=0.
#     I_unpolarised[6:-1]=0.
#     I_up.append(I_unpolarised)
#
#     I_total = abs(I_polarised) + I_unpolarised
#     I_tot.append(I_total)
#
# I_tot = np.asarray(I_tot)
# I_up = np.asarray(I_up)
# I_p = np.asarray(I_p)
#
# eta = []
#
# for i in range(107):
#     wavelengths = np.array([wavelength_transitions])[0,i,:] *1*10**6
#
#     bbo = AlphaBBO(wavelengths, constant)
#
#     Intensity_displacer = I_p[i,:] * np.exp(1j*bbo.phi_0)
#
#     contrast = abs(np.sum(Intensity_displacer))/np.sum(I_tot[i,:])
#
#     eta.append(contrast)
#
# eta = abs(np.array([eta]))[0,:]
#
# lambdas = np.array([wavelength_transitions])[0,:,0]
#
# # plt.figure()
# # plt.plot(bbo.phi_0, eta)
# # plt.xlabel('delay')
# # plt.ylabel('Constrast')
# # plt.show()
# #
# # plt.figure()
# # markerline, stemlines, baseline = plt.stem(wavelength_transitions[50]*10**9, transitions, linefmt='--', markerfmt=' ')
# # plt.setp(baseline, color='black', linewidth=2)
# # plt.setp(stemlines[0:3], color='blue')
# # plt.setp(stemlines[6:9], color='blue')
# # plt.setp(stemlines[3:6], color='red')
# # plt.ylabel('Transition Intensity')
# # plt.xlabel('Wavelength (m)')
# # plt.show()
#
# # plt.figure()
# # plt.plot(r_fromduct[r_at0:], lambda_shift[r_at0:]*10**9)
# # plt.xlabel('Distance along Beam from the Duct (m)')
# # plt.ylabel('Doppler Shifted Wavelength (nm)')
# # plt.show()


# with open('beam.pkl', 'rb') as handle:
#     beam_params = pickle.load(handle)
#
# yc = beam_params['xc']
# zc = beam_params['yc']
# xc = beam_params['zc']
#
# # xyz co-ordinates of the collection optics
#
# collection_optics_vector = np.array([-0.949, -2.228, 0.000])  # xyz
#
# # Find the points along the beam in machine-coordinates.
#
# xc = np.asarray(xc) - abs(beam.source_coordinates[1])
# yc = np.asarray(yc) - abs(beam.source_coordinates[2])
# zc = np.asarray(zc) - abs(beam.source_coordinates[0])
#
# idx_0 = abs(xc - beam.yxz[1]).argmin()
#
# x_machine = xc[idx_0:]
# y_machine = yc[idx_0:]
# z_machine = zc[idx_0:]
#
# view_vectors = np.array([x_machine, y_machine, z_machine])
#
# def beam_geometry():
#
#     #Define some sample points along the beam in machine co-ordinates
#
#     with open('beam_sample2.pkl', 'rb') as handle:
#         beam_params = pickle.load(handle)
#
#     #Where are the collection optics in machine co-ordinates
#     collection_optics_vector = np.array([-0.949, -2.228,  0.000])
#
#     x_coords = beam_params['xc'] #- beam.vector[2]
#     y_coords = beam_params['yc'] #- beam.vector[0]
#     z_coords = beam_params['zc'] #- beam.vector[1]
#
#     print(z_coords)
#
#     coords = np.array([x_coords, y_coords, z_coords])
#
#     return coords, collection_optics_vector
#
# def cartesian2cylindrical(x,y,z):
#
#     r = np.asarray(np.sqrt(x**2 + y**2))
#     phi = np.asarray(np.arctan2(y,x))
#
#     return np.array([r,phi,z])
#
# def normalise(vector):
#
#     vector_transpose = np.transpose(vector)
#     mod = np.sqrt(np.dot(vector, vector_transpose))
#     return vector / mod
#
# def doppler_shift(lambda_vac, coords):
#
#     print(coords.shape)
#
#     lambda_shift = rest_wavelength + (lambda_vac / constant.c) * beam.velocity * np.dot(beam.vector, coords)
#
#     #calculate for all points (which include outside the duct, so cut from the beam duct)
#
#     coords_r = np.sqrt(coords[0]**2 + coords[1]**2)
#     beam_source_r = np.sqrt(beam.source_coordinates[0]**2 + beam.source_coordinates[1]**2)
#
#     duct_r = abs(coords_r - beam_source_r - 0).argmin()
#
#     lambda_shift = lambda_shift[duct_r:]
#     coords_from_duct = coords[:,duct_r:]
#
#     return lambda_shift, coords_from_duct
#
# def get_major_radii(coords_from_duct):
#
#     coordinates = cartesian2cylindrical(coords_from_duct[0,:], coords_from_duct[1,:], coords_from_duct[2,:])
#
#     distance_to_center_fromduct = beam.xyz
#
#     r_distance = np.sqrt(distance_to_center_fromduct[1]**2 - distance_to_center_fromduct[0]**2)
#
#     r_tangency = 0.705
#
#     major_radius = np.linspace(r_distance, r_tangency, len(coordinates[-1]))
#
#     return major_radius, coordinates

