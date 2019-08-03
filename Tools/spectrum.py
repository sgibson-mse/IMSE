from raysect.core import Point3D, Vector3D, translate, rotate_basis
import numpy as np
from IMSE.Model.Constants import Constants
import matplotlib.pyplot as plt
from IMSE.Tools.Plotting.graph_format import plot_format

cb = plot_format()
constant = Constants()

def calculate_doppler_shift(beam_duct, beam_span, lens_pos, beam_velocity):

    emission_vectors = []
    sample_points = []
    doppler_shift = []

    #Calculate the beam vector
    beam_vector = beam_source.vector_to(beam_duct).normalise()

    #Sample some points along the beam, calculate the vector to the lens for each point,
    #Calculate the Doppler shift at each position

    for i, point in enumerate(beam_span):
        sample = beam_duct + beam_vector * point
        emission_vector = sample.vector_to(lens_pos).normalise()

        wavelength_shift = rest_wavelength - (lambda_vacuum/constant.c) * beam_velocity * beam_vector.dot(emission_vector)

        emission_vectors.append(emission_vector)
        sample_points.append(sample)
        doppler_shift.append(wavelength_shift)

    return beam_vector, emission_vectors, sample_points, doppler_shift

def calculate_Efield(sample_points, beam_velocity, beam_vector):

    sr = []
    sphi = []
    sz = []

    phi_hat = Vector3D(0, 1, 0)
    E = []
    E_vectors = []

    for i, point in enumerate(sample_points):

        sr = np.sqrt(point.x**2 + point.y**2)
        sphi = np.arctan(point.y/point.x)
        sz = point.z

        Bt = 1 / sr
        B = Vector3D(np.sin(sphi), np.cos(sphi), 0)

        E_vector = beam_velocity * beam_vector.cross(B)
        E_vectors.append(E_vector)

        E_mag = np.sqrt(E_vector.dot(E_vector))
        E.append(E_mag)

    return E, E_vectors

def calculate_line_intensities(E, E_vectors, emission_vectors, doppler_shift):

    #Given weights of the intensities of each pi and sigma transition

    r0 = 0.28
    r1 = 0.11
    r2 = 0.04
    r3 = 0.12
    r4 = 0.09

    m = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])

    transition_weights = np.array([-r4, -r3, -r2, r1, r0, r1, -r2, -r3, -r4])

    I_polarised = np.zeros((len(transition_weights), len(emission_vectors)))
    I_unpolarised = np.zeros((len(transition_weights), len(emission_vectors)))
    stark_wavelength = np.zeros((len(transition_weights), len(emission_vectors)))

    for i, E in enumerate(E_vectors):

        stark_shift = m * 2.77 * 10 ** -7 * E.z * (10 ** -10)

        print('stark shift', stark_shift, 'doppler shift', doppler_shift[i])

        stark_wavelength[:,i] = stark_shift + doppler_shift[i]

        I_polarised[:,i] = (1 - ((E.dot(emission_vectors[i]))**2 / E.dot(E))) * transition_weights

        I_unpolarised[:,i] = 2 * (E.dot(emission_vectors[i]))**2/ E.dot(E) * abs(transition_weights)

        I_unpolarised[0:3,i] = 0
        I_unpolarised[6:-1,i] = 0

    return I_polarised, I_unpolarised, stark_wavelength

def gaussian(x, mu, a, sig):
    return a*(np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))

def plot_spectrum(stark_wavelength, I_polarised, stark_wavelength2, I_polarised2, stark_wavelength3, I_polarised3):

    wavelengths = np.arange(655,663,0.01)
    pi_mu = [660.725, 661.37]
    sigma_mu = [661.05]
    sigma_intensity = [0.28]
    intensity = [-0.128, -0.128]
    sd = 0.085

    dalpha_mu = 656.28
    dalpha_sd = 0.28

    pi_mu_half = [658.427,658.719]
    pi_sd_half = [0.075]

    sigma_mu_half = [658.575]
    sigma_sd_half = [0.075]

    pi_mu_third = [657.64, 657.86]
    pi_sd_third = [0.03]

    sigma_mu_third = [657.75]
    sigma_sd_third = [0.03]

    plt.figure()

    #first energy
    textstr = '$E$'
    plt.annotate(textstr, xy=(661, 0.31), fontsize=28, color='black')
    markerline, stemlines, baseline = plt.stem(stark_wavelength[:, 0] * 10 ** 9, I_polarised[:, 0],
                                                markerfmt=' ')
    plt.setp(baseline, color='black', linewidth=2)
    plt.setp(stemlines[0:3], color=cb[3], linewidth=2)
    plt.setp(stemlines[6:9], color=cb[3],  linewidth=2)
    plt.setp(stemlines[3:6], color=cb[0],  linewidth=2)

    #second energy
    textstr = '$E/2$'
    plt.annotate(textstr, xy=(658.6, 0.31), fontsize=28, color='black')
    markerline, stemlines, baseline = plt.stem(stark_wavelength2[:, 0] * 10 ** 9, I_polarised2[:, 0],
                                               markerfmt=' ')
    plt.setp(baseline, color='black', linewidth=2)
    plt.setp(stemlines[0:3], color=cb[3], linewidth=2)
    plt.setp(stemlines[6:9], color=cb[3], linewidth=2)
    plt.setp(stemlines[3:6], color=cb[0], linewidth=2)

    #third energy

    textstr = '$E/3$'
    plt.annotate(textstr, xy=(657.6, 0.31), fontsize=28, color='black')
    markerline, stemlines, baseline = plt.stem(stark_wavelength3[:, 0] * 10 ** 9, I_polarised3[:, 0],
                                               markerfmt=' ')
    plt.setp(baseline, color='black', linewidth=2)
    plt.setp(stemlines[0:3], color=cb[3], linewidth=2)
    plt.setp(stemlines[6:9], color=cb[3], linewidth=2)
    plt.setp(stemlines[3:6], color=cb[0], linewidth=2)

    plt.ylabel('Intensity (Arb Units)')
    plt.xlabel('Wavelength (nm)')


    for m, mean in enumerate(pi_mu):
        plt.plot(wavelengths, gaussian(wavelengths, mean, intensity[m], sd), color='black')
        plt.fill(wavelengths, gaussian(wavelengths, mean, intensity[m], sd), alpha=0.55, color=cb[3])

    for m2, mean2 in enumerate(pi_mu_half):
        plt.plot(wavelengths, gaussian(wavelengths, mean2, intensity[m2], pi_sd_half), color='black')
        plt.fill(wavelengths, gaussian(wavelengths, mean2, intensity[m2], pi_sd_half), alpha=0.55, color=cb[3])

    for m3, mean3 in enumerate(pi_mu_third):
        plt.plot(wavelengths, gaussian(wavelengths, mean3, intensity[m3], pi_sd_third), color='black')
        plt.fill(wavelengths, gaussian(wavelengths, mean3, intensity[m3], pi_sd_third), alpha=0.55, color=cb[3])

    plt.plot(wavelengths, gaussian(wavelengths, sigma_mu, sigma_intensity, sd), color='black')
    plt.fill(wavelengths, gaussian(wavelengths, sigma_mu, sigma_intensity, sd), alpha=0.55, color=cb[0])

    plt.plot(wavelengths, gaussian(wavelengths, sigma_mu_half, sigma_intensity, sigma_sd_half), color='black')
    plt.fill(wavelengths, gaussian(wavelengths, sigma_mu_half, sigma_intensity, sigma_sd_half), alpha=0.55, color=cb[0])


    plt.plot(wavelengths, gaussian(wavelengths, sigma_mu_third, sigma_intensity, sigma_sd_third), color='black')
    plt.fill(wavelengths, gaussian(wavelengths, sigma_mu_third, sigma_intensity, sigma_sd_third), alpha=0.55, color=cb[0])

    textstr = '$D_{\\alpha}$'
    plt.annotate(textstr, xy=(655.8, 0.489), fontsize=28, color='black')
    plt.plot(wavelengths, gaussian(wavelengths, dalpha_mu, [0.5], dalpha_sd), color='black')

    plt.fill(wavelengths, gaussian(wavelengths, dalpha_mu, [0.5], dalpha_sd), color='gray', alpha=0.5)

    plt.show()

    return

#quantum number n - balmer alpha transition n=3 to n=2

n1 = 2.0
n2 = 3.0

#rest wavelength of balmer alpha emission in m

rest_wavelength = 656.1 * 10**-9

#Rydberg constant for calculating delta lambda of the transition
rydberg_constant = constant.rydberg/(1 + constant.mass_e/constant.mass_p)
lambda_vacuum = (1./(rydberg_constant*(1/n1**2 - 1/n2**2)))
lambda_air = lambda_vacuum/constant.n_air

#Beam energy in eV
beam_energy = 75 * 10 ** 3
#Beam velocity in m/s
beam_velocity = np.sqrt(2 * constant.charge_e * beam_energy / constant.mass_p)

#Poisition of the beam duct and PINI grid in machine co-ordinates
beam_duct = Point3D(0.539, -1.926, 0.)
beam_source = Point3D(0.188, -6.88, 0.)

#Position of the collection optics in machine co-ordinates
lens_pos = Point3D(-0.949, -2.228,  0.000)

#Calculate the radial position of the duct in toroidal co-ordinates
R_duct = np.sqrt(beam_duct.x**2 + beam_duct.y**2)

#Tangency radius for MAST
Rt = 0.6736

#Define some points along the beam from the duct to the tangency radius
beam_span = np.linspace(Rt,R_duct,100)

#first energy component
beam_vector, emission_vectors, sample_points, doppler_shift = calculate_doppler_shift(beam_duct, beam_span, lens_pos, beam_velocity)
E, E_vectors = calculate_Efield(sample_points, beam_velocity, beam_vector)
I_polarised, I_unpolarised, stark_wavelength = calculate_line_intensities(E, E_vectors, emission_vectors, doppler_shift)

#second energy component
beam_vector2, emission_vectors2, sample_points2, doppler_shift2 = calculate_doppler_shift(beam_duct, beam_span, lens_pos, beam_velocity/2)
E2, E_vectors2 = calculate_Efield(sample_points2, beam_velocity/2, beam_vector2)
I_polarised2, I_unpolarised2, stark_wavelength2 = calculate_line_intensities(E2, E_vectors2, emission_vectors2, doppler_shift2)

#third energy
beam_vector3, emission_vectors3, sample_points3, doppler_shift3 = calculate_doppler_shift(beam_duct, beam_span, lens_pos, beam_velocity/3)
E3, E_vectors3 = calculate_Efield(sample_points3, beam_velocity/3, beam_vector3)
I_polarised3, I_unpolarised3, stark_wavelength3 = calculate_line_intensities(E3, E_vectors3, emission_vectors3, doppler_shift3)


plot_spectrum(stark_wavelength, I_polarised, stark_wavelength2, I_polarised2, stark_wavelength3, I_polarised3)

Er_r = [0.90, 0.93, 0.96, 0.99, 1.03, 1.06, 1.09, 1.12, 1.15, 1.18, 1.22, 1.25, 1.28, 1.31, 1.34, 1.37, 1.41, 1.44,
        1.47, 1.50]
Er = [3.00E+02, 5.70E+03, 1.02E+04, 1.36E+04, 1.61E+04, 1.79E+04, 1.90E+04, 1.96E+04, 1.92E+04, 1.80E+04, 1.43E+04, 8.60E+03,
      -2.50E+03, -4.40E+03, -3.80E+03, -1.90E+03, -8.00E+02, -4.00E+02, -1.00E+01, 0.00]




