#External imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Internal imports
from Model.bbo_model import AlphaBBO
from Model.read_output import Channel, CentralPoints, GridPoints, Resolution, SpectralData
from Model.Physics_Constants import Constants

constants = Constants()
channel = Channel()
central_points = CentralPoints()
grid_points = GridPoints()
resolution = Resolution()
spectral_data = SpectralData()

major_radius = resolution.data['resolution_vector(R)'][:,0]

#Get the output from one pixel

stokes = spectral_data.data['total_stokes']
pi_stokes = spectral_data.data['pi_stokes']
sigma_stokes = spectral_data.data['sigma_stokes']
wavelength_vector = spectral_data.data['wavelength_vector']

def find_nonzero(array):
    nonzero = np.where(array > 0)
    return nonzero

def get_intensities(I_stokes):

    I_total = I_stokes[0,:] #Total intensity
    I_polarised = np.sqrt(I_stokes[1,:]**2 + I_stokes[2,:]**2) #total linear polarisation

    SN = I_polarised/np.sqrt(I_total)
    SN[np.isnan(SN)] = 0

    return I_total, I_polarised, SN

def define_crystals(wavelength_vector, delay_thickness, delay_cut_angle, displacer_thickness, displacer_cut_angle):

    delay = AlphaBBO(wavelength_vector, thickness=delay_thickness, cut_angle=delay_cut_angle)
    displacer = AlphaBBO(wavelength_vector, thickness=displacer_thickness, cut_angle=displacer_cut_angle)

    return delay, displacer

def add_delay(I_total, I_stokes, delay, displacer):

    Intensity_displacer = np.exp(1j*(delay.phi_0+displacer.phi_0)) * (I_stokes[1,:]  + (I_stokes[2,:]/1j))

    contrast = abs(np.sum(Intensity_displacer))/np.sum(I_total)

    return contrast

def find_optimal_delay_thickness(wavelength_vector):

    pixels = channel.data['channels']

    delay_thickness = [15000]
    delay_cut_angle = 0.

    displacer_cut_angle= 45.
    displacer_thickness = 3000

    contrasts = np.zeros((len(pixels),len(delay_thickness)))

    for i in range(len(pixels)):
        I_stokes = stokes[i, :, :]
        I_total, I_polarised, SN = get_intensities(I_stokes)
        for l in range(len(delay_thickness)):
            delay, displacer = define_crystals(wavelength_vector, delay_thickness[l], delay_cut_angle, displacer_thickness, displacer_cut_angle)
            contrast = add_delay(I_total, I_stokes, delay, displacer)
            contrasts[i,l] = contrast

    return contrasts

def angle_of_incidence():

    effective_focal_length = 78.
    pixel_xpositions = np.linspace(-10,10,21)
    pixel_ypositions = np.zeros((len(pixel_xpositions)))
    alpha = np.arctan2(np.sqrt(pixel_xpositions**2 + pixel_ypositions**2), effective_focal_length)
    beta = np.arctan2(pixel_ypositions, pixel_xpositions)

    return alpha, beta, pixel_xpositions

def design_filter(alpha, pixel_xpositions, wavelength_vector):

    lambda_0 = wavelength_vector[int(len(wavelength_vector)/2)]
    n_material = 2

    lambda_at_alpha = lambda_0 * np.sqrt(1 - (constants.n_air/n_material)**2 * (np.sin(alpha)**2))

    def find_nearest(array, value):
        idx = (np.abs(array - value)).argmin()
        return idx


    Tx = 1
    FWHM = 3 #nm

    filter_coefficients = [lambda_0 -15.*FWHM, lambda_0 -5.4*FWHM, lambda_0 - 3.2 * FWHM, lambda_0 -2.2*FWHM,
                           lambda_0 -1.5*FWHM, lambda_0 - FWHM, lambda_0 - 0.65*FWHM, lambda_0+0.65*FWHM, lambda_0 + FWHM,
                           lambda_0 + 1.5*FWHM,lambda_0 + 2.2*FWHM, lambda_0 + 3.2 * FWHM, lambda_0 + 5.4*FWHM, lambda_0 + 15.*FWHM]

    transmittance_peak = [10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 0.5*np.max(Tx), 0.9*np.max(Tx), 0.9*np.max(Tx),
                          0.5*np.max(Tx), 10**-1, 10**-2, 10**-3, 10**-4, 10**-5]

    interp_transmittance = np.interp(wavelength_vector, filter_coefficients,transmittance_peak)

    plt.figure()
    plt.plot(wavelength_vector, interp_transmittance)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Transmittance [%]')
    plt.show()

    # plt.figure()
    # plt.plot(pixel_xpositions, lambda_at_alpha*10**9, label='Filter $\lambda$')
    # plt.title('Central wavelength of the filter for a given angle of incidence')
    # plt.legend()
    # plt.xlabel('Pixel position from center of the sensor [mm]')
    # plt.ylabel('Wavelength (nm)')

    # plt.figure()
    # plt.title('Angle of incidence of light to normal of the filter ')
    # plt.plot(pixel_xpositions, alpha*(180./np.pi))
    # plt.xlabel('Pixel position from center of the sensor [mm]')
    # plt.ylabel('Angle of Incidence [deg]')

    return

alpha, beta, pixel_xpositions = angle_of_incidence()
design_filter(alpha, pixel_xpositions, wavelength_vector)
#contrasts = find_optimal_delay_thickness(wavelength_vector)


def plot_optimal_contrast(contrasts, major_radius):
    plt.figure()
    plt.title('Contrast for delay (15mm, $\Theta$=0) + displacer (3mm, $\Theta$=45$^{\circ}$) ')
    plt.plot(major_radius, contrasts[:, 0])
    plt.xlabel('Major Radius [m]')
    plt.ylabel('Contrast [Arb Units]')
    plt.show()
    return

def plot_contrast_contourplot(contrasts, delay_thickness, displacer_thickness, displacer_cut_angle, delay_cut_angle):

    pp,ll = np.meshgrid(delay_thickness,major_radius)

    levels=np.arange(0,0.43,0.05)

    plt.figure()
    plt.title('Displacer plate 3mm, $\Theta$=45$^{\circ}$')
    plt.pcolormesh(ll,pp/1000,contrasts,shading='gouraud')
    cbar = plt.colorbar()
    cbar.set_label('Contrast', rotation=90)
    plt.xlabel('Major Radius [m]')
    plt.ylabel('Delay plate thickness [mm]')
    plt.show()

    plt.figure()
    plt.title('Displacer plate 3mm, $\Theta$=45$^{\circ}$')
    CS =plt.contour(ll,pp/1000,contrasts, levels=levels)
    plt.clabel(CS, inline=1, fontsize=10)
    cbar2 = plt.colorbar()
    plt.xlabel('Major Radius [m]')
    plt.ylabel('Delay plate thickness [mm]')
    cbar2.set_label('Contrast', rotation=90)
    plt.show()

    return







