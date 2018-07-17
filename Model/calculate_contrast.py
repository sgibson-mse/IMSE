#External imports
import numpy as np
import matplotlib.pyplot as plt
import idlbridge as idl
from scipy.interpolate import interp1d
from scipy import interp

#Internal imports
from Model.Crystal import Crystal
from Model.read_fullenergy import ChannelFull, ResolutionFull, SpectralDataFull
from Model.read_halfenergy import ResolutionHalf, SpectralDataHalf
from Model.read_thirdenergy import ResolutionThird, SpectralDataThird
from Model.Constants import Constants

SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def get_stokes_components(spectral_data, resolution):

    '''

    :param spectra_data: Spectral output from msesim, contains stokes components for linearly and circularly polarised light
    :param resolution: resoltuion grid of msesim run, contains the major radius points we view.
    :return: wavelength vector across field of view, total stokes components, major radius and linearly polarised S0.
    '''

    major_radius = resolution.data['resolution_vector(R)'][:,0]
    stokes_full = spectral_data.data['total_stokes']
    pi_stokes_full = spectral_data.data['pi_stokes']
    sigma_stokes_full = spectral_data.data['sigma_stokes']
    wavelength_vector = spectral_data.data['wavelength_vector']

    shape = stokes_full.shape

    linearly_polarised = np.zeros((shape[0], shape[2]))

    for i in range(len(stokes_full)):
        linearly_polarised[i,:] = np.sqrt(stokes_full[i,1,:]**2 + stokes_full[i,2,:]**2)
        #linearly_polarised[i,:] = linearly_polarised[i,:]/np.max(linearly_polarised[i,:])

    # plt.figure()
    # plt.title('Stokes components at R = {}m'.format("%.2f" % major_radius[1]))
    # plt.plot(wavelength_vector/10, sigma_stokes_full[1,0,:]/10**7, color='black', label='S0 Total Intensity')
    # plt.plot(wavelength_vector/10, linearly_polarised[1,:]/10**7, '--', color='red', label='Linearly polarised')
    # plt.plot(wavelength_vector/10, sigma_stokes_full[1,3,:]/10**7, '-.', color='blue', label='Circularly Polarised')
    # plt.legend(prop={'size': 10})
    # plt.xlim(659.2,659.8)
    # plt.xlabel('Wavelength (nm)')
    # plt.ylabel('Intensity [10$^{7}$photons/s]')
    # plt.show()

    return major_radius, stokes_full, wavelength_vector, linearly_polarised

def get_intensities(I_stokes_full):

    '''
     Find the total stokes intensity S0.

    :param I_stokes_full: total S0 component
    :return:
    '''

    I_total = I_stokes_full[0,:]
    I_polarised = np.sqrt(I_stokes_full[1,:]**2 + I_stokes_full[2,:]**2) #total linear polarisation


    SN = I_polarised/np.sqrt(I_total)
    SN[np.isnan(SN)] = 0

    return I_total, I_polarised, SN

def define_crystals(wavelength_vector, delay_thickness, delay_cut_angle, displacer_thickness, displacer_cut_angle):

    """
    Design the crystals we want to place in the system to calculate optimal delay. Here we use a 15mm delay plate + 3mm delay plate with 45 degree cut angle.

    :param wavelength_vector: Doppler shifted wavelengths.
    :param delay_thickness: Thickness of the delay plate in mm
    :param delay_cut_angle: Optic axis cut angle of the delay plate in degrees
    :param displacer_thickness: Displacer plate thickness in mm
    :param displacer_cut_angle: Displacer plate cut angle in degrees
    :return: Separate instances of the AlphaBBO class - parameters for the displacer and delay plates.
    """
    delay = Crystal(wavelength=wavelength_vector, thickness=delay_thickness, cut_angle=delay_cut_angle, name='alpha_bbo',
                    nx=21, ny=21, pixel_size=20 * 10 ** -6, orientation=90, two_dimensional=False)

    displacer = Crystal(wavelength=wavelength_vector, thickness=displacer_thickness, cut_angle=displacer_cut_angle, name='alpha_bbo',
                          nx=21, ny=21, pixel_size=20*10**-6, orientation=90, two_dimensional=False)

    return delay, displacer

def add_delay(I_total, I_stokes_full, delay, displacer):

    """
    Add the delay to the stokes components of the incoming light - beginning the forward model.
    The intensity of the light after going through the delay and displacer plate is exp(iphi) * (S1 + S2/i) where phi is
    the total delay due to the delay and displacer plate. Currently only contains the constant offset, but this phi should
    include the linear ramp and the hyperbolic terms.

    To calculate the contrast we take the ratio of the sum of the real part of the polarised intensity to unpolarised intensity. It's equivalent
    to performing a fourier transform!

    :param I_total: Total SO intensity (I_polarised + I_unpolarised)
    :param I_stokes: Total linearly polarised intensity (S0, S1, S2, S3)
    :param delay: Instance of AlphaBBO class corresponding to the delay plate.
    :param displacer: Instance of AlphaBBO class corresponding to displacer plate.
    :return:
    """

    Intensity_displacer = np.exp(1j*(delay.phi_0+displacer.phi_0)) * (I_stokes_full[1,:]  + (I_stokes_full[2,:]/1j))

    #take real part of this

    contrast = abs(np.sum(Intensity_displacer))/np.sum(I_total)

    total_delay = delay.phi_0 + displacer.phi_0

    return contrast, total_delay

def angle_of_incidence():

    effective_focal_length = 85 #mm
    pixel_xpositions = np.linspace(-10.24,10.24,21)
    pixel_ypositions = np.zeros((len(pixel_xpositions)))
    alpha = np.arctan2(pixel_xpositions, effective_focal_length)
    beta = np.arctan2(pixel_ypositions, pixel_xpositions)

    return alpha, beta, pixel_xpositions, effective_focal_length

def find_optimal_delay_thickness(wavelength_vector, stokes_full, channel_full):

    """
    Find out what the optimal delay plate thickness should be. Increased thickness means more delay, but increases weight of the higher order terms in the delay equation.
    We can fiddle with the thickness of the delay and the cut angle of the displacer to optimize that the delay should be the same as the splitting of pi/sigma lines. Change the
    cut angle of the displacer to change the number of fringes in the image.

    :param wavelength_vector: wavelengths of emission
    :param stokes: total stokes components S0 S1 S2 S3
    :return: contrast given the specific crystal parameters
    """

    pixels = channel_full.data['channels']

    shape = stokes_full.shape #n_pixels, n_stokes, n_intensities

    contrasts = np.zeros((len(pixels),len(displacer_cut_angle)))
    total_polarised = np.zeros((len(pixels),shape[2]))
    phi_total = []

    for i in range(len(pixels)):

        I_stokes_full = stokes_full[i, :, :]
        I_total, I_polarised, SN = get_intensities(I_stokes_full)
        total_polarised[i,:] = I_total

        for l in range(len(displacer_cut_angle)):
            delay, displacer = define_crystals(wavelength_vector, delay_thickness, delay_cut_angle, displacer_thickness, displacer_cut_angle[l])
            contrast, total_delay = add_delay(I_total, I_stokes_full, delay, displacer)
            contrasts[i,l] = contrast
            phi_total.append(total_delay)

    return contrasts, total_polarised, phi_total

def calc_fringe_frequency(wavelength_vector_1, displacer_cut_angle):

    wavelength = 660*10**-9
    pixel_size = 20*10**-6
    focal_lengths = np.array([50*10**-3, 85*10**-3])
    pixels_per_fringe = np.zeros((len(displacer_cut_angle), len(focal_lengths)))


    for j in range(len(focal_lengths)):
        for i in range(len(displacer_cut_angle)):
            waves_per_pixel = (2*np.pi*(3*10**-3)*0.117*np.sin(2*displacer_cut_angle[i]*(np.pi/180.)))/(1.61*(focal_lengths[j])*(660*10**-9))
            waves_per_m = waves_per_pixel*pixel_size/(2*np.pi)
            pixels_per_fringe[i,j] = 1/waves_per_m

    'plot of pixels per fringe for a given focal length lens, varying displacer cut angle'

    plt.figure()
    plt.plot(displacer_cut_angle, pixels_per_fringe[:,0], label='$f$ = 50mm')
    plt.plot(displacer_cut_angle, pixels_per_fringe[:,1], label='$f$ = 85mm')
    plt.legend()
    plt.ylim(5,20)
    plt.xlabel('Displacer cut angle [degrees]')
    plt.ylabel('Number of pixels per fringe')
    plt.show()

    plt.figure()
    plt.plot(displacer_cut_angle, 1/np.array(pixels_per_fringe))
    plt.xlabel('Displacer cut angle [degrees]')
    plt.ylabel('fringes per pixel')
    plt.show()

    return pixels_per_fringe

def find_fwhm(x,y):

    half_max = max(y) / 2.
    d = np.sign(half_max - np.array(y[0:-1])) - np.sign(half_max - np.array(y[1:]))
    left_idx = np.where(d > 0)[0]
    right_idx = np.where(d < 0)[-1]
    fwhm = x[right_idx] - x[left_idx]

    return fwhm

def design_filter_2(alpha, lambda_0, n_material, tilt_angle, wavelength_vector_1, scale_fwhm):

    cavity_filter = np.loadtxt('3_cavity_filter.txt')
    x_vals = np.arange(0, len(cavity_filter), 1)

    lambda_nm = wavelength_vector_1/10

    lambda_step = wavelength_vector_1[1]/10-wavelength_vector_1[0]/10
    lambda_min = np.min(wavelength_vector_1/10)
    lambda_max = np.max(wavelength_vector_1/10)

    lambda_filter = np.arange(-len(cavity_filter)/2. + 0.5, len(cavity_filter)/2. +0.5, 1)

    fwhm_val = find_fwhm(x_vals, cavity_filter)

    lambdas = lambda_0 * np.sqrt(1 - (constants.n_air / n_material) ** 2 * (np.sin((alpha + tilt_angle)*(np.pi/180.)) ** 2))

    interp_transmittance = []

    for i in range(len(lambdas)):

        lambda_values = (lambda_filter * scale_fwhm / fwhm_val[0]) + lambdas[i]

        interp_t = interp1d(lambda_values, cavity_filter)(lambda_nm)
        interp_transmittance.append(interp_t)

    interp_trans = np.array([interp_transmittance])[0,:,:]

    return interp_transmittance, lambda_nm

delay_thickness = 15000.
delay_cut_angle = 0.

displacer_cut_angle = np.arange(0.,90.,1)
displacer_thickness = 3000

# define a central wavelength for the filter - the middle of our wavelength array for now
lambda_0 = 660.5
scale_fwhm = 1.9

n_material = 2  # refractive index of the material

tilt_angle = -1 * (np.pi / 180.)

constants = Constants()

# FULL ENERGY COMPONENT
idl.execute("restore, '/home/sam/Desktop/msesim/runs/mast_imse_photron/output/data/density2e19m3_MAST_photron.dat', /VERBOSE")

channel_full = ChannelFull()
resolution_full = ResolutionFull()
spectral_full = SpectralDataFull()

major_radius_1, stokes_1, wavelength_vector_1, linearly_polarised_1 = get_stokes_components(spectral_full,
                                                                                            resolution_full)
contrasts, total_polarised, phi_total = find_optimal_delay_thickness(wavelength_vector_1, stokes_1, channel_full)
alpha, beta, pixel_xpositions, effective_focal_length = angle_of_incidence()

# calc_fringe_frequency(wavelength_vector_1,displacer_cut_angle)


def plot_full_energy_component(channel_full, resolution_full, spectral_full):

    major_radius_1, stokes_1, wavelength_vector_1, linearly_polarised_1 = get_stokes_components(spectral_full, resolution_full)

    print('Wavelengths at 63kV -', wavelength_vector_1/10)

    contrasts, total_polarised, phi_total = find_optimal_delay_thickness(wavelength_vector_1, stokes_1, channel_full)
    alpha, beta, pixel_xpositions, effective_focal_length = angle_of_incidence()

    interp_transmittance_full, lambda_nm = design_filter_2(alpha, lambda_0, n_material, tilt_angle, wavelength_vector_1, scale_fwhm)

    I_full = stokes_1[:,1,:] / np.max(abs(stokes_1[:, 1, :]))

    ll, rr = np.meshgrid(wavelength_vector_1 / 10, major_radius_1)

    plt.figure(1)
    plt.title('Filter CWL $\lambda$={} nm, FWHM = {} nm EFL = {} mm - Full Energy'.format(lambda_0, scale_fwhm, effective_focal_length))
    cont = plt.contour(ll, rr, interp_transmittance_full, cmap='gray')
    plt.clabel(cont, inline=1, fontsize=10)
    plt.pcolormesh(ll, rr, I_full)
    cs = plt.colorbar()
    cs.set_label('Intensity [Arb Units]', rotation=90.)
    plt.ylabel('Major radius [m]')
    plt.xlabel('Wavelength of spectrum [nm]')
    plt.show()

    return ll, rr, interp_transmittance_full, lambda_nm

def plot_half_energy_component():
    # #HALF ENERGY COMPONENT

    idl.execute(
        "restore, '/home/sam/Desktop/msesim/runs/mast_imse_photron_half/output/data/density2e19m3_MAST_photron.dat' , /VERBOSE")

    resolution_half = ResolutionHalf()
    spectral_half = SpectralDataHalf()

    major_radius_2, stokes_2, wavelength_vector_2, linearly_polarised_2 = get_stokes_components(spectral_half,
                                                                                                resolution_half)
    lambda_half, r_half = np.meshgrid(wavelength_vector_2 / 10, major_radius_2)
    I_half = stokes_2[:, 1, :] / np.max(abs(stokes_2[:, 1, :]))
    interp_transmittance_half, lambda_nm = design_filter_2(alpha, lambda_0, n_material, tilt_angle, wavelength_vector_1, scale_fwhm)

    ll, rr = np.meshgrid(wavelength_vector_1 / 10, major_radius_1)

    plt.figure(2)
    plt.title('Filter CWL $\lambda$={} nm, FWHM = {} nm, EFL = {} mm - Full Energy'.format(lambda_0, scale_fwhm,
                                                                                           effective_focal_length))
    cont2 = plt.contour(ll, rr, interp_transmittance_half, cmap='gray')
    plt.clabel(cont2, inline=1, fontsize=10)
    plt.pcolormesh(lambda_half, r_half, I_half)
    cs = plt.colorbar()
    cs.set_label('Intensity [Arb Units]', rotation=90.)
    plt.ylabel('Major radius [m]')
    plt.xlabel('Wavelength of spectrum [nm]')
    plt.show()

    return interp_transmittance_half, lambda_half

#ll, r, interp_transmittance_full, lambda_nm = plot_full_energy_component(channel_full, resolution_full, spectral_full)
#interp_transmittance_half, lambda_half = plot_half_energy_component()

#Plotting code

def plot_nonaxial_angles(pixel_xpositions, lambda_at_alpha, alpha):
    plt.figure()
    plt.plot(pixel_xpositions, lambda_at_alpha*10**9, label='Filter $\lambda$')
    plt.title('Central wavelength of the filter for a given angle of incidence')
    plt.legend()
    plt.xlabel('Pixel position from center of the sensor [mm]')
    plt.ylabel('Wavelength (nm)')

    plt.figure()
    plt.title('Angle of incidence of light to normal of the filter ')
    plt.plot(pixel_xpositions, alpha*(180./np.pi))
    plt.xlabel('Pixel position from center of the sensor [mm]')
    plt.ylabel('Angle of Incidence [deg]')
    return


def plot_optimal_contrast(contrasts, major_radius):
    plt.figure()
    plt.title('Contrast for delay (15mm, $\Theta$=0) + displacer (3mm, $\Theta$=45$^{\circ}$) ')
    plt.plot(major_radius, contrasts[:, 0])
    plt.xlabel('Major Radius [m]')
    plt.ylabel('Contrast [Arb Units]')
    plt.show()
    return

def plot_contrast_contourplot(contrasts, delay_thickness, major_radius):

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

def plot_displacer_thickness(contrasts, displacer_thickness, major_radius):

    pp,ll = np.meshgrid(displacer_thickness,major_radius)

    levels=np.arange(0,0.43,0.05)

    plt.figure()
    plt.title('Delay plate = 15mm, Displacer Cut angle $\Theta$=45$^{\circ}$')
    plt.pcolormesh(ll,pp/1000,contrasts,shading='gouraud')
    cbar = plt.colorbar()
    cbar.set_label('Contrast', rotation=90)
    plt.xlabel('Major Radius [m]')
    plt.ylabel('Displacer plate thickness [mm]')
    plt.show()

    plt.figure()
    plt.title('Delay plate = 15mm, Displacer Cut angle $\Theta$=45$^{\circ}$')
    CS =plt.contour(ll,pp/1000,contrasts, levels=levels)
    plt.clabel(CS, inline=1, fontsize=10)
    cbar2 = plt.colorbar()
    plt.xlabel('Major Radius [m]')
    plt.ylabel('Displacer plate thickness [mm]')
    cbar2.set_label('Contrast', rotation=90)
    plt.show()

    return
