#External imports
import numpy as np
from scipy.interpolate import interp1d
import idlbridge as idl

#Internal imports
from Model.scratch.Crystal import Crystal
from Model.Constants import Constants
from Tools.load_msesim import MSESIM
from Tools.Plotting.graph_format import plot_format

plot_format()

nx, ny = 32, 32
idl.execute("restore, '/home/sgibson/PycharmProjects/msesim/runs/imse_1d_32_f80mm/output/data/MAST_18501_imse.dat', /VERBOSE")
msesim = MSESIM(nx, ny, dimension=1)
data = msesim.load_msesim_spectrum()
constants = Constants()

def get_stokes_components():

    '''
    :param spectra_data: Spectral output from msesim, contains stokes components for linearly and circularly polarised light
    :param resolution: resoltuion grid of msesim run, contains the major radius points we view.
    :return: wavelength vector across field of view, total stokes components, major radius and linearly polarised S0.
    '''


    major_radius = data['resolution_vector(R)'][:,0]
    stokes_full = data['total_stokes']
    pi_stokes_full = data['pi_stokes']
    sigma_stokes_full = data['sigma_stokes']
    wavelength_vector = data['wavelength_vector']/10

    shape = stokes_full.shape

    linearly_polarised = np.zeros((shape[0], shape[2]))

    for i in range(len(stokes_full)):
        linearly_polarised[i,:] = stokes_full[i,1,:] + stokes_full[i,2,:]


    return major_radius, stokes_full, wavelength_vector, linearly_polarised

def get_intensities(stokes_full):

    '''
     Find the total stokes intensity S0.

    :param I_stokes_full: total S0 component
    :return:
    '''

    I_total = stokes_full[0,:]
    I_polarised = np.sqrt(stokes_full[1,:]**2 + stokes_full[2,:]**2) #total linear polarisation

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
    :return: Separate instances of the Crystal class - parameters for the displacer and delay plates.
    """
    delay = Crystal(wavelength=wavelength_vector*10**-9, thickness=delay_thickness, cut_angle=delay_cut_angle, name='alpha_bbo',
                    nx=32, ny=32, pixel_size=20*10**-6, orientation=90, two_dimensional=False)

    displacer = Crystal(wavelength=wavelength_vector*10**-9, thickness=displacer_thickness, cut_angle=displacer_cut_angle, name='alpha_bbo',
                          nx=32, ny=32, pixel_size=20*10**-6, orientation=90, two_dimensional=False)

    return delay, displacer

def add_delay(I_total, stokes_full, delay, displacer):

    """
    Add the delay to the stokes components of the incoming light - beginning the forward model.
    The intensity of the light after going through the delay and displacer plate is exp(iphi) * (S1 + S2/i) where phi is
    the total delay due to the delay and displacer plate. Currently only contains the constant offset, use the full Veries
    equation (see Crystal class - phi_total) for a more accurate measure of the phase shift

    To calculate the contrast we take the ratio of the sum of the real part of the polarised intensity to unpolarised intensity. It's equivalent
    to performing a fourier transform!

    :param I_total: Total SO intensity (I_polarised + I_unpolarised)
    :param stokes_full: Stokes vector array (S0, S1, S2, S3)
    :param delay: Instance of Crystal class corresponding to the delay plate.
    :param displacer: Instance of Crystal class corresponding to displacer plate.
    :return:
    """

    Intensity_displacer = np.exp(1j*(delay.phi_0+displacer.phi_0)) * (stokes_full[1,:]  + (stokes_full[2,:]/1j))

    #take real part of this

    contrast = abs(np.sum(Intensity_displacer))/np.sum(I_total)
    phase = np.arctan2(Intensity_displacer.imag, Intensity_displacer.real)

    total_delay = delay.phi_0 + displacer.phi_0

    return contrast, total_delay

def find_optimal_crystal(wavelength_vector, stokes_full, delay_thickness, delay_cut_angle, displacer_thickness, displacer_cut_angle):

    """
    Find out what the optimal delay plate thickness should be. Increased thickness means more delay, but increases weight of the higher order terms in the delay equation.
    We can fiddle with the thickness of the delay and the cut angle of the displacer to optimize that the delay should be the same as the splitting of pi/sigma lines. Change the
    cut angle of the displacer to change the number of fringes in the image.

    :param wavelength_vector: wavelengths of emission
    :param stokes: total stokes components S0 S1 S2 S3
    :return: contrast given the specific crystal parameters, total polarised intensity, number of waves delay as a function of wavelength
    """

    pixels = np.arange(1,33,1)

    shape = stokes_full.shape #n_pixels, n_stokes, n_intensities

    contrasts = np.zeros((len(pixels),len(displacer_cut_angle)))
    total_polarised = np.zeros((len(pixels),shape[2]))
    phi_total = []

    for i in range(len(pixels)):

        I_stokes_full = stokes_full[i,:]
        I_total, I_polarised, SN = get_intensities(I_stokes_full)
        total_polarised[i,:] = I_total

        for l in range(len(displacer_cut_angle)):
            delay, displacer = define_crystals(wavelength_vector, delay_thickness, delay_cut_angle, displacer_thickness, displacer_cut_angle[l])
            contrast, total_delay = add_delay(I_total, I_stokes_full, delay, displacer)
            contrasts[i,l] = contrast
            phi_total.append(total_delay)

    return contrasts, total_polarised, phi_total

def angle_of_incidence():

    """
    Not all rays will be axial to the crystal face. Can calculate the range of angles that the non axial rays will enter as - most severe at the edge.
    :return: alpha, beta = angles to define x-y polar coordinates on the sensor, effective focal length of the camera lens
    """

    nx = 32
    effective_focal_length = 85*10**-3 #mm
    pixel_x = np.linspace(-10.24*10**-3,10.24*10**-3,nx)
    pixel_ypositions = np.zeros((len(pixel_x)))
    alpha = np.arctan2(pixel_x, effective_focal_length)
    beta = np.arctan2(pixel_ypositions, pixel_x)

    return alpha, beta, pixel_x

def calc_fringe_frequency(displacer_cut_angle):

    """
    Calculate the number of pixels per fringe for a given cut angle and focal length lens
    :param displacer_cut_angle: Angle of optical axis of the displacer (0-90 degrees)
    :return: Number of pixels per fringe on the sensor
    """

    pixel_size = 20*10**-6
    focal_lengths = np.array([50*10**-3, 85*10**-3])
    pixels_per_fringe = np.zeros((len(displacer_cut_angle), len(focal_lengths)))


    for j in range(len(focal_lengths)):
        for i in range(len(displacer_cut_angle)):
            waves_per_pixel = (2*np.pi*(3*10**-3)*0.117*np.sin(2*displacer_cut_angle[i]*(np.pi/180.)))/(1.61*(focal_lengths[j])*(660*10**-9))
            waves_per_m = waves_per_pixel*pixel_size/(2*np.pi)
            pixels_per_fringe[i,j] = 1/waves_per_m

    return pixels_per_fringe

def find_fwhm(x,y):

    """
    Find full width half maximum of the transmission filter
    :param x: wavelength
    :param y: intensity
    :return: Full width half maximum of function
    """

    half_max = max(y) / 2.
    d = np.sign(half_max - np.array(y[0:-1])) - np.sign(half_max - np.array(y[1:]))
    left_idx = np.where(d > 0)[0]
    right_idx = np.where(d < 0)[-1]
    fwhm = x[right_idx] - x[left_idx]

    return fwhm

def filter(wavelength, alpha, tilt_angle=-1.5, n_material=2):

    lambda_step = wavelength[1] - wavelength[0]

    tilt_angle = tilt_angle * np.pi/180.
    fwhm = 0.5
    cwl_normal_incidence = 660.5
    transmission = []

    transmittance_peak = [10**-5,  10**-4, 10**-3, 10**-2, 10**-1,   0.5,    0.9,  1., 0.9, 0.5, 10**-1, 10**-2, 10**-3, 10**-4, 10**-5]

    for i in range(len(alpha)):

        cwl =  cwl_normal_incidence * np.sqrt(1 - (np.sin(alpha[i] + tilt_angle) ** 2)/n_material**2)

        multipliers = (np.array([-15,  -5.4,    -3.2,   -2.2,   -1.5,    -1.,  -0.65, 0., 0.65, 1., 1.5,     2.2,     3.2,    5.4,   15])*fwhm) + cwl

        interp_t = interp1d(multipliers, transmittance_peak)(wavelength)

        filter_fwhm = find_fwhm(wavelength, interp_t)

        transmission.append(interp_t)

    return transmission


major_radius, stokes_full, wavelength, linearly_polarised = get_stokes_components()
alpha, beta, pixel_x = angle_of_incidence()

print(np.max(alpha*(180./np.pi)))
#
# transmission = filter(wavelength, alpha, tilt_angle=0, n_material=2)
#
# wl, rr = np.meshgrid(wavelength, major_radius)
# spectrum = np.sqrt(stokes_full[:,1,:]**2 + stokes_full[:,2,:]**2)
# spectrum /= np.max(spectrum)
#
# delay_thickness = np.arange(5*10**-3,20*10**-3,1*10**-3)
#
# contrasts, total_polarised, phi_total = find_optimal_crystal(wavelength_vector=wavelength, stokes_full=stokes_full, delay_thickness=np.arange(5*10**-3,20*10**-3,1*10**-3), delay_cut_angle=0., displacer_thickness=3*10**-3, displacer_cut_angle=45.)
#
# dd, rr2 = np.meshgrid(delay_thickness, major_radius)
#
# plt.figure()
# plt.pcolormesh(rr2, dd*1000, contrasts, shading='gourand')
# plt.xlabel('R (m)')
# plt.ylabel('L (mm)')
# plt.colorbar()
# plt.show()
#
# plt.figure()
# plt.plot(wavelength, transmission[5][:])
# plt.plot(wavelength, spectrum[0,:])
# plt.plot(wavelength, transmission[-5][:])
# plt.plot(wavelength, spectrum[-5,:])
# plt.show()
