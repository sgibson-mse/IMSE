import idlbridge as idl
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
import pandas as pd

from Tools.MSESIM import MSESIM
from Tools.demodulate_TSH_synthetic_image import demodulate_image

from Model.Constants import Constants
from Model.Crystal import Crystal
from Model.Camera import Camera


import numpy as np

nx, ny = 32, 32
idl.execute("restore, '/home/sgibson/PycharmProjects/msesim/runs/imse_2d_32x32_f80mm_opticaxis/output/data/MAST_18501_imse.dat', /VERBOSE")
msesim = MSESIM(nx, ny)
data = msesim.load_msesim_spectrum()
constants = Constants()
camera = Camera(photron=True)


def define_crystals(wavelength, delay_thickness, delay_cut_angle, displacer_thickness, displacer_cut_angle):

    """
    Design the crystals we want to place in the system to calculate optimal delay. Here we use a 15mm delay plate + 3mm delay plate with 45 degree cut angle.

    :param wavelength_vector: Doppler shifted wavelengths.
    :param delay_thickness: Thickness of the delay plate in mm
    :param delay_cut_angle: Optic axis cut angle of the delay plate in degrees
    :param displacer_thickness: Displacer plate thickness in mm
    :param displacer_cut_angle: Displacer plate cut angle in degrees
    :return: Separate instances of the Crystal class - parameters for the displacer and delay plates.
    """
    delay = Crystal(wavelength=wavelength, thickness=delay_thickness, cut_angle=delay_cut_angle, name='alpha_bbo',
                    nx=1024, ny=1024, pixel_size=camera.pixel_size, orientation=90, two_dimensional=False)

    displacer = Crystal(wavelength=wavelength, thickness=displacer_thickness, cut_angle=displacer_cut_angle, name='alpha_bbo',
                          nx=1024, ny=1024, pixel_size=camera.pixel_size, orientation=90, two_dimensional=False)

    return delay, displacer

def add_delay(delay_thickness, delay_cut_angle, displacer_thickness, displacer_cut_angle):

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

    I0 = np.zeros((1024,1024,len(msesim.wavelength)), dtype='complex')
    phase = np.zeros((1024,1024,len(msesim.wavelength)))

    x_small = np.linspace(-camera.sensor_size / 2, camera.sensor_size / 2, 32)
    y_small = np.linspace(-camera.sensor_size / 2, camera.sensor_size / 2, 32)

    x = np.linspace(-camera.sensor_size / 2, camera.sensor_size / 2, 1024)
    y = np.linspace(-camera.sensor_size / 2, camera.sensor_size / 2, 1024)

    S0 = np.zeros((1024,1024,len(msesim.wavelength)))

    for i in range(len(msesim.wavelength)):

        S0_interp = interp2d(x_small, y_small, msesim.S0[:, :, i], kind='quintic')
        S0[:,:,i] = S0_interp(x, y)

        S1_interp = interp2d(x_small, y_small, msesim.S1[:, :, i], kind='quintic')
        S1 = S1_interp(x, y)

        S2_interp = interp2d(x_small, y_small, msesim.S1[:, :, i], kind='quintic')
        S2 = S2_interp(x, y)

        delay, displacer = define_crystals(msesim.wavelength[i]*10**-10, delay_thickness, delay_cut_angle, displacer_thickness, displacer_cut_angle)

        S_total = np.exp(1j * (delay.phi_0 + delay.phi_shear + displacer.phi_0 + displacer.phi_shear)) * (S2+1j*S1)

        #S_total = np.exp(1j * (delay.phi_total + displacer.phi_total)) * (S1 + (S2 / 1j))

        I0[:, :, i] = S_total
        phase[:, :, i] = np.arctan2(S_total.imag, S_total.real)

    contrast = abs(np.sum(I0, axis=2))/np.sum(S0, axis=2)
    phase = np.sum(phase, axis=2)

    return contrast, phase

def demodulated_contrast(image):
    phase, contrast, dc_amplitude = demodulate_image(image)
    return contrast

def calculate_ideal_contrast():
    delay_thickness = 15*10**-3
    delay_cut_angle = 0.
    displacer_thickness = 3*10**-3
    displacer_cut_angle = 45.*np.pi/180.

    contrast, phase = add_delay(delay_thickness, delay_cut_angle, displacer_thickness, displacer_cut_angle)

    r_big = np.linspace(np.min(msesim.major_radius), np.max(msesim.major_radius), 1024)
    z_big = np.linspace(np.min(msesim.z), np.max(msesim.z), 1024)

    rr, zz = np.meshgrid(r_big[::-1], z_big)

    levels = np.arange(0,110,10)

    plt.figure(1)
    CS1 = plt.pcolormesh(rr, zz, contrast*100, shading='gourand', rasterized=True)
    CS = plt.contour(rr, zz, contrast*100, '--', colors='black', levels=levels)
    plt.clabel(CS, inline=True, fontsize=14, inline_spacing=10, manual=True)
    cbar = plt.colorbar(CS1)
    cbar.ax.set_ylabel('Contrast ($\%$)')
    cbar.add_lines(CS)
    plt.xlabel('R (m)')
    plt.ylabel('Z (m)')

    plt.figure(2)
    plt.plot(r_big[::-1],contrast[512,:]*100)
    plt.xlabel('R (m)')
    plt.ylabel('Contrast ($\%$)')
    plt.show()

def demod_contrast():

    image_file = pd.HDFStore(path='/home/sgibson/PycharmProjects/IMSE/Images/f80mm_opticaxis/TSH_1.hdf')
    image = image_file['/a']

    contrast_demod = demodulated_contrast(image.values)

    r_big = np.linspace(np.min(msesim.major_radius), np.max(msesim.major_radius), 1024)
    z_big = np.linspace(np.min(msesim.z), np.max(msesim.z), 1024)

    rr, zz = np.meshgrid(r_big[::-1], z_big)

    levels = np.arange(0,110,5)

    plt.figure(1)
    CS1 = plt.pcolormesh(rr[300:700], zz[300:700], contrast_demod[300:700]*100, shading='gourand', rasterized=True)
    CS = plt.contour(rr[300:700], zz[300:700], contrast_demod[300:700]*100, '--', colors='black', levels=levels)
    plt.clabel(CS, inline=True, fontsize=14, inline_spacing=10, manual=True)
    cbar = plt.colorbar(CS1)
    cbar.ax.set_ylabel('Contrast ($\%$)')
    cbar.add_lines(CS)
    plt.xlabel('R (m)')
    plt.ylabel('Z (m)')

    plt.figure(2)
    plt.plot(r_big[::-1], contrast_demod[512,:]*100)
    plt.xlabel('R (m)')
    plt.ylabel('Contrast ($\%$)')
    plt.show()

demod_contrast()