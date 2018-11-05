
from Model.Crystal import Crystal
from Model.Camera import Camera
from Tools.MSESIM import MSESIM

import idlbridge as idl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp2d
import pandas as pd

idl.execute("restore, '/home/sgibson/PycharmProjects/msesim/runs/imse_2d_32x32_f80mm_opticaxis/output/data/MAST_18501_imse.dat', /VERBOSE")

nx, ny = 32, 32
camera = Camera(photron=True)
msesim = MSESIM(nx,ny)
data = msesim.load_msesim_spectrum()

def simulate_TSH(FLC_state):

    """
    Calculate the intensity of an image for the imaging MSE system with an FLC as a function of wavelength.
    :param FLC_state: 1 or 2, switches between 45 degrees and 90 degrees polarisation state
    :return: Intensity on image, summed up over wavelength.
    """

    #Large interpolation x and y
    x = np.linspace(-camera.sensor_size/2, camera.sensor_size/2, camera.n_pixels)
    y = np.linspace(-camera.sensor_size/2, camera.sensor_size/2, camera.n_pixels)

    S_total = np.zeros((len(x), len(y), len(msesim.wavelength)))

    for i in range(len(msesim.wavelength)):
        #Make the delay and displacer plate to find the phase shift for a given wavelength

        delay_plate = Crystal(wavelength=msesim.wavelength[i]*10**-10, thickness=0.015, cut_angle=0., name='alpha_bbo', nx=len(x), ny=len(y), pixel_size=camera.pixel_size, orientation=90., two_dimensional=True)
        displacer_plate = Crystal(wavelength=msesim.wavelength[i]*10**-10, thickness=0.003, cut_angle=45., name='alpha_bbo', nx=len(x), ny=len(y), pixel_size=camera.pixel_size, orientation=90.,two_dimensional=True)

        #Interpolate our small msesim grid to the real image size, 1024x1024

        S0_interp = interp2d(msesim.pixel_x, msesim.pixel_y, msesim.S0[:, :, i], kind='quintic')
        S0 = S0_interp(x,y)

        S1_interp = interp2d(msesim.pixel_x, msesim.pixel_y, msesim.S1[:, :, i], kind='quintic')
        S1 = S1_interp(x,y)

        S2_interp = interp2d(msesim.pixel_x, msesim.pixel_y, msesim.S2[:, :, i], kind='quintic')
        S2 = S2_interp(x,y)

        #Calculate the total intensity for a given wavelength, propagating stokes components through a delay plate and a displacer plate.

        if FLC_state == 1:
            S_total[:,:,i] = S0 + S1*np.sin((delay_plate.phi_total + displacer_plate.phi_total)) + S2*np.cos((delay_plate.phi_total + displacer_plate.phi_total))

        else:
            S_total[:,:,i] = S0 + S1*np.sin((delay_plate.phi_total + displacer_plate.phi_total)) - S2*np.cos((delay_plate.phi_total + displacer_plate.phi_total))

    tsh_image = np.sum(S_total, axis=2)

    return tsh_image

def simulate_field_widened(FLC_state):

    """
    Calculate intensity on camera for an FLC system, including a field widened delay plate (Two thin delay plates with half waveplate in between. Gives zero net delay, but gives shear delay
    across crystal and reduces higher order effects that manifest as curved interference fringes.
    :param FLC_state: 1 or 2
    :return:
    """

    #Large interpolation x and y
    x = np.linspace(-camera.sensor_size/2, camera.sensor_size/2, camera.n_pixels)
    y = np.linspace(-camera.sensor_size/2, camera.sensor_size/2, camera.n_pixels)

    S_total = np.zeros((len(x), len(y), len(msesim.wavelength)))


    for i in range(len(msesim.wavelength)):
        #Make the delay and displacer plate to find the phase shift for a given wavelength

        delay_plate_1 = Crystal(wavelength=msesim.wavelength[i]*10**-10, thickness=0.0075, cut_angle=0., name='alpha_bbo', nx=len(x), ny=len(y), pixel_size=camera.pixel_size, orientation=0., two_dimensional=True)
        delay_plate_2 = Crystal(wavelength=msesim.wavelength[i]*10**-10, thickness=0.0075, cut_angle=0., name='alpha_bbo', nx=len(x), ny=len(y), pixel_size=camera.pixel_size, orientation=90., two_dimensional=True)
        displacer_plate = Crystal(wavelength=msesim.wavelength[i]*10**-10, thickness=0.003, cut_angle=45., name='alpha_bbo', nx=len(x), ny=len(y), pixel_size=camera.pixel_size, orientation=90., two_dimensional=True)

        #Interpolate our small msesim grid to the real image size, 1024x1024

        S0_interp = interp2d(msesim.pixel_x, msesim.pixel_y, msesim.S0[:, :, i], kind='quintic')
        S0 = S0_interp(x,y)

        S1_interp = interp2d(msesim.pixel_x, msesim.pixel_y, msesim.S1[:, :, i], kind='quintic')
        S1 = S1_interp(x,y)

        S2_interp = interp2d(msesim.pixel_x, msesim.pixel_y, msesim.S2[:, :, i], kind='quintic')
        S2 = S2_interp(x,y)

        #Calculate the total intensity for a given wavelength, propagating stokes components through a delay plate and a displacer plate.

        if FLC_state == 1:
            S_total[:,:,i] = S0 + S1*np.sin((delay_plate_1.phi_total + delay_plate_2.phi_total + displacer_plate.phi_total)) - S2*np.cos((delay_plate_1.phi_total + delay_plate_2.phi_total + displacer_plate.phi_total))

        else:
            S_total[:,:,i] = S0 - S1*np.sin((delay_plate_1.phi_total + delay_plate_2.phi_total + displacer_plate.phi_total)) + S2*np.cos((delay_plate_1.phi_total + delay_plate_2.phi_total + displacer_plate.phi_total))

    # Field widened image intensity as a function of wavelength - sum up over the wavelength.

    tsh_fw_image = np.sum(S_total, axis=2)

    return tsh_fw_image

def simulate_ASH(circular):
    """
    Calculate the intensity of the image for an amplitude spatial heterodyne system
    :param circular: True/False - Include S3 component to realistically model effect of S3 on carrier amplitudes.
    :return:
    """

    #Large interpolation x and y
    x = np.linspace(-camera.sensor_size/2, camera.sensor_size/2, camera.n_pixels)
    y = np.linspace(-camera.sensor_size/2, camera.sensor_size/2, camera.n_pixels)

    S_total = np.zeros((len(x), len(y), len(msesim.wavelength)))


    for i in range(len(msesim.wavelength)):
        savart_1 = Crystal(wavelength=msesim.wavelength[i]*10**-10, thickness=0.00210, cut_angle=45., name='alpha_bbo', nx=len(x), ny=len(y),
                           pixel_size=camera.pixel_size, orientation=45., two_dimensional=True)

        savart_2 = Crystal(wavelength=msesim.wavelength[i]*10**-10, thickness=0.00210, cut_angle=45., name='alpha_bbo', nx=len(x), ny=len(y),
                           pixel_size=camera.pixel_size, orientation=135., two_dimensional=True)

        delay_plate = Crystal(wavelength=msesim.wavelength[i]*10**-10, thickness=0.015, cut_angle=0., name='alpha_bbo', nx=len(x),
                              ny=len(y),
                              pixel_size=camera.pixel_size, orientation=90., two_dimensional=True)

        displacer_plate = Crystal(wavelength=msesim.wavelength[i]*10**-10, thickness=0.003, cut_angle=45., name='alpha_bbo', nx=len(x),
                                  ny=len(y), pixel_size=camera.pixel_size, orientation=90., two_dimensional=True)

        #Interpolate our small msesim grid to the real image size, 1024x1024

        S0_interp = interp2d(msesim.pixel_x, msesim.pixel_y, msesim.S0[:, :, i], kind='quintic')
        S0 = S0_interp(x,y)

        S1_interp = interp2d(msesim.pixel_x, msesim.pixel_y, msesim.S1[:, :, i], kind='quintic')
        S1 = S1_interp(x,y)

        S2_interp = interp2d(msesim.pixel_x, msesim.pixel_y, msesim.S2[:, :, i], kind='quintic')
        S2 = S2_interp(x,y)

        if circular == True:
            S3_interp = interp2d(msesim.pixel_x, msesim.pixel_y, msesim.S3[:, :, i], kind='quintic')
            S3 = S3_interp(x, y)

            S_total[:, :, i] = 2 * S0 + 2 * S2 * np.cos(delay_plate.phi + displacer_plate.phi) + S1 * (
                    np.cos(displacer_plate.phi + delay_plate.phi + savart_1.phi - savart_2.phi) - np.cos(
                delay_plate.phi + displacer_plate.phi - savart_1.phi + savart_2.phi)) - S3 * (np.sin(
                displacer_plate.phi + delay_plate.phi + savart_1.phi - savart_2.phi) + np.sin(
                displacer_plate.phi + delay_plate.phi - savart_1.phi + savart_2.phi))
        else:
            S_total[:, :, i] = 2 * S0 + 2 * S2 * np.cos(delay_plate.phi + displacer_plate.phi) + S1 * (
                    np.cos(displacer_plate.phi + delay_plate.phi + savart_1.phi - savart_2.phi) - np.cos(
                delay_plate.phi + displacer_plate.phi - savart_1.phi + savart_2.phi))

    ash_image = np.sum(S_total, axis=2)

    return ash_image

def load_image(filename):
    image_file = pd.HDFStore(filename)
    image = image_file['/a']
    return image

def save_image(image, filename):

    #save image as a HDF file

    x = pd.HDFStore(filename)

    x.append("a", pd.DataFrame(image))
    x.close()
