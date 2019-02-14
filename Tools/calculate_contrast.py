import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from IMSE.Model.Light import Light
from IMSE.Model.Observer import Camera
from IMSE.Model.Optics import Lens
from IMSE.Model.Crystal import Crystal

def project_light(orientation):
    """
    :param lens: Lens object containing info about the focussing lens
    :param camera: Camera object, contains information about the camera sensor.
    :return:  Alpha (array): Angle light hits the sensor at, given that the rays exiting the crystal are non-axial
              Beta (Array): Azimuthal angle around the camera sensor
    """

    # Find out where the light will focus onto the sensor.
    # We need to do this first as the shift between ordinary/extraordinary rays exiting the crystal depends on
    # the incident angle when the optical axis is not parallel to the plate surface.

    alpha = np.arctan(np.sqrt((camera.xx)**2 + (camera.yy)**2)/(lens.focal_length))
    beta = np.arctan2(camera.yy,camera.xx)
    delta = beta - (orientation*np.pi/180.)

    return alpha, beta, delta

def calculate_contrast(delay_thickness, delay_cut_angle, delay_orientation, displacer_thickness, displacer_cut_angle, displacer_orientation, material_name):

    alpha, beta, delta = project_light(orientation=90)

    S_out = np.zeros((len(camera.x), len(light.wavelength)), dtype='complex')
    S0 = np.zeros((len(camera.x), len(light.wavelength)))

    for i, wavelength in enumerate(light.wavelength):

        S0_interp = interp1d(light.x, light.S0[16,:,i], kind='quadratic')
        S1_interp = interp1d(light.x, light.S1[16,:,i], kind='quadratic')
        S2_interp = interp1d(light.x, light.S2[16,:,i], kind='quadratic')

        S0[:,i] = S0_interp(camera.x)
        S1 = S1_interp(camera.x)
        S2 = S2_interp(camera.x)

        delay = Crystal(delay_thickness, delay_cut_angle, material_name, delay_orientation, wavelength, alpha[16,:], delta[16,:])
        displacer = Crystal(displacer_thickness, displacer_cut_angle, material_name, displacer_orientation, wavelength, alpha[16,:], delta[16,:])

        phi = delay.phi + displacer.phi

        S_out[:,i] =  np.exp(1j*(phi)) * (S1  + (S2/1j))

    contrast = abs(np.sum(S_out, axis=1))/np.sum(S0, axis=1)

    return contrast

filepath = '/work/sgibson/msesim/runs/imse_2d_32x32_f85mm/output/data/MAST_18501_imse.dat'

light = Light(filepath, dimension=2)

FLC = 90

lens_focal_length = 85*10**-3

material_name = 'alpha_bbo'

delay_L = 15*10**-3
delay_cut = 0.
delay_orientation = 90.

displacer_L = np.linspace(1*10**-3, 10*10**-3, 50)
displacer_cut = 45.
displacer_orientation = 90.

camera = Camera(name='photron-sa4')
lens = Lens(lens_focal_length)

contrast = np.zeros((len(camera.x), len(displacer_L)))

for i in range(len(displacer_L)):
    contrast[:,i] = calculate_contrast(delay_L, delay_cut, delay_orientation, displacer_L[i], displacer_cut, displacer_orientation, material_name)

plt.figure()
plt.pcolormesh(camera.x, displacer_L, contrast.T)
plt.colorbar()
plt.show()