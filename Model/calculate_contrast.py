#External imports
import numpy as np
import matplotlib.pyplot as plt

#Internal imports
from Model.bbo_model import AlphaBBO
from Model.read_output import Channel, CentralPoints, GridPoints, Resolution, SpectralData

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

def get_intensities(I_stokes, wavelength_vector):

    I_total = I_stokes[0,:] #Total intensity
    I_polarised = np.sqrt(I_stokes[1,:]**2 + I_stokes[2,:]**2) #total linear polarisation

    SN = I_polarised/np.sqrt(I_total)
    SN[np.isnan(SN)] = 0

    # plt.figure()
    # plt.plot(wavelength_vector, I_total, label='Total intensity')
    # plt.plot(wavelength_vector, I_polarised, '--', label='linear polarised')
    # #plt.plot(wavelength_vector, SN, '.', label='S/N')
    # plt.legend()
    # plt.show()

    return I_total, I_polarised, wavelength_vector, SN

def define_crystals(wavelength_vector, delay_thickness, displacer_thickness):

    delay = AlphaBBO(wavelength_vector, thickness=delay_thickness, cut_angle=0)
    displacer = AlphaBBO(wavelength_vector, thickness=displacer_thickness, cut_angle=45)

    return delay, displacer

def add_delay(I_total, I_stokes, delay, displacer):

    Intensity_displacer = np.exp(1j*(displacer.phi_0+delay.phi_0)) * (I_stokes[1,:]  + (I_stokes[2,:]/1j))

    contrast = abs(np.sum(Intensity_displacer))/np.sum(I_total)

    return contrast

pixels = channel.data['channels']

delay_thickness = np.arange(0,27000,100)
displacer_thickness = 3000

contrasts = np.zeros((len(pixels),len(delay_thickness), 1))
delay = []

for i in range(len(pixels)):
    I_stokes = stokes[i, :, :]
    I_total, I_polarised, wavelength_vector, SN = get_intensities(I_stokes, wavelength_vector)
    for l in range(len(delay_thickness)):
        delay, displacer = define_crystals(wavelength_vector, delay_thickness[l], displacer_thickness)
        contrast = add_delay(I_total, I_stokes, delay, displacer)
        contrasts[i,l,:] = contrast

pp,ll = np.meshgrid(delay_thickness,major_radius)

plt.figure()
plt.contourf(ll,pp/1000,contrasts[:,:,0])
plt.colorbar()
plt.xlabel('Major Radius (m)')
plt.ylabel('Delay plate thickness (mm)')
plt.show()








