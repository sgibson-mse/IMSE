
from Model.read_fullenergy import ChannelFull, CentralPointsFull, GridPointsFull, ResolutionFull, SpectralDataFull
from Model.Constants import Constants
import idlbridge as idl

import matplotlib.pyplot as plt
import numpy as np

#idl.execute("restore, '/home/sam/Desktop/msesim/runs/mast_imse_2d_f85mm_edge/output/data/density2e19_MAST_photron_2d.dat', /VERBOSE")
idl.execute("restore, '/home/sam/Desktop/msesim/runs/mast_imse_photron/output/data/density2e19m3_MAST_photron.dat', /VERBOSE")

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

constants = Constants()
channel = ChannelFull()
central_points = CentralPointsFull()
grid_points = GridPointsFull()
resolution = ResolutionFull()
spectral_data = SpectralDataFull()

#Get the output from one pixel

stokes = spectral_data.data['total_stokes']
pi_stokes = spectral_data.data['pi_stokes']
sigma_stokes = spectral_data.data['sigma_stokes']
wavelength_vector = spectral_data.data['wavelength_vector']
major_radius = resolution.data['resolution_vector(R)'][:,0]
wavelength, radius = np.meshgrid(wavelength_vector,major_radius)

print(major_radius[10])

plt.figure()
plt.plot(wavelength_vector/10, stokes[10,0,:]/(10**6), color='black', label='$S_{0}$')
plt.plot(wavelength_vector/10, stokes[10,1,:]/(10**6), '--', color='blue', label='$S_{1}$')
#plt.plot(wavelength_vector/10,stokes[10,2,:]/(10**6), label='$S_{2}$')
plt.plot(wavelength_vector/10, stokes[10,3,:]/(10**6), '-.', color='red', label='$S_{3}$')
plt.xlim(659.5,660.5)
plt.xlabel('Wavelength [nm]')
plt.ylabel('Intensity [10$^{6}$ph/s]')
plt.legend(loc=2, prop={'size': 20})
plt.show()