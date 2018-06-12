
from Model.read_fullenergy import ChannelFull, CentralPointsFull, GridPointsFull, ResolutionFull, SpectralDataFull
from Model.Physics_Constants import Constants
import idlbridge as idl

import matplotlib.pyplot as plt
import numpy as np

idl.execute("restore, '/home/sam/Desktop/msesim/runs/mast_imse_2d_f85mm_edge/output/data/density2e19_MAST_photron_2d.dat', /VERBOSE")

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

plt.figure()
plt.plot(stokes[230,0,:], label='S0')
plt.plot(stokes[230,1,:], label='s1')
plt.plot(stokes[230,2,:], label='s2')
plt.plot(stokes[230,3,:], label='s3')
plt.legend()
plt.show()