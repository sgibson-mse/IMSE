import numpy as np
import matplotlib.pyplot as plt

#internal imports
from Model.bbo_model import AlphaBBO

#Load the intensity I(x,y,lambda)
I0 = np.load('synthetic_image.npy')
S1 = np.load('S1.npy')
# S2 = np.load('S2.npy')
# wavelength = np.load('wavelength.npy')
# R = np.load('R.npy')

nx = 1024
ny= 1024
pixel_size = 20*10**-3

x = np.arange(-int(nx/2), int(nx/2)+1, 1)*pixel_size
y = np.arange(-int(ny/2), int(ny/2)+1, 1)*pixel_size

# Make the delay and displacer plates
delay_plate = AlphaBBO(wavelength=wavelength, thickness=15000, cut_angle=0, alpha=11.3)
displacer_plate = AlphaBBO(wavelength=wavelength, thickness=3000, cut_angle=45, alpha=11.3)

print(delay_plate.phi_0.shape)


# plt.figure()
# plt.imshow(np.sum(image, axis=2))
# plt.gca().invert_xaxis()
# plt.show()