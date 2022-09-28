import pandas as pd
from Tools.Demodulate_TSSSH import demodulate_images
import matplotlib.pyplot as plt
import numpy as np

filename1 = '/work/sgibson/MAST/IMSE/Images/Edge_Current/mast_24409_1.hdf'
filename2 = '/work/sgibson/MAST/IMSE/Images/Edge_Current/mast_24409_2.hdf'

#fname1 = '/work/sgibson/MAST/IMSE/Images/Edge_Current/mastu_fiesta1.hdf'
#fname2 = '/work/sgibson/MAST/IMSE/Images/Edge_Current/mastu_fiesta2.hdf'

fname1 = '/work/sgibson/MAST/IMSE/Tools/mastu_1MA_fiesta_1.hdf'
fname2 = '/work/sgibson/MAST/IMSE/Tools/mastu_1MA_fiesta_2.hdf'

image1 = pd.read_hdf(filename1, key='/a')
image2 = pd.read_hdf(filename2, key='/a')

fiesta1 = pd.read_hdf(fname1, key='/a')
fiesta2 = pd.read_hdf(fname2, key='/a')

polarisation_angle_ideal = demodulate_images(image1, image2)
polarisation_angle_nonideal = demodulate_images(fiesta1, fiesta2)

# plt.figure()
# plt.imshow(polarisation_angle_ideal[::-1]*180./np.pi)
# plt.colorbar()
# plt.show()

plt.figure()
plt.plot(polarisation_angle_ideal[512,:]*(180./np.pi), label='24409')
plt.plot(-1*polarisation_angle_nonideal[512,:]*180./np.pi, label='mastu')
plt.legend()
plt.show()

