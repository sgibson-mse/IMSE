from IMSE.Tools.Demodulate_TSSSH import load_image, demodulate_images
import matplotlib.pyplot as plt
import numpy as np
#filepaths to the imse images

filename1 = '/work/sgibson/MAST/IMSE/Tools/mastu_nodivergence1.hdf'
filename2 = '/work/sgibson/MAST/IMSE/Tools/mastu_nodivergence2.hdf'


#load the images

image_1 = load_image(filename1)
image_2 = load_image(filename2)

#get the polarisation angle from demodulating images

polarisation_angle = demodulate_images(image_1, image_2)

plt.figure()
plt.plot(polarisation_angle*180./np.pi)
plt.colorbar()
plt.show()