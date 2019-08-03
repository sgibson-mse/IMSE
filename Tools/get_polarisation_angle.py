from Tools.Demodulate_TSSSH import load_image, demodulate_images
import matplotlib.pyplot as plt
import numpy as np
#filepaths to the imse images

filename1 = '/work/sgibson/MAST/IMSE/Tools/mastu_1MA_P4_f85mm_-k0.1_FLC_45.hdf'
filename2 = '/work/sgibson/MAST/IMSE/Tools/mastu_1MA_P4_f85mm_-k0.1_FLC_90.hdf'

#load the images

image_1 = load_image(filename1)
image_2 = load_image(filename2)

plt.figure()
plt.imshow(image_2)
plt.colorbar()
plt.show()

#get the polarisation angle from demodulating images

polarisation_angle = demodulate_images(image_1, image_2)

print(np.min(polarisation_angle*(180./np.pi)), np.max(polarisation_angle*180./np.pi))

plt.figure()
plt.imshow(polarisation_angle*180./np.pi)
plt.colorbar()
plt.clim(-20,0)
plt.show()