from Tools.demodulate_TSH_synthetic_image import demodulate_nfw_images, load_image, msesim_profiles


import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# gamma, R = msesim_profiles(npx=32)
image_1 = load_image(filename='/home/sgibson/PycharmProjects/IMSE/Model/edgecurrent24409_1.hdf')
image_2 = load_image(filename='/home/sgibson/PycharmProjects/IMSE/Model/edgecurrent24409_2.hdf')
polarisation = demodulate_nfw_images(image_1, image_2)

plt.figure(1)

plt.subplot(121)
plt.imshow(image_1)
plt.xlabel('x')
plt.ylabel('y')
plt.gca().invert_xaxis()
cbar1 = plt.colorbar(fraction=0.046, pad=0.04)
cbar1.set_label('FLC state 1, Intensity (ph/s)')

plt.subplot(122)
plt.imshow(image_2)
plt.xlabel('x')
plt.ylabel('y')
plt.gca().invert_xaxis()
cbar2 = plt.colorbar(fraction=0.046, pad=0.04)
cbar2.set_label('FLC state 2, Intensity (ph/s)')

plt.show()

print(np.min(polarisation*(180./np.pi)), np.max(polarisation*(180./np.pi)))

x = np.linspace(-10.24,10.24,1025)
y = np.linspace(-10.24,10.24,1025)

xx, yy = np.meshgrid(x,y)

#levels = np.arange(-30,40,1)
levels = np.arange(20,35,1)

plt.figure(2)
plt.subplot(111)
CS = plt.contourf(xx, yy, polarisation*(180./np.pi), cmap='inferno', levels=levels)
#CS2 = plt.contour(CS, levels=levels,colors='black')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
cbar3 = plt.colorbar(fraction=0.046, pad=0.04)
cbar3.set_label('Polarisation angle $\gamma$')
plt.show()

plt.figure()
plt.plot(polarisation[512,:]*(180./np.pi), label='demodulated')
plt.legend()
plt.ylabel('Polarisation angle $\gamma$')
plt.xlabel('Major radius (m)')
plt.show()
#
# axins = zoomed_inset_axes(ax, 5, loc=4)
# axins.plot(R[:-1], polarisation[512,:]*(180./np.pi), label='demodulated')
# axins.plot(R, gamma[512,:]*(180./np.pi), '--', color='red', label='msesim')
# x1, x2, y1, y2 = 0.93, 0.97, -2, 2 # specify the limits
# axins.set_xlim(x1, x2) # apply the x-limits
# axins.set_ylim(y1, y2) # apply the y-limits
# plt.yticks(visible=False)
# plt.xticks(visible=False)
# mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")
# plt.show()

# image_1_fw, image_2_fw, polarisation_fw = demodulate_fw_images()
# plt.subplot(212)
# plt.imshow(polarisation_fw*(180./np.pi))
# plt.gca().invert_yaxis()
# plt.gca().invert_xaxis()
# plt.colorbar()
# plt.show()
#
# plt.figure()
# plt.plot(R[int(len(image_1)/2),:], -1*polarisation[int(len(image_1)/2),:]*(180./np.pi), label='demodulated, no field widening')
# plt.plot(R[int(len(image_1_fw)/2),:], -1*polarisation_fw[int(len(image_1_fw)/2),:]*(180./np.pi), label='demodulated, field widening')
# plt.plot(R[int(len(image_1)/2),:], gamma[int(len(image_1)/2),:]*(180./np.pi), '--', label='msesim output')
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(image_2.iloc[500,:].values, color='black', label='No field widening')
# plt.plot(image_1.iloc[500,:].values, color='red', label='Field widening')
# plt.xlabel('X pixel')
# plt.ylabel('Intensity [ph/s]')
# plt.legend(loc=1, prop={'size': 20})
# plt.show()

#
# plt.figure()
# plt.imshow(contrast_45)
# plt.gca().invert_xaxis()
# plt.colorbar()
# plt.clim(0.1,0.5)
# plt.show()
