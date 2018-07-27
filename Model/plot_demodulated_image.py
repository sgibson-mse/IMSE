from Model.demodulate_TSH_synthetic_image import demodulate_nfw_images, demodulate_fw_images, msesim_profiles


import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

gamma, R = msesim_profiles(npx=32)
image_1, image_2, polarisation = demodulate_nfw_images()

plt.figure(1)

plt.subplot(131)
plt.imshow(image_1)
plt.xlabel('x')
plt.ylabel('y')
plt.gca().invert_xaxis()
cbar1 = plt.colorbar(fraction=0.046, pad=0.04)
cbar1.set_label('FLC state 1, Intensity (ph/s)')

plt.subplot(132)
plt.imshow(image_2)
plt.xlabel('x')
plt.ylabel('y')
plt.gca().invert_xaxis()
cbar2 = plt.colorbar(fraction=0.046, pad=0.04)
cbar2.set_label('FLC state 2, Intensity (ph/s)')

plt.subplot(133)
plt.imshow(polarisation*(180./np.pi))
plt.xlabel('x')
plt.ylabel('y')
plt.gca().invert_xaxis()
cbar3 = plt.colorbar(fraction=0.046, pad=0.04)
cbar3.set_label('Polarisation angle $\gamma$')
plt.show()

fig,ax = plt.subplots()
ax.plot(R[:-1], polarisation[512,:]*(180./np.pi), label='demodulated')
ax.plot(R, gamma[512,:]*(180./np.pi), '--', color='red', label='msesim')
plt.legend()
plt.ylabel('Polarisation angle $\gamma$')
plt.xlabel('Major radius (m)')

axins = zoomed_inset_axes(ax, 5, loc=4)
axins.plot(R[:-1], polarisation[512,:]*(180./np.pi), label='demodulated')
axins.plot(R, gamma[512,:]*(180./np.pi), '--', color='red', label='msesim')
x1, x2, y1, y2 = 0.93, 0.97, -2, 2 # specify the limits
axins.set_xlim(x1, x2) # apply the x-limits
axins.set_ylim(y1, y2) # apply the y-limits
plt.yticks(visible=False)
plt.xticks(visible=False)
mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")
plt.show()

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
