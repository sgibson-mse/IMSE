from Tools.Plotting.graph_format import plot_format, plot_save
from Model.Crystal import Crystal

import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np

cb = plot_format()

nx = 1024
ny = 1024
pixel_sized = 20*10**-6
wavelength = 660*10**-9
crystal_thickness = 15*10**-3
displacer_thickness = 3*10**-3
cut_angle = 0.
displacer_cut = 45.
optic_axis = 90. #(ie. the crystal optical axis is vertical)

#Everything must be given in meters, angles in degrees.

delay_plate = Crystal(wavelength=wavelength, thickness=crystal_thickness, cut_angle=cut_angle, name='alpha_bbo', nx=nx, ny=ny, pixel_size=pixel_sized, orientation=optic_axis, two_dimensional=False)
displacer_plate = Crystal(wavelength=wavelength, thickness=displacer_thickness, cut_angle=displacer_cut, name='alpha_bbo', nx=nx, ny=ny, pixel_size=pixel_sized, orientation=optic_axis, two_dimensional=False)

print(delay_plate.phi_hyperbolic)

# plt.figure()
# im = plt.imshow((delay_plate.phi_hyperbolic+displacer_plate.phi_hyperbolic))
# # im = plt.imshow(delay_plate.phi_shear)
# cbar = plt.colorbar(orientation='vertical',fraction=0.046, pad=0.04)
# cbar.set_label('$\phi_{\mathrm{hyperbolic}}$')
# tick_locator = ticker.MaxNLocator(nbins=3)
# cbar.locator = tick_locator
# cbar.update_ticks()
# plt.gca().invert_xaxis()
# plt.gca().invert_yaxis()
# plt.tight_layout()
# plt.show()

plt.figure(1)
im = plt.imshow(displacer_plate.phi_shear*displacer_plate.yy / (2 * np.pi), interpolation=None)
im2 = plt.contour(displacer_plate.phi_shear*displacer_plate.yy / (2 * np.pi), colors='white')
plt.clabel(im2, inline=1, fontsize=10, manual=True)
cbar = plt.colorbar(im, orientation='vertical')
cbar.set_label('$\phi_{\mathrm{S}}$ (Radians)', labelpad=4)
tick_locator = ticker.MaxNLocator(nbins=8)
cbar.locator = tick_locator
cbar.update_ticks()
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()


plt.figure(2)
im3 = plt.imshow(displacer_plate.phi_hyperbolic*((3 - np.cos(2*displacer_plate.cut_angle))*displacer_plate.xx**2 - ((3*np.cos(2*displacer_plate.cut_angle)-1)*displacer_plate.yy**2))*1000/(2*np.pi), interpolation=None)
im4 = plt.contour(displacer_plate.phi_hyperbolic*((3 - np.cos(2*displacer_plate.cut_angle))*displacer_plate.xx**2 - ((3*np.cos(2*displacer_plate.cut_angle)-1)*displacer_plate.yy**2))*1000/(2*np.pi), colors='white')
cbar = plt.colorbar(im3, orientation='vertical')
plt.clabel(im4, inline=1, fontsize=10, manual=True)
cbar.set_label('$\phi_{\mathrm{H}}$ \n (10$^{3}$ Radians)', labelpad=4)
tick_locator = ticker.MaxNLocator(nbins=8)
cbar.locator = tick_locator
cbar.update_ticks()
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.show()