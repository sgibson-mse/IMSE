from Tools.load_msesim import MSESIM
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


from Tools.Plotting.graph_format import plot_format

plot_format()

#filepath = '/work/sgibson/msesim/runs/jet_mse_beamchange/output/data/JET_87123_linesplitting.dat'

filepath='/work/sgibson/msesim/runs/imse_2d_MASTU_1MA_wedge/output/data/MASTU_1MA.dat'

msesim= MSESIM(filepath, dimension=2)

r_res = msesim.radial_res

major_radii = r_res[:,0]
radial_resolution = r_res[:,5]

print(major_radii)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(major_radii, radial_resolution*100, 'o', color='C0')
ax.set_xlabel('Major Radius (m)')
ax.set_ylabel('Radial Resolution (cm)')
plt.show()



