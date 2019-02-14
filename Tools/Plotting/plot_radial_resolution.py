from IMSE.Tools.load_msesim import MSESIM
import matplotlib.pyplot as plt

from IMSE.Tools.Plotting.graph_format import plot_format

plot_format()
#
# filepath = '/work/sgibson/msesim/runs/imse_2d_32x32_f85mm/output/data/MAST_18501_imse.dat'
# msesim = MSESIM(filepath, dimension=2)
#
# r_res = msesim.radial_res.reshape(32,32,7)
# major_radii = r_res[16,:,0]
# radial_resolution = r_res[16,:,5]
#
# filepath_2 = '/work/sgibson/msesim/runs/imse_2d_32x32_f85mm_-k0.2/output/data/MAST_18501_imse.dat'
# msesim2 = MSESIM(filepath_2, dimension=2)
#utl
# r_res2 = msesim2.radial_res.reshape(32,32,7)
# major_radii2 = r_res2[16,:,0]
# radial_resolution2 = r_res2[16,:,5]

filepath_3 = '/work/sgibson/msesim/runs/imse_1d_1024/output/data/MAST_18501_imse.dat'

msesim3= MSESIM(filepath_3, dimension=1)

r_res3 = msesim3.radial_res
major_radii3 = r_res3[:,0]
radial_resolution3 = r_res3[:,5]

plt.figure()
plt.plot(major_radii3, radial_resolution3*100, 's', color='black', label='72.8mm')
plt.xlabel('Major Radius (m)')
plt.ylabel('Radial resolution (cm)')
plt.legend()
plt.show()
