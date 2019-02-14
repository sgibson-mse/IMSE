from IMSE.Tools.Demodulate_TSSSH import demodulate_images, load_image
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import numpy as np
from IMSE.Tools.load_msesim import MSESIM
from scipy.io import readsav
import pyuda
from scipy.interpolate import interp1d, interp2d

client = pyuda.Client()

#filepaths to the imse images

filename1 = '/work/sgibson/MAST/IMSE/Tools/mastu_fiesta1.hdf'
filename2 = '/work/sgibson/MAST/IMSE/Tools/mastu_fiesta2.hdf'

#load the images

image_1 = load_image(filename1)
image_2 = load_image(filename2)

#get the polarisation angle from demodulating images

polarisation_angle = demodulate_images(image_1, image_2)

#load the msesim run

filepath = '/work/sgibson/msesim/runs/imse_2d_32x32_MASTU_edgecurrent/output/data/MASTU_edgecurrent.dat'
msesim = MSESIM(filepath=filepath, dimension=2)

#load the equilibrium used in msesim run
eq = readsav('/work/sgibson/msesim/equi/MASTU_equilibrium.sav')

#get r,z values
r = msesim.major_radius
z = msesim.central_coordinates[:,:,2]

#interpolate onto the size of the camera sensor
new_r = np.linspace(np.min(r), np.max(r), 1024)
new_z = np.linspace(np.min(z), np.max(z), 1024)

rr,zz = np.meshgrid(new_r, new_z)

rplt = np.linspace(np.min(r), np.max(r), 1024)[::-1]
zplt = np.linspace(np.min(z), np.max(z), 1024)

rrplt, zzplt = np.meshgrid(rplt, zplt)

#plot the polarisation angle

# levels=np.arange(0,1.1,0.1)

# plt.figure()
# plt.pcolormesh(rrplt, zzplt, -1*polarisation_angle*(180./np.pi), cmap='inferno')
# ax = plt.colorbar()
# ax.set_label('Polarisation Angle $\gamma$ (Degrees)', fontsize=24)
# plt.clim(0,40)
# ax2 = plt.contour(eq['r'],eq['z'],eq['fluxcoord'], linestyles='dashed', colors='black', levels=levels)
# plt.ylim(-0.14,0.14)
# plt.xlim(np.min(new_r), np.max(new_r))
# plt.clabel(ax2, inline=1, fontsize=13, manual=True)
# plt.xlabel('Major Radius R (m)')
# plt.ylabel('Z (m)')
# plt.show()

levels = np.arange(0,1.2,0.2)

#calculate bz

#get the A coefficients from the database from a previous calibration

acoeffs = client.get('AMS_ACOEFF', 24409).data
rpos = client.get('AMS_RPOS', 24409).data[0,:]

ams_a0 = acoeffs[0,0,:]
ams_a1 = acoeffs[0,1,:]
ams_a2 = acoeffs[0,2,:]
ams_a3 = acoeffs[0,3,:]
ams_a4 = acoeffs[0,4,:]
ams_a5 = acoeffs[0,5,:]

#interpolate onto the larger r grid

a0_interp = interp1d(rpos, ams_a0)
a2_interp = interp1d(rpos, ams_a2)
a3_interp = interp1d(rpos, ams_a3)
a5_interp = interp1d(rpos, ams_a5)

a0n = a0_interp(new_r)
a2n = a2_interp(new_r)
a3n = a3_interp(new_r)
a5n = a5_interp(new_r)

#assume the geometry coefficients are the same for each vertical height (probably not true, but we don't have a calibration in 2d yet)

a0 = np.repeat(a0n, 1024).reshape(1024,1024)
a2 = np.repeat(a2n, 1024).reshape(1024,1024)
a3 = np.repeat(a3n, 1024).reshape(1024,1024)
a5 = np.repeat(a5n, 1024).reshape(1024,1024)

#interp bfield
Bphi_interp = interp2d(eq['r'][0,:], eq['z'][:,0], eq['bfld'][:,:,2])
Bphi = Bphi_interp(new_r, new_z)

#calculate Bz from the polarisation angle and the toroidal field from the equilibrium - fix up a sign error from the equilibrium (plasma current was +ve, not -ve for this equilibrium)

Bz = (np.tan(-1*polarisation_angle)*a5 - a2)*Bphi / (a0 - np.tan(-1*polarisation_angle)*a3)

#plot the Bz

levels = np.arange(-0.6,0,0.1)
ticks = np.arange(1,1.4,0.1)

plt.figure()
ax = plt.pcolormesh(rrplt, zzplt, -1*Bz, cmap='viridis')
plt.axvline(x=1.34, color='red')
plt.xlim(1,1.36)
plt.xticks(ticks)
cbar = plt.colorbar(ax)
cbar.ax.set_ylabel('Bz (T)')
plt.clim(0,-0.6)
cs = plt.contour(eq['r'], eq['z'], eq['Bfld'][:,:,1], linestyle='dashed', colors='white', levels=levels)
plt.clabel(cs, manual=True, fontsize=16)
plt.xlabel('Major Radius (m)')
plt.ylabel('Z (m)')
plt.show()



