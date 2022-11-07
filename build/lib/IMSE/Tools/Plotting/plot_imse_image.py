from Tools.Demodulate_TSSSH import demodulate_images, load_image
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import numpy as np
from Tools.load_msesim import MSESIM
from scipy.io import readsav
import pyuda
from scipy.interpolate import interp1d, interp2d

def get_geometry_coefficients():

    # get the A coefficients from the database from a previous calibration

    acoeffs = client.get('AMS_ACOEFF', 24409).data
    rpos = client.get('AMS_RPOS', 24409).data[0, :]

    ams_a0 = acoeffs[0, 0, :]
    ams_a1 = acoeffs[0, 1, :]
    ams_a2 = acoeffs[0, 2, :]
    ams_a3 = acoeffs[0, 3, :]
    ams_a4 = acoeffs[0, 4, :]
    ams_a5 = acoeffs[0, 5, :]

    # interpolate onto the larger r grid

    a0_interp = interp1d(rpos, ams_a0)
    a2_interp = interp1d(rpos, ams_a2)
    a3_interp = interp1d(rpos, ams_a3)
    a5_interp = interp1d(rpos, ams_a5)

    a0n = a0_interp(new_r)
    a2n = a2_interp(new_r)
    a3n = a3_interp(new_r)
    a5n = a5_interp(new_r)

    return a0n, a2n, a3n, a5n

def calculate_Bz(a0, a2, a3, a5, eq):

    # interp bfield
    Bphi = eq['bfld'][2, :, :]

    # calculate Bz from the polarisation angle and the toroidal field from the equilibrium - fix up a sign error from the equilibrium (plasma current was +ve, not -ve for this equilibrium)

    Bz = (np.tan(-1 * polarisation_angle) * a5 - a2) * Bphi / (a0 - np.tan(-1 * polarisation_angle) * a3)

    return Bz

def calculate_Br(r, z, eq):

    Br_eq = eq['bfld'][0,:,:]
    r_eq = eq['r']#[0,:]
    z_eq = eq['z']#[:,0]
    
    return Br

def calculate_current(Bz,Br):

    mu_0 = 1.256


    grad_r = np.gradient(r)
    grad_z = np.gradient(z)

    dBzdr = np.gradient(Bz, axis=0)/grad_r

    plt.figure()
    plt.imshow(dBzdr/1.256)
    plt.colorbar()
    plt.clim(-0.1,1)
    plt.show()

    dBrdz = np.gradient(Br, axis=1)/grad_z

    plt.figure()
    plt.imshow(dBzdr)
    plt.colorbar()
    plt.show()

    # plt.figure()
    # plt.imshow(dBrdz)
    # plt.colorbar()
    # plt.show()

    j_phi = (dBrdz - dBzdr)/mu_0

    plt.figure()
    plt.pcolormesh(rr[::-1], zz, j_phi, rasterized=True, shading='gourand')
    cbar = plt.colorbar(label='$j_{\phi}$ MA/m$^{2}$')
    # plt.clim(0.1, 0.75)
    plt.xlabel('R')
    plt.ylabel('Z')
    plt.show()

    return j_phi

def plot_2d_polarisation_angle(r, z, polarisation_angle, eq):

    rr, zz = np.meshgrid(new_r, new_z)

    rplt = np.linspace(np.min(r), np.max(r), 1024)[::-1]
    zplt = np.linspace(np.min(z), np.max(z), 1024)

    rrplt, zzplt = np.meshgrid(rplt, zplt)

    # plot the polarisation angle

    levels = np.arange(0, 1.1, 0.1)

    plt.figure()
    plt.pcolormesh(rrplt, zzplt, -1 * polarisation_angle * (180. / np.pi), cmap='inferno')
    ax = plt.colorbar()
    ax.set_label('Polarisation Angle $\gamma$ (Degrees)', fontsize=24)
    plt.clim(15,28)
    ax2 = plt.contour(eq['r'], eq['z'], eq['fluxcoord'].T, linestyles='dashed', colors='black', levels=levels)
    plt.ylim(-0.14, 0.14)
    plt.xlim(1.25,1.45)
    plt.xlim(np.min(new_r), np.max(new_r))
    plt.clabel(ax2, inline=1, fontsize=13, manual=True)
    plt.xlabel('Major Radius R (m)')
    plt.ylabel('Z (m)')
    plt.show()

    return

def plot_Bz(r,z,Bz,eq):

    rr, zz = np.meshgrid(new_r, new_z)

    rplt = np.linspace(np.min(r), np.max(r), 1024)[::-1]
    zplt = np.linspace(np.min(z), np.max(z), 1024)

    rrplt, zzplt = np.meshgrid(rplt, zplt)

    levels = np.arange(0,1.2,0.2)

    #plot the Bz

    levels = np.arange(-0.6,0,0.1)
    ticks = np.arange(1,1.4,0.1)

    plt.figure()
    ax = plt.pcolormesh(rrplt, zzplt, -1*Bz)
    plt.axvline(x=1.34, color='red')
    plt.xlim(1.25,1.36)
    plt.xticks(ticks)
    cbar = plt.colorbar(ax)
    cbar.ax.set_ylabel('Bz (T)')
    cbar.clim(-0.4,0.1)
    cs = plt.contour(eq['r'], eq['z'], eq['Bfld'][:,:,1], linestyle='dashed', colors='white', levels=levels)
    plt.clabel(cs, manual=True, fontsize=16)
    plt.xlabel('Major Radius (m)')
    plt.ylabel('Z (m)')
    plt.show()

    return

client = pyuda.Client()

#filepaths to the imse images

# filename1 = '/work/sgibson/MAST/IMSE/Images/Edge_Current/mastu_fiesta1.hdf'
# filename2 = '/work/sgibson/MAST/IMSE/Images/Edge_Current/mastu_fiesta2.hdf'

filename1 = '/work/sgibson/MAST/IMSE/Tools/mastu_1MA_P4_f85mm_-k0.1_FLC_45.hdf'
filename2 = '/work/sgibson/MAST/IMSE/Tools/mastu_1MA_P4_f85mm_-k0.1_FLC_90.hdf'

#load the images

image_1 = load_image(filename1)
image_2 = load_image(filename2)

#get the polarisation angle from demodulating images

polarisation_angle = demodulate_images(image_1, image_2)

#load the msesim run

#filepath = '/work/sgibson/msesim/runs/imse_2d_32x32_MASTU_edgecurrent/output/data/MASTU_edgecurrent.dat'
filepath = '/work/sgibson/msesim/runs/imse_2d_MASTU_1MA_wedge/output/data/MASTU_1MA.dat'
msesim = MSESIM(filepath=filepath, dimension=2)

#load the equilibrium used in msesim run
eq = readsav('/work/sgibson/msesim/equi/equi_MASTU_1MA_P4_CATIA.sav')

#get r,z values
r = msesim.major_radius
z = msesim.central_coordinates[:,:,2]

#interpolate onto the size of the camera sensor
new_r = np.linspace(np.min(r), np.max(r), 1024)
new_z = np.linspace(np.min(z), np.max(z), 1024)

#get geometry coefficients to calculate Bz
# a0, a2, a3, a5 = get_geometry_coefficients()
# Bz = calculate_Bz(a0, a2, a3, a5, eq)
# Br = calculate_Br(new_r, new_z, eq)
# calculate_current(new_r, new_z,Bz,Br)

#calculate current density directly from data

plot_2d_polarisation_angle(r, z, polarisation_angle, eq)
# plot_Bz(r,z,Bz,eq)

