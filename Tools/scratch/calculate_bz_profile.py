import idlbridge as idl
import pyuda
import numpy as np
from scipy.interpolate import interp2d, interp1d
from numpy.linalg import norm

import matplotlib.pyplot as plt
# matplotlib.use('Qt4Agg',warn=False, force=True)
# from matplotlib import pyplot as plt
# print("Switched to:",matplotlib.get_backend())

from Tools.MSESIM import MSESIM
from Model.Observer import Camera
# from Model.SyntheticImage import simulate_TSH, simulate_field_widened, simulate_ASH, load_image, save_image
from Tools.demodulate_TSH_synthetic_image import demodulate_nfw_images, load_image

# Instantiate the msesim class, create plotting grid
idl.execute(
    "restore, '/home/sgibson/PycharmProjects/msesim/runs/imse_2d_32x32_f80mm_opticaxis/output/data/MAST_18501_imse.dat', /VERBOSE")
msesim = MSESIM(nx=32, ny=32)
data = msesim.load_msesim_spectrum()

def find_nearest(array, value):
    if value < array.min() or value > array.max():
        raise IndexError(
            "Requested value is outside the range of the data. The range of the data is from {}m to {}m".format(
                array.min(), array.max()))

    index = np.searchsorted(array, value, side="left")

    print(index)

    if (value - array[index]) ** 2 < (value - array[index + 1]) ** 2:
        return index
    else:
        return index + 1

def interp_msesim_gamma():
    # Load input from msesim
    stokes = msesim.data["total_stokes"]

    print(stokes.shape)

    msesim_S2 = np.sum(stokes[:, 2, :], axis=1)
    msesim_S1 = np.sum(stokes[:, 1, :], axis=1)

    msesim_gamma = 0.5 * np.arctan(msesim_S2 / msesim_S1)
    msesim_gamma = msesim_gamma.reshape(32, 32)

    x = np.linspace(-10.24,10.24,1024)
    y = np.linspace(-10.24,10.24,1024)

    x_small = np.linspace(-10.24,10.24,32)
    y_small = np.linspace(-10.24,10.24,32)

    gamma_interp = interp2d(x_small, y_small, msesim_gamma, kind='linear')

    msesim_gamma_new = gamma_interp(x,y)

    r_big = np.linspace(np.min(msesim.major_radius), np.max(msesim.major_radius), 1024)
    z_big = z = np.linspace(np.min(msesim.z), np.max(msesim.z), 1024)

    return msesim_gamma_new, r_big, z_big

def get_a_coefficients(r_big):
    client = pyuda.Client()
    a_coeffs = client.get('AMS_ACOEFF', 24409)
    radius_data = client.get('AMS_RPOS', 24409)

    radius = radius_data.data[0,:]
    A_0 = a_coeffs.data[0,0,:]
    A_1 = a_coeffs.data[0,1,:]
    A_2 = a_coeffs.data[0,2,:]
    A_3 = a_coeffs.data[0,3,:]
    A_4 = a_coeffs.data[0,4,:]
    A_5 = a_coeffs.data[0,5,:]

    A0_interp = interp1d(radius, A_0)
    A0 = A0_interp(r_big)

    A1_interp = interp1d(radius, A_1)
    A1 = A1_interp(r_big)

    A2_interp = interp1d(radius, A_2)
    A2 = A2_interp(r_big)

    A3_interp = interp1d(radius, A_3)
    A3 = A3_interp(r_big)
    
    A4_interp = interp1d(radius, A_4)
    A4 = A0_interp(r_big)

    A5_interp = interp1d(radius, A_5)
    A5 = A5_interp(r_big)

    return A0, A1, A2, A3, A4, A5

def efit_polarisation_angle(A0, A1, A2, A3, A4, A5, r_big, z_big):
    br = msesim.bfld[:,:,0]
    bz = msesim.bfld[:,:,1]
    bphi = msesim.bfld[:,:,2]

    # interp Br
    br_interp_func = interp2d(msesim.efitR, msesim.efitZ, br)
    interp_br = br_interp_func(r_big, z_big)

    # #interpolate onto the large grid
    bphi_interp_func = interp2d(msesim.efitR, msesim.efitZ, bphi)
    interp_bphi = bphi_interp_func(r_big, z_big)

    # interp Bz
    bz_interp_func = interp2d(msesim.efitR, msesim.efitZ, bz)
    interp_bz = bz_interp_func(r_big, z_big)

    bz_midplane = interp_bz[512,:]
    br_midplane = interp_br[512,:]
    bphi_midplane = interp_bphi[512,:]

    A0 = np.ones((np.shape(interp_bz)))*A0
    A1 = np.ones((np.shape(interp_bz)))*A1
    A2 = np.ones((np.shape(interp_bz)))*A2
    A3 = np.ones((np.shape(interp_bz)))*A3
    A4 = np.ones((np.shape(interp_bz)))*A4
    A5 = np.ones((np.shape(interp_bz)))*A5

    tan_gamma = (A0*interp_bz + A1*interp_br + A2*interp_bphi) / (A3*interp_bz + A4*interp_br + A5*interp_bphi)
    efit_gamma = np.arctan(tan_gamma)

    return efit_gamma, interp_br, interp_bz, interp_bphi

def plot_polarisation_angles(r_big, efit_gamma, polarisation):
    plt.figure()
    plt.plot(r_big, polarisation[512,::-1]*(180./np.pi), label='MSE')
    plt.plot(r_big, efit_gamma[512,:] * (180. / np.pi), '--', label='EFIT')
    plt.xlabel('R (m)')
    plt.ylabel('Polarisation angle $\gamma$ (degrees)')
    plt.legend()
    plt.show()

def calculate_bz_mse(A5, A0, polarisation, interp_bphi):
    # #calculate bz from polarisation angle
    bz_mse = (A5/A0) * interp_bphi * np.tan(polarisation)
    return bz_mse


tsh_1 = load_image(filename='/home/sgibson/PycharmProjects/IMSE/Images/f80mm_opticaxis/TSH_1.hdf')
tsh_2 = load_image(filename='/home/sgibson/PycharmProjects/IMSE/Images/f80mm_opticaxis/TSH_2.hdf')
polarisation = demodulate_nfw_images(tsh_1, tsh_2)


r_big = np.linspace(np.min(msesim.major_radius), np.max(msesim.major_radius), 1024)
z_big = np.linspace(np.min(msesim.z), np.max(msesim.z), 1024)
rr,zz = np.meshgrid(r_big[::-1],z_big)

A0, A1, A2, A3, A4, A5 = get_a_coefficients(r_big)

msesim_gamma_new, r_big, z_big = interp_msesim_gamma()
efit_gamma, interp_br, interp_bz, interp_bphi = efit_polarisation_angle(A0, A1, A2, A3, A4, A5, r_big, z_big)
bz_mse = calculate_bz_mse(A5, A0, polarisation, interp_bphi)
efitrr, efitzz = np.meshgrid(r_big, z_big)

plt.figure()
CS = plt.pcolormesh(rr, zz, bz_mse, cmap='jet', rasterized=True)
cbar = plt.colorbar(label='Bz (T)')
plt.clim(-0.1,0.1)
# CS2 = plt.contour(efitrr, efitzz, interp_bz, linestyles='dashed', colors='white')
# plt.clabel(CS2, inline=1, fontsize=10)
# cbar.add_lines(CS2)
plt.xlabel('R(m)')
plt.ylabel('Z(m)')
plt.show()


# plt.figure()
# plt.plot(r_big[::-1], polarisation[512,:]*180./np.pi, label='Demodulated')
# plt.plot(r_big[::-1], msesim_gamma_new[512,:]*180./np.pi, '--', label='MSESIM')
# plt.plot(r_big, efit_gamma[512,:]*180./np.pi, '--', label='EFIT')
# plt.xlabel('R(m)')
# plt.ylabel('$\gamma$ (Deg)')
# plt.legend()
#
# rr,zz = np.meshgrid(r_big[::-1], z_big)
#
# efitrr, efitzz = np.meshgrid(r_big, z_big)
#
# plt.figure()
# plt.plot(r_big[::-1], bz_mse[512,:], label='Demodulated')
# plt.plot(r_big, interp_bz[512,:], '--', label='EFIT')
# plt.xlabel('R(m)')
# plt.ylabel('B$_{z}$ (T)')
# plt.legend()
#
# plt.figure()
# CS = plt.pcolormesh(rr, zz, bz_mse, cmap='jet')
# cbar = plt.colorbar(label='Bz (T)')
# plt.clim(-0.15,0.15)
# CS2 = plt.contour(efitrr, efitzz, interp_bz, cmap='jet', linestyles='dashed')
# plt.clabel(CS2, inline=1, fontsize=10)
# cbar.add_lines(CS2)
# plt.xlabel('R(m)')
# plt.ylabel('Z(m)')


# #calculate current density assume j ~ dbz/dr
#
grad_r = np.gradient(rr, axis=1)

grad_z = np.gradient(zz, axis=0)

dbz_dr = np.gradient(bz_mse, axis=0)/grad_r

dbz_dr_efit = np.gradient(interp_bz, axis=0)/grad_r

dbr_dz = np.gradient(interp_br, axis=0)/grad_z

j_phi = (2.5*dbr_dz - dbz_dr)/(1.256)

j_phi_efit = (dbr_dz-dbz_dr_efit)/(1.256)

levels=np.arange(np.min(j_phi), np.max(j_phi), 0.1)

plt.figure(2)
plt.pcolormesh(efitrr, efitzz, j_phi_efit, rasterized=True, shading='gourand')
plt.xlabel('R (m)')
plt.ylabel('Z (m)')
plt.ylim(-0.1,0.1)
plt.colorbar(label='$j_{\phi}$ MA/m$^{2}$')
plt.clim(0.65,0.85)
plt.show()


#
# #get database efit current density
#
# efit_database_jphi_data = client.get('EFM_J(R)', 18501)
# efit_database_time = client.get('EFM_TIME', 18501)
#
# efit_database_jphi = efit_database_jphi_data.data[57]/10**6 #in MA/m^2 18 for efk
#
# plt.figure(1)
# plt.plot(rr[512,:], bz[512,:], label='MSE')
# plt.plot(msesim.efitR, msesim.bfld[32,:,1], '--', label='Efit')
# plt.legend()
# plt.xlabel('R (m)')
# plt.ylabel('Bz (T)')
# plt.show()
#
# plt.figure(2)
# plt.pcolormesh(rr, zz, j_phi, rasterized=True, shading='gourand')
# plt.xlabel('R (m)')
# plt.ylabel('Z (m)')
# plt.ylim(-0.1,0.1)
# plt.colorbar(label='$j_{\phi}$ MA/m$^{2}$')
# plt.clim(0.65,0.85)
# plt.show()
#
# plt.figure(3)
# plt.plot(rr[512,:], j_phi[512,:], label='MSE')
# plt.plot(msesim.efitR, efit_database_jphi, '--', color='black', label='Efit')
# plt.legend()
# plt.xlabel('R (m)')
# plt.ylabel('$j_{\phi}$ (MA/m$^{2}$)')
# plt.show()
#
# #2d efit grid
#
# efit_rr, efit_zz = np.meshgrid(msesim.efitR, msesim.efitZ)

# plt.figure(4)
# c = plt.contourf(rr, zz, bz, cmap='jet', levels=levels, rasterized=True)
# plt.contour(efit_rr, efit_zz, bz_efit, colors='black', linestyles='--', levels=levels)
# plt.colorbar(c, label='Bz (T)')
# plt.xlabel('R (m)')
# plt.ylabel('Z (m)')
# #plt.xlim(0.9,1.4)
# #plt.ylim(-0.1,0.1)
# plt.show()

# levels=np.arange(np.min(polarisation*(180./np.pi)), np.max(polarisation*(180./np.pi)), 5)
#
# plt.figure(5)
# c = plt.contourf(rr, zz, polarisation*(180./np.pi), levels=levels, cmap='jet', rasterized=True)
# plt.contour(efit_rr, efit_zz, efit_gamma, levels=levels, colors='black', linestyles='--')
# plt.colorbar(c, label='Polarisation Angle (Degrees)')
# plt.xlabel('R (m)')
# plt.ylabel('Z (m)')
# plt.show()

# plt.figure()
# plt.subplot(211)
# plt.imshow(interp_bphi)
# plt.colorbar()
#
# plt.subplot(212)
# plt.imshow(b_phi)
# plt.colorbar()
#
# plt.show()

# plt.figure()
# c = plt.pcolormesh(rr, zz, polarisation*180./np.pi, rasterized=True)
# plt.colorbar(c, label='Polarisation angle $\gamma$ (Degrees)')
# plt.contour(rr, zz, polarisation*180./np.pi, colors='white', linestyle='dashed', levels=[0])
# plt.ylim(-0.1,0.1)
# plt.clim(-10,30)
# plt.xlabel('R (m)')
# plt.ylabel('Z (m)')
# # plt.show()

# plt.figure()
# plt.plot(rr[0,:], polarisation[512,:]*(180/np.pi))
# plt.show()


# TSH_1 = simulate_TSH(FLC_state=1)
# save_image(TSH_1, filename='/home/sgibson/PycharmProjects/IMSE/Images/f80mm_opticaxis/TSH_1.hdf')
# plt.figure()
# plt.pcolormesh(rr, zz, ash_circular.values, rasterized=True)
# plt.xticks(np.arange(0.84, 1.5, 0.2))
# plt.xlabel('R (m)')
# plt.ylabel('Z (m)')
# cbar = plt.colorbar()
# cbar.set_label('Intensity (photons/s)', rotation=90)
# plt.show()