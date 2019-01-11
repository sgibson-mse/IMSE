import numpy as np

from Tools.demodulate_TSH_synthetic_image import load_image, demodulate_nfw_images
from Tools.calculate_bz_profile import interp_msesim_gamma, get_a_coefficients, efit_polarisation_angle
from Model.scratch.Noise import digitize_image, demodulate_images
from Tools.Plotting.graph_format import plot_format

cb = plot_format()

def calculate_noisy_polarisation_profile():
    # Translate photons/s into photons/ms, reduce intensity by 60% for the filter.
    image1 = (tsh_1.values) / 1000
    image2 = (tsh_2.values) / 1000

    n_samples = 1
    poisson_gamma = np.zeros((n_samples, 1024, 1024))

    for i in range(n_samples):
        # noisy image
        digitized_image1 = digitize_image(image1)
        digitized_image2 = digitize_image(image2)
        poisson_gamma[i, :, :] = demodulate_images(digitized_image1, digitized_image2)

    gamma_ave = np.average(poisson_gamma, axis=0)

    return poisson_gamma, gamma_ave

msesim_gamma_new, r_big, z_big = interp_msesim_gamma()
A0, A1, A2, A3, A4, A5 = get_a_coefficients(r_big)
efit_gamma, interp_br, interp_bz, interp_bphi = efit_polarisation_angle(A0, A1, A2, A3, A4, A5, r_big, z_big)

#Load in the hdf images
tsh_1 = load_image(filename='/home/sgibson/PycharmProjects/IMSE/Images/f80mm_opticaxis/TSH_1.hdf')
tsh_2 = load_image(filename='/home/sgibson/PycharmProjects/IMSE/Images/f80mm_opticaxis/TSH_2.hdf')

#Demodulate the images
polarisation = demodulate_nfw_images(tsh_1, tsh_2)

poisson_gamma,gamma_ave = calculate_noisy_polarisation_profile()

r_invert = r_big[::-1]

# plt.figure(2)
# plt.plot(r_big[::-1], msesim_gamma_new[512,:]*(180./np.pi), '--', color='black', label='msesim')
# plt.plot(r_invert[::10], polarisation[512,::10]*(180./np.pi), '.', label='IMSE')
# plt.plot(r_big[::-1], gamma_ave[512,:]*(180./np.pi), alpha=0.7, label='Average $\gamma$')
# plt.legend()
# plt.xlabel('Major Radius (m)')
# plt.ylabel('Polarisation angle (degrees)')
# plt.show()

# bz_mse = calculate_bz_mse(A5, A0,polarisation, interp_bphi)



# plt.figure(1)
# plt.imshow(gamma_ave*(180./np.pi))
# plt.gca().invert_xaxis()
# plt.colorbar()
# plt.show()


# plt.subplot(312)
# plt.plot(r_big[::-1], np.average(abs((msesim_gamma_new[512, :-1] - gamma_ave[512, :]) * (180. / np.pi))), sharex=ax1)
# plt.ylabel('Average residual between msesim and noisy image')


# plt.figure(4)
# plt.plot(R[:-1], (gamma[512,:-1]-gamma_ave[512,:])*(180./np.pi))
# plt.xlabel('Major Radius (m)')
# plt.ylabel('Polarisation Angle residual (degrees)')
# plt.show()
#
# plt.figure(3)
# plt.imshow(electrons_in-electrons_out)
# cbar = plt.colorbar()
# cbar.set_label('Difference in no. electrons due to read noise')
# plt.show()

#plotting code

#plot_polarisation_angles(r_big, efit_gamma, polarisation)
# rr,zz = np.meshgrid(r_big, z_big)
#
# levels = np.arange(-0.18,0.08,0.02)
#
# plot_polarisation_angles(r_big, efit_gamma, polarisation)
#
# plt.figure()
# plt.subplot(211)
# cs=plt.contourf(rr, zz, bz_mse[:,::-1], levels=levels)
# cbar = plt.colorbar(cs)
# cbar.set_clim(vmin=-0.18, vmax=0.06)
# cs2=plt.contour(rr, zz, interp_bz, linestyles='dashed', colors='white', levels=cs.levels)
# cbar.add_lines(cs2)
#
# plt.subplot(212)
# plt.contourf(rr, zz, interp_bz, levels=levels)
# cbar2 = plt.colorbar()
# cbar2.set_clim(vmin=-0.18, vmax=0.06)
# plt.show()
#
# plt.figure()
# plt.plot(r_big, bz_mse[512,::-1], label='MSE')
# plt.plot(r_big, interp_bz[512,:], '--', label='EFIT')
# plt.xlabel('R (m)')
# plt.ylabel('Bz (T)')
# plt.show()
#


