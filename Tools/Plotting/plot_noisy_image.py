import idlbridge as idl
import matplotlib.pyplot as plt
import numpy as np

from Model.Observer import Camera
from Model.Constants import Constants
from Model.scratch.Noise import digitize_image, demodulate_images, load_image, image_shot_noise

from Tools.Plotting.graph_format import plot_format
from Tools.MSESIM import MSESIM

nx, ny = 32, 32
idl.execute("restore, '/home/sgibson/PycharmProjects/msesim/runs/imse_2d_32x32_f80mm_opticaxis/output/data/MAST_18501_imse.dat', /VERBOSE")
msesim = MSESIM(nx, ny)
data = msesim.load_msesim_spectrum()
constants = Constants()
camera = Camera(photron=True)

cb = plot_format()

exposure_time = 1*10**-3
image1 = load_image(filename='/home/sgibson/PycharmProjects/IMSE/Images/f80mm_opticaxis/TSH_1.hdf')
image2 = load_image(filename='/home/sgibson/PycharmProjects/IMSE/Images/f80mm_opticaxis/TSH_2.hdf')

def get_ideal_gamma(image1, image2):

    ideal_gamma = demodulate_images(image1, image2)

    r_big = np.linspace(np.min(msesim.major_radius), np.max(msesim.major_radius), len(image1))
    z_big = np.linspace(np.min(msesim.z), np.max(msesim.z), len(image1))

    rr, zz = np.meshgrid(r_big[::-1],z_big)

    # plt.figure(1)
    # plt.pcolormesh(rr, zz, image1/np.max(image1), rasterized=True)
    # plt.ylim(-0.12,0.12)
    # cbar1 = plt.colorbar()
    # cbar1.set_label('Intensity (Arb. Units)')
    # plt.xlabel('R (m)')
    # plt.ylabel('Z (m)')
    # plt.show()
    #
    # plt.figure(2)
    # plt.pcolormesh(rr, zz, np.log10(abs(shift_45)), rasterized=True)
    # plt.ylim(-0.12,0.12)
    # plt.colorbar()
    # plt.clim(10,13)
    # plt.show()
    #
    # plt.figure(3)
    # plt.pcolormesh(rr, zz, ideal_gamma*(180./np.pi), rasterized=True)
    # plt.ylim(-0.12,0.12)
    # plt.xlabel('R (m)')
    # cbar = plt.colorbar()
    # plt.clim(-10,25)
    # cbar.set_label('Polarisation Angle $\gamma$ (Deg.)')

    return ideal_gamma


def get_noisy_gamma(image1, image2, exposure_time):

    digitized_image1 = digitize_image(image1*0.6*0.85*0.85*0.6*0.5, exposure_time)

    shot_noise1 = image_shot_noise(digitized_image1)

    digitized_image2 = digitize_image(image2*0.6*0.85*0.85*0.6*0.5, exposure_time)
    shot_noise2 = image_shot_noise(digitized_image2)

    noisy_gamma = demodulate_images(shot_noise1, shot_noise2)

    return noisy_gamma

ideal_gamma = get_ideal_gamma(image1, image2)
noisy_gammas = []

for i in range(100):
    noisy_gamma = get_noisy_gamma(image1, image2, exposure_time)
    noisy_gammas.append(noisy_gamma)

noisy_gammas = np.array(noisy_gammas)

ave_noisy_gamma = np.average(noisy_gammas, axis=0)*(180./np.pi)
std_noisy_gamma = np.std(noisy_gammas, axis=0)*(180./np.pi)

r_big = np.linspace(np.min(msesim.major_radius), np.max(msesim.major_radius), 1024)
z_big = np.linspace(np.min(msesim.z), np.max(msesim.z), 1024)
rr,zz = np.meshgrid(r_big[::-1],z_big)

plt.figure()
CS = plt.pcolormesh(rr, zz, ave_noisy_gamma, rasterized=True)
plt.xlabel('R (m)')
plt.ylabel('Z (m)')
cbar = plt.colorbar(CS)
cbar.set_label('Average $\gamma$ (Deg.)')
plt.show()


plt.figure()
plt.plot(r_big[::-1], ideal_gamma[512,:]*(180./np.pi), color='black', label='$\gamma_{\mathrm{ideal}}$')
plt.plot(r_big[::-1], ave_noisy_gamma[512,:], ':', color='black', label='$\gamma_{\mathrm{noise}}$')
plt.fill_between(r_big[::-1], ave_noisy_gamma[512,:]-1*std_noisy_gamma[512,:], ave_noisy_gamma[512,:]+std_noisy_gamma[512,:], color=cb[2], alpha=1, label='$\gamma_{\mathrm{noise}} \pm \sigma$')
plt.fill_between(r_big[::-1], ave_noisy_gamma[512,:]-1*std_noisy_gamma[512,:], ave_noisy_gamma[512,:]-2*std_noisy_gamma[512,:], color=cb[2], alpha=0.4,label='$\gamma_{\mathrm{noise}} \pm 2\sigma$')
plt.fill_between(r_big[::-1], ave_noisy_gamma[512,:]+1*std_noisy_gamma[512,:], ave_noisy_gamma[512,:]+2*std_noisy_gamma[512,:], color=cb[2], alpha=0.4)
plt.legend(prop={'size': 20})
plt.xlabel('R (m)')
plt.ylabel('Polarisation angle $\gamma$ (Deg.)')
plt.show()

# plt.figure()
# plt.imshow((ideal_gamma-noisy_gamma)*(180./np.pi))
# plt.colorbar()
# plt.clim(-2,2)
# plt.show()

def plot_noisy_gamma(noisy_gamma):
    r_big = np.linspace(np.min(msesim.major_radius), np.max(msesim.major_radius), len(image1))
    z_big = np.linspace(np.min(msesim.z), np.max(msesim.z), len(image1))

    rr, zz = np.meshgrid(r_big[::-1],z_big)

    plt.figure()
    plt.imshow(noisy_gamma*(180./np.pi))
    plt.colorbar()
    plt.show()

    #
    # plt.figure()
    # CS = plt.pcolormesh(rr, zz, shot_noise1, rasterized=True)
    # plt.xlabel('R (m)')
    # plt.ylabel('Z (m)')
    # cbar = plt.colorbar(CS)
    # cbar.set_label('Intensity [ADU]')
    # plt.show()
