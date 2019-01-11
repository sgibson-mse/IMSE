import idlbridge as idl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.io import readsav
import pyuda

from Model.Observer import Camera
from Tools.Plotting.graph_format import plot_format

cb = plot_format()

camera = Camera(photron=True)
client = pyuda.Client()

class MSESIM(object):

    def __init__(self, nx, ny):

        self.nx = nx
        self.ny = ny

        #load in commonly used parameters from the msesim file... everything else is stored in the data array.
        self.object_names = ('central_coordinates', 'beam_velocity_vector', 'beam_duct_coordinates', 'half_beam_sampling_width', 'beam_axis_vector',
                            'collection_lens_coordinates', 'optical_axis', 'emission_vector', 'bfield_vector', 'efield_vector',
                            'doppler_shift', 'max_stark_shift', 'polarisation_angles', 'psi_normalised',

                            'grid_coordinates', 'beam_velocity_vector', 'emission_vector',
                            'bfield_vector', 'efield_vector', 'polarisation_angles',
                            'psi_normalised', 'emission_intensity',

                            'R', 'Z', 'psi', 'psi(R,Z)', 'magnetic_axis',
                            'emission_intensity(RZ)', 'emission_intensity(R)', 'emission_intensity(psi)',
                            'resolution_vector(psi)', 'resolution_vector(R)',

                            'wavelength_vector', 'pi_stokes', 'sigma_stokes', 'total_stokes',
                            'cwl_stokes', 'optimal_sigma_wavelength_stokes', 'optimal_blueshift_pi_wavelength',
                            'optimal_redshift_pi_wavelength')

        self.data = self.load_msesim_spectrum()
        self.parameters()
        self.bfield_parameters()

    @staticmethod
    def _find_nearest(array, value):

        if value < array.min() or value > array.max():
            raise IndexError("Requested value is outside the range of the data. The range of the data is from {}m to {}m".format(array.min(),array.max()))

        index = np.searchsorted(array, value, side="left")

        print(index)

        if (value - array[index])**2 < (value - array[index + 1])**2:
            return index
        else:
            return index + 1

    def load_msesim_spectrum(self):

        """
        Stores the output of an msesim run (stored in a .dat file) in a dictionary for use in python.
        :return: Dictionary of outputs from msesim. Retrievable using the object_names as given below.
        """

        self.data = {}

        key_names = ("xyz0", "B_v0", "B_xyz", "B_w", "B_vec",
                     "C_xyz", "C_k", "vec0", "Bfld0", "Efld0",
                     "Dshift0", "Sshift0", "alpha0", "psi0",

                     "gp_xyz", "gp_vel", "gp_vec", "gp_bfld",
                     "gp_efld", "gp_alpha", "gp_psi", "gp_emis",

                     "R", "Z", "psi", "RZpsi", "Rm",
                     "RZ_emis", "R_emis", "psi_emis", "psi_res",
                     "R_res",

                     "lambda", "pstokes", "sstokes", "stokes",
                     "cwlstokes", "sigstokes", "pibstokes", "pirstokes")

        for key_name, object_name in zip(key_names, self.object_names):
            self.data[object_name] = idl.get(key_name)

        return self.data

    def parameters(self):

        self.stokes_total = np.array(self.data["total_stokes"])
        self.wavelength = np.array(self.data["wavelength_vector"])

        #Reshape array into 2D grid
        self.S0 = self.stokes_total[:,0,:].reshape((self.nx,self.ny,len(self.wavelength)))
        self.S1 = self.stokes_total[:,1,:].reshape((self.nx,self.ny,len(self.wavelength)))
        self.S2 = self.stokes_total[:,2,:].reshape((self.nx,self.ny,len(self.wavelength)))
        self.S3 = self.stokes_total[:,3,:].reshape((self.nx,self.ny,len(self.wavelength)))

        # Major radius values where the lines of sight are
        self.major_radius = np.array(self.data["resolution_vector(R)"])[:,0].reshape(32,32)
        self.major_radius = self.major_radius[0,:]

        #Pixel co-ordinates in meters
        self.pixel_x = np.linspace(-1*camera.sensor_size/2, camera.sensor_size/2, self.nx)
        self.pixel_y = np.linspace(-1*camera.sensor_size/2, camera.sensor_size/2, self.ny)

        #Polarisation fractions - total polarised fraction, total unpolarised, total circularly polarised and linearly polarised fractions
        self.polarised_fraction = np.sqrt(self.S1**2 + self.S2**2+self.S3**2)/self.S0
        self.total_unpolarised = (self.S0 - np.sqrt(self.S1**2 + self.S2**2 + self.S3**2))/self.S0
        self.total_circular = np.sqrt(self.S3**2 - (self.total_unpolarised**2*self.S0))/self.S0

        self.LPF = np.sqrt(self.S1**2 + self.S2**2)/self.S0
        self.CPF = self.total_circular/self.S0
        self.circular_total = np.sqrt(self.S3**2)/self.S0

        #polarisation angle as calculated by MSESIM
        self.gamma = 0.5*np.arctan2(self.S2, self.S1)*(180./np.pi)

        self.xyz0 = self.data['central_coordinates'].reshape(32,32,3)
        self.z = self.xyz0[:,:,2]
        self.z = self.z[:,0]


    def plot_intensity(self):

        xyz0 = self.data['central_coordinates'].reshape(32,32,3)
        z = xyz0[:,:,2]
        r = self.data['resolution_vector(R)'].reshape(32,32,7)[:,:,0]

        z = z[:,0]
        r = r[0,:]

        rr, zz = np.meshgrid(r,z)
        emission = np.sum(self.data['emission_intensity(R)'].reshape(32,32,400),axis=2)

        plt.figure()
        plt.pcolormesh(rr, zz, emission, shading='gouraud')
        cbar = plt.colorbar()
        cbar.set_label('Emission intensity (photons/s)', rotation=90)
        plt.xlabel('R (m)')
        plt.ylabel('Z (m)')
        plt.show()

        return

    def plot_spectrum(self, radius):

        # Find the radii for the spectrum you want to plot
        idx = self._find_nearest(self.major_radius[::-1], value=radius)

        fig= plt.figure(1)
        plt.plot(self.wavelength/10, self.S0[idx,idx,:].T, color='black', label='$I_{\mathrm{total}}$')
        plt.plot(self.wavelength/10, np.sqrt(self.S2[idx,idx,:].T**2 + self.S1[idx,idx,:].T**2), label='$I_{\mathrm{linear}}$')
        plt.plot(self.wavelength/10, self.S3[idx,idx,:].T, label='$I_{\mathrm{circular}}$')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useOffset=False, useMathText=True)
        #plt.xlim(659.6,660.2)
        plt.legend(prop={'size': 40})
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity $I$ (photons/s)', labelpad=5)
        plt.show()

        #inset graph!
        # left, bottom, width, height = [0.5, 0.65, 0.2, 0.2]
        # ax2 = fig.add_axes([left, bottom, width, height])
        # ax2.plot(self.wavelength, self.circular_total[idx,idx,:].T*100, label='Circular fraction')
        # plt.xlabel('Fraction of circularly polarised light (%)')

        plt.figure(2)
        plt.subplot(121)
        plt.plot(self.wavelength/10, self.gamma[16,idx,:].T, color='black', label='$\gamma$')
        #plt.yticks(np.arange(-45., 46, 45))
        #plt.xlim(659.4, 660.4)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Polarisation angle $\gamma$ (deg.)')

        plt.subplot(122)
        plt.plot(self.wavelength / 10, self.LPF[idx, idx, :].T, '--', label='$LPF$')
        plt.plot(self.wavelength/10, self.circular_total[idx, idx, :].T, label='$CPF$')
        #plt.plot(self.wavelength/10, self.total_unpolarised[idx, idx, :].T, color='black', label='$Unpolarised$')
        plt.yticks(np.arange(0, 1, 0.2))
        plt.xlim(659.6, 660.3)
        plt.legend(prop={'size': 18})
        plt.ylabel('Polarised Fraction', labelpad=10)
        plt.xlabel('Wavelength (nm)')
        plt.show()

    def bfield_parameters(self):
        R = readsav(file_name='/home/sgibson/PycharmProjects/msesim/equi/R.sav')
        self.efitR = R['r']
        Z = readsav(file_name='/home/sgibson/PycharmProjects/msesim/equi/Z.sav')
        self.efitZ = Z['z']
        fluxcoord = readsav(file_name='/home/sgibson/PycharmProjects/msesim/equi/FLUXCOORD.sav')
        self.efitfluxcoord = fluxcoord['fluxcoord']
        rm = readsav(file_name='/home/sgibson/PycharmProjects/msesim/equi/RM.sav')
        self.efitrm = rm['rm']
        BFLD = readsav(file_name='/home/sgibson/PycharmProjects/msesim/equi/BFLD.sav')
        self.bfld = BFLD['bfld']

    def plot_view(self):
        # Load equilibrium data

        diagnostic_view = np.ones((len(self.major_radius), len(self.z)))

        msesim_r, msesim_z = np.meshgrid(self.major_radius, self.z)

        rr, zz = np.meshgrid(self.efitR, self.efitZ)

        levels = np.arange(-0.5, 1.0, 0.2)

        plt.figure()
        plt.contour(rr, zz, self.efitfluxcoord, '--', colors='black', linestyles='dashed', levels=levels)
        plt.contourf(msesim_r, msesim_z, diagnostic_view, colors=cb[0], alpha=0.6)
        plt.ylim(-1.0, 1.0)
        plt.xlim(0.3, 1.5)
        plt.xlabel('R (m)')
        plt.ylabel('Z (m)')
        plt.show()

# nx, ny = 32, 32
# #idl.execute("restore, '/home/sgibson/PycharmProjects/msesim/runs/imse_2d_32x32_f80mm/output/data/MAST_18501_imse.dat', /VERBOSE")
# idl.execute("restore, '/work/sgibson/msesim/runs/imse_2d_32x32_edgecurrent/output/data/MAST_24763_imse.dat', /VERBOSE")
# msesim = MSESIM(nx, ny)
# data = msesim.load_msesim_spectrum()
# msesim.plot_spectrum(radius=1.4)
