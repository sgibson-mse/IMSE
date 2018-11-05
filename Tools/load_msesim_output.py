import idlbridge as idl
import numpy as np

# import matplotlib as matplotlib
# matplotlib.use('Qt4Agg',warn=False, force=True)
# print("Switched to:",matplotlib.get_backend())

import matplotlib.pyplot as plt
from Tools.Plotting.graph_format import plot_format, plot_save
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec

cb = plot_format()

from Model.Camera import Camera

camera = Camera(photron=True)

class MSESIM(object):

    def __init__(self, nx, ny, dimension):

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

        if dimension == 2:

            self.nx = nx
            self.ny = ny
            self.data = self.load_msesim_spectrum()
            self.parameters_2d()
        else:
            self.nx = nx
            self.data = self.load_msesim_spectrum()
            self.parameters_1d()


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

    def parameters_1d(self):

        self.stokes_total = np.array(self.data["total_stokes"])
        #Reshape array into 2x2 grid
        self.S0 = self.stokes_total[:,0,:]
        self.S1 = self.stokes_total[:,1,:]
        self.S2 = self.stokes_total[:,2,:]
        self.S3 = self.stokes_total[:,3,:]
        self.wavelength = np.array(self.data["wavelength_vector"])/10 #in nanometers
        self.major_radius = np.array(self.data["resolution_vector(R)"])[:,0]
        # major radius 2d grid basically each row is the same...
        self.major_radius = self.major_radius[::-1]
        self.x_coords = np.linspace(-1*camera.sensor_size/2, camera.sensor_size/2, self.nx)
        self.polarised_fraction = np.sqrt(self.S1**2 + self.S2**2+self.S3**2)/self.S0
        self.total_unpolarised = (self.S0 - np.sqrt(self.S1**2 + self.S2**2 + self.S3**2))/self.S0
        self.total_circular = np.sqrt(self.S3**2 - (self.total_unpolarised**2*self.S0))/self.S0

        self.LPF = np.sqrt(self.S1**2 + self.S2**2)/self.S0
        self.CPF = self.total_circular/self.S0

        self.circular_total = np.sqrt(self.S3**2)/self.S0
        self.gamma = 0.5*np.arctan2(self.S2, self.S1)*(180./np.pi)

    def parameters_2d(self):

        self.stokes_total = np.array(self.data["total_stokes"])
        #Reshape array into 2x2 grid
        self.stokes_total = self.stokes_total.reshape(self.nx,self.ny,4,len(self.data["wavelength_vector"]))
        self.S0 = self.stokes_total[:,:,0,:]
        self.S1 = self.stokes_total[:,:,1,:]
        self.S2 = self.stokes_total[:,:,2,:]
        self.S3 = self.stokes_total[:,:,3,:]
        self.wavelength = np.array(self.data["wavelength_vector"])/10 #in nanometers
        self.major_radius = np.array(self.data["resolution_vector(R)"])[:,0]
        # major radius 2d grid basically each row is the same...
        self.major_radius = self.major_radius.reshape(self.nx, self.ny)[16,::-1]
        self.x_coords = np.linspace(-1*camera.sensor_size/2, camera.sensor_size/2, self.nx)
        self.y_coords = np.linspace(-1*camera.sensor_size/2, camera.sensor_size/2, self.ny)
        self.polarised_fraction = np.sqrt(self.S1**2 + self.S2**2+self.S3**2)/self.S0
        self.total_unpolarised = (self.S0 - np.sqrt(self.S1**2 + self.S2**2 + self.S3**2))/self.S0
        self.total_circular = np.sqrt(self.S3**2 - (self.total_unpolarised**2*self.S0))/self.S0

        self.LPF = np.sqrt(self.S1**2 + self.S2**2)/self.S0
        self.CPF = self.total_circular/self.S0

        self.circular_total = np.sqrt(self.S3**2)/self.S0
        self.gamma = 0.5*np.arctan2(self.S2, self.S1)*(180./np.pi)

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
        idx = self._find_nearest(self.major_radius, value=radius)

        #manager = plt.get_current_fig_manager()
        #manager.window.showMaximized()

        fig= plt.figure(1)
        gs1 = gridspec.GridSpec(nrows=3, ncols=3)
        ax1 = fig.add_subplot(gs1[:-1, :])
        plt.plot(self.wavelength, self.S0[idx,idx,:].T, color='black', label='$I_{\mathrm{total}}$')
        plt.plot(self.wavelength, np.sqrt(self.S2[idx,idx,:].T**2 + self.S1[idx,idx,:].T**2), label='$I_{\mathrm{linear}}$')
        plt.plot(self.wavelength, self.S3[idx,idx,:].T, label='$I_{\mathrm{circular}}$')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useOffset=False, useMathText=True)
        plt.xlim(659.6,660.2)
        plt.legend(prop={'size': 40})
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity $I$ (photons/s)', labelpad=5)

        #inset graph!
        # left, bottom, width, height = [0.5, 0.65, 0.2, 0.2]
        # ax2 = fig.add_axes([left, bottom, width, height])
        # ax2.plot(self.wavelength, self.circular_total[idx,idx,:].T*100, label='Circular fraction')
        # plt.xlabel('Fraction of circularly polarised light (%)')

        ax2 = fig.add_subplot(gs1[-1, :-1])
        plt.plot(self.wavelength, self.gamma[16,idx,:].T, color=cb[3], label='$\gamma$')
        plt.yticks(np.arange(-45., 46, 45))
        plt.xlim(659.4, 660.4)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Polarisation angle $\gamma$ (deg.)')

        ax3 = fig.add_subplot(gs1[-1, -1])
        ax3.plot(self.wavelength, self.circular_total[idx, idx, :].T, color=cb[2], label='$CPF$')
        ax3.plot(self.wavelength, self.total_unpolarised[idx, idx, :].T, color='black', label='$UF$')
        plt.yticks(np.arange(0, 1, 0.2))
        plt.xlim(659.6, 660.1)
        ax3.yaxis.tick_right()
        ax3.yaxis.set_label_position("right")
        ax3.legend(prop={'size': 18})
        plt.ylabel('Polarised Fraction', labelpad=10)
        plt.xlabel('Wavelength (nm)')
        plt.show()
        #plot_save(filename='mse_spectrum_imse.pdf')


#Example
# idl.execute("restore, '/home/sgibson/PycharmProjects/msesim/runs/imse_2d_32x32_f80mm_annuli/output/data/MAST_18501_imse.dat', /VERBOSE")
# msesim = MSESIM(nx=32, ny=32)
# data = msesim.load_msesim_spectrum()
# msesim.plot_spectrum(radius=0.94)

# idl.execute("restore, '/home/sgibson/PycharmProjects/msesim/runs/imse_2d_32x32_f80mm_nostarkzeeman/output/data/MAST_18501_imse.dat', /VERBOSE")
# msesim_nostarkzeeman = MSESIM(nx=32, ny=32)
# data_nosz = msesim.load_msesim_spectrum()
# msesim_nostarkzeeman.plot_spectrum(radius=0.94)

#msesim.plot_spectrum(radius=0.94)