import idlbridge as idl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

"""
This code takes the output .dat file from MSESIM and reads in the commonly used parameters from the file. Can easily plot the spectrum for a specific major radius, and the 
beam emission intensity as a function of R,Z. Can make multiple instances of this class for different MSESIM runs by specifying the filepath.

"""

class MSESIM(object):

    def __init__(self, filepath):

        idl.execute("restore, '{0}' , /VERBOSE".format(filepath))
        self.data = self.create_data_dictionary()
        self.get_values()

    @staticmethod
    def _find_nearest(array, value):

        if value < array.min() or value > array.max():
            raise IndexError("Requested value is outside the range of the data. The range of the data is from {}m to {}m".format(array.min(),array.max()))

        index = np.searchsorted(array, value, side="right")

        if (value - array[index])**2 < (value - array[index + 1])**2:
            return index
        else:
            return index + 1

    def create_data_dictionary(self):

        """
        Stores the output of an msesim run (stored in a .dat file) in a dictionary for use in python.
        :return: Data (dict) - Dictionary of parameters output by msesim. Retrievable using dictionary formatting,
                 using the more descriptive *object_names* as given below.
        """

        self.data = {}

        key_names = ("channels", "chanid",
                     "xyz0", "B_v0", "B_xyz", "B_w", "B_vec",
                     "C_xyz", "C_k", "vec0", "Bfld0", "Efld0",
                     "Dshift0", "Sshift0", "alpha0", "psi0",

                     "gp_xyz", "gp_vel", "gp_vec", "gp_bfld",
                     "gp_efld", "gp_alpha", "gp_psi", "gp_emis",

                     "R", "Z", "psi", "RZpsi", "Rm",
                     "RZ_emis", "R_emis", "psi_emis", "psi_res",
                     "R_res",

                     "lambda", "pstokes", "sstokes", "stokes",
                     "cwlstokes", "sigstokes", "pibstokes", "pirstokes")

        self.object_names = ("channels", "channel_id",
                        'central_coordinates', 'beam_velocity_vector', 'beam_duct_coordinates', 'half_beam_sampling_width', 'beam_axis_vector',
                        'collection_lens_coordinates', 'optical_axis',
                        'emission_vector', 'bfield_vector', 'efield_vector',
                        'doppler_shift', 'max_stark_shift',
                        'polarisation_angles', 'psi_normalised',

                        'grid_coordinates', 'beam_velocity_vector', 'emission_vector',
                        'bfield_vector', 'efield_vector', 'polarisation_angles',
                        'psi_normalised', 'emission_intensity',

                        'R', 'Z', 'psi_normalised', 'psi(R,Z)', 'magnetic_axis',
                        'emission_intensity(RZ)', 'emission_intensity(R)', 'emission_intensity(psi)',
                        'resolution_vector(psi)', 'resolution_vector(R)',

                        'wavelength', 'pi_stokes', 'sigma_stokes', 'total_stokes',
                        'cwl_stokes', 'optimal_sigma_wavelength_stokes', 'optimal_blueshift_pi_wavelength', 'optimal_redshift_pi_wavelength'
                        )

        for key_name, object_name in zip(key_names, self.object_names):
            self.data[object_name] = idl.get(key_name)

        return self.data

    def get_values(self):

        """
        :return: Get some of the commonly used parameters from the save file:
            From the file:

            - Channel numbers, wavelength vector (nm), stokes vector and components, major radii that correspond to each channel,
            - R, Z, poloidal flux function and normalised flux,
            - Beam velocity vector, half sampling width, axis and the central co-ordinate of the beam duct
            - Collection optics position and optical axis,
            - B and E field vectors

            Derived quanities:
             - Linearly polarised fraction, circularly polarised fraction, unpolarised fraction, and polarisation angle.

        """

        self.channels = self.data['channels']

        #Stokes components

        self.wavelength = self.data['wavelength']/10
        self.stokes_vector = self.data['total_stokes'] # channels, stokes component, wavelength
        self.major_radius = self.data["resolution_vector(R)"][:,0]
        self.S0 = self.stokes_vector[:, 0, :]
        self.S1 = self.stokes_vector[:, 1, :]
        self.S2 = self.stokes_vector[:, 2, :]
        self.S3 = self.stokes_vector[:, 3, :]

        #Flux function

        self.R = self.data["R"]
        self.Z = self.data["Z"]
        self.psi_2d = self.data['psi(R,Z)']
        self.psi_normalised = self.data['psi_normalised']

        # Beam Geometry

        self.beam_velocity_vector = self.data["beam_velocity_vector"]
        self.beam_half_sampling_width = self.data["half_beam_sampling_width"]
        self.beam_axis_vector = self.data["beam_axis_vector"]
        self.duct_coordinates = self.data["beam_duct_coordinates"]

        # Collection Optics
        self.collection_lens = self.data["collection_lens_coordinates"]
        self.optical_axis = self.data["optical_axis"]

        self.central_coordinates = self.data['central_coordinates']

        #Field Vectors

        self.Bfld_vector = self.data["bfield_vector"]
        self.Efld_vector = self.data["efield_vector"]

        self.emission_intensity_R = self.data['emission_intensity(R)']

        #Check if the stokes array is square - if so, then it's probably an image so make the array 2D

        if int(np.sqrt(len(self.channels)) + 0.5) ** 2 == int(len(self.channels)):
            self.stokes_vector = self.stokes_vector.reshape(int(np.sqrt(len(self.channels))),int(np.sqrt(len(self.channels))),len(self.stokes_vector[0,:,0]), len(self.wavelength)) # x, y, stokes, wavelength

            self.major_radius = self.major_radius.reshape(int(np.sqrt(len(self.channels))),int(np.sqrt(len(self.channels))))[0,:]
            self.central_coordinates = self.central_coordinates.reshape(int(np.sqrt(len(self.channels))),int(np.sqrt(len(self.channels))), 3)

            self.S0 = self.stokes_vector[:, :, 0, :]
            self.S1 = self.stokes_vector[:, :, 1, :]
            self.S2 = self.stokes_vector[:, :, 2, :]
            self.S3 = self.stokes_vector[:, :, 3, :]

            self.Bfld_vector = self.Bfld_vector.reshape(int(np.sqrt(len(self.channels))),int(np.sqrt(len(self.channels))), 20 , 3)
            self.Efld_vector = self.Efld_vector.reshape(int(np.sqrt(len(self.channels))),int(np.sqrt(len(self.channels))), 20 , 3)

            self.emission_intensity_R = self.data['emission_intensity(R)'].reshape(int(np.sqrt(len(self.channels))),
                                                       int(np.sqrt(len(self.channels))), 400)


        else:
            pass

        np.seterr(divide='ignore', invalid='ignore')

        self.polarised_fraction = np.sqrt(self.S1**2 + self.S2**2 + self.S3**2)/self.S0

        self.unpolarised_fraction = (self.S0 - np.sqrt(self.S1**2 + self.S2**2 + self.S3**2))/self.S0
        self.circular_fraction = np.sqrt(self.S3**2 - (self.unpolarised_fraction**2*self.S0))/self.S0

        self.LPF = np.sqrt(self.S1**2 + self.S2**2)/self.S0
        self.CPF = self.circular_fraction/self.S0

        self.total_circular_fraction = np.sqrt(self.S3**2)/self.S0

        self.polarisation_angle = 0.5*np.arctan2(self.S2, self.S1)*(180./np.pi)

        return

    def plot_spectrum(self, radius):

        # Find the radii for the spectrum you want to plot
        idx = self._find_nearest(np.sort(self.major_radius), value=radius)

        fig= plt.figure(1)
        gs1 = gridspec.GridSpec(nrows=3, ncols=3)
        ax1 = fig.add_subplot(gs1[:-1, :])
        plt.plot(self.wavelength, self.S0[idx,idx,:].T, color='black', label='$I_{\mathrm{total}}$')
        plt.plot(self.wavelength, np.sqrt(self.S2[idx,idx,:].T**2 + self.S1[idx,idx,:].T**2), label='$I_{\mathrm{linear}}$')
        plt.plot(self.wavelength, self.S3[idx,idx,:].T, label='$I_{\mathrm{circular}}$')
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useOffset=False, useMathText=True)
        plt.legend(prop={'size': 40})
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity $I$ (photons/s)', labelpad=5)

        ax2 = fig.add_subplot(gs1[-1, :-1])
        plt.plot(self.wavelength, self.polarisation_angle[int(len(self.major_radius)/2),idx,:].T, label='$\gamma$')
        plt.yticks(np.arange(-45., 46, 45))
        plt.xlim(659.4, 660.4)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Polarisation angle $\gamma$ (deg.)')

        ax3 = fig.add_subplot(gs1[-1, -1])
        ax3.plot(self.wavelength, self.total_circular_fraction[idx, idx, :].T, label='$CPF$')
        ax3.plot(self.wavelength, self.unpolarised_fraction[idx, idx, :].T, color='black', label='$UF$')
        plt.yticks(np.arange(0, 1, 0.2))
        ax3.yaxis.tick_right()
        ax3.yaxis.set_label_position("right")
        ax3.legend(prop={'size': 18})
        plt.ylabel('Polarised Fraction', labelpad=10)
        plt.xlabel('Wavelength (nm)')
        plt.show()

    def plot_emission(self):

        """
        Sum up over wavelengths. Plot the emission intensity from the neutral beam as a function of R,Z.
        :return:
        """

        #Take Z co-ordinate
        z = self.central_coordinates[:,:,2]

        z = z[:,0]
        r = self.major_radius

        rr, zz = np.meshgrid(r,z)

        emission = np.sum(self.emission_intensity_R,axis=2)

        plt.figure()
        plt.pcolormesh(rr, zz, emission, shading='gouraud')
        cbar = plt.colorbar()
        cbar.set_label('Beam emission intensity (photons/s)', rotation=90)
        plt.xlabel('R (m)')
        plt.ylabel('Z (m)')
        plt.show()

        return

filepath = '/work/sgibson/msesim/runs/imse_2d_32x32_magneticsequi_edgecurrent/output/data/MAST_24763_imse.dat'
msesim = MSESIM(filepath=filepath)

msesim.plot_spectrum(radius=1.4)
msesim.plot_emission()