import idlbridge as idl
import numpy as np
import matplotlib.pyplot as plt

from Tools.Plotting.graph_format import plot_format

plot_format()


class MSESIM(object):

    def __init__(self, nx, ny):

        self.nx = nx
        self.ny = ny
        self.data = self.load_spectrum()
        self.stokes_total = np.array(self.data["total_stokes"])
        #Reshape array into 2x2 grid
        self.stokes_total = self.stokes_total.reshape(self.nx,self.ny,4,len(self.data["wavelength_vector"]))
        self.S0 = self.stokes_total[:,:,0,:]
        self.S1 = self.stokes_total[:,:,1,:]
        self.S2 = self.stokes_total[:,:,2,:]
        self.S3 = self.stokes_total[:,:,3,:]
        self.wavelength = np.array(self.data["wavelength_vector"])/10 #nm
        self.major_radius = np.array(self.data["resolution_vector(R)"])[:,0]
        # major radius 2d grid basically each row is the same...
        self.major_radius = self.major_radius.reshape(self.nx, self.ny)[16,::-1]

    def load_spectrum(self):
        """
        This function restores an idl .dat file from an msesim run, and stores the output in a dictionary for use in python.
        :return: Dictionary of outputs from msesim. Retrievable using the object_names as given below.
        """

        data = {}

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

        object_names = (
            'central_coordinates', 'beam_velocity_vector', 'beam_duct_coordinates', 'half_beam_sampling_width',
            'beam_axis_vector',
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

        for key_name, object_name in zip(key_names, object_names):
            data[object_name] = idl.get(key_name)

        return data

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

    def plot_spectrum(self, radius):

        # Find the radii for the spectrum you want to plot
        idx = msesim._find_nearest(msesim.major_radius, value=radius)

        plt.figure(1)
        plt.plot(self.wavelength, self.S0[idx,idx,:].T, color='black', label='Total Intensity')
        plt.plot(self.wavelength, np.sqrt(self.S2[idx,idx,:].T**2 + self.S1[idx,idx,:].T**2), label='Linearly Polarised')
        plt.plot(self.wavelength, self.S3[idx,idx,:].T, label='Circularly Polarised')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Pixel Intensity ($\gamma$/s)')
        plt.show()


#Example
#Restore the .dat file
idl.execute("restore, '/home/sgibson/PycharmProjects/msesim/runs/imse_2d_32x32_f80mm/output/data/MAST_18501_imse.dat', /VERBOSE")
#Give msesim run dimensions
nx,ny = 32,32
#Create instance of the class
msesim = MSESIM(nx,ny)
#Plot the spectrum and feed it the major radius you want to plot for.
msesim.plot_spectrum(radius=2)
