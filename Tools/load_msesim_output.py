import idlbridge as idl
import numpy as np

class MSESIM(object):

    def __init__(self):

        self.data = self.load_msesim_spectrum()
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


#Example
#idl.execute("restore, '/home/sgibson/PycharmProjects/msesim/runs/imse_2d_32x32_centerpixel/output/data/MAST_18501_imse.dat', /VERBOSE")
#msesim = MSESIM()
#data =