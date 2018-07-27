import idlbridge as idl
import numpy as np
from scipy.interpolate import interp2d

def load_msesim_spectrum():
    idl.execute(
        "restore, '/home/sgibson/PycharmProjects/msesim/runs/imse_2d_32x32_centerpixel/output/data/MAST_18501_imse.dat', /VERBOSE")
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
    'central_coordinates', 'beam_velocity_vector', 'beam_duct_coordinates', 'half_beam_sampling_width', 'beam_axis_vector',
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

# data = load_msesim_spectrum()
#
# stokes_total = np.array(data["total_stokes"])
#
# stokes_total = stokes_total.reshape(32,32,4,544)
# stokes_total = stokes_total[:,:,0,:]
#
# x = np.arange(-10.23, 10.25, 0.02)
# y = np.arange(-10.23, 10.25, 0.02)
#
# x_small = np.linspace(-10.23,10.23,32)
# y_small = np.linspace(-10.23,10.23,32)
#
# wavelength = data["wavelength_vector"]
# wavelength = wavelength * 10 ** -10  # convert to m
#
# print(wavelength)
#
# S_total = np.zeros((len(x), len(y), len(wavelength)))
#
# for i in range(len(wavelength)):
#     S0_interp = interp2d(x_small, y_small, stokes_total[:,:,i], kind='linear')
#     S0 = S0_interp(x, y)
#     S_total[:,:,i] = S0
#
# tsh_image = np.sum(S_total, axis=2)
#
# import matplotlib.pyplot as plt
#
# plt.figure()
# plt.imshow(tsh_image)
# plt.legend()
# plt.show()

