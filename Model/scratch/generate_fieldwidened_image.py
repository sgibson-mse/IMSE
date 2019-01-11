import idlbridge as idl
import numpy as np

def get_msesim_output():
    idl.execute("restore, '/work/sgibson/msesim/runs/imse_2d_32x32_edgecurrent_test/output/data/MAST_24409_imse.dat' , /VERBOSE")

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

    object_names = ('central_coordinates', 'beam_velocity_vector', 'beam_duct_coordinates', 'half_beam_sampling_width', 'beam_axis_vector',
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

    stokes_total = data["total_stokes"]

    wavelength = data["wavelength_vector"]
    wavelength = wavelength*10**-10 #convert to m

    R = data["resolution_vector(R)"]
    R = R[:,0]

    nx = 1024
    ny= 1024
    npx = 32
    pixel_size = 20*10**-6

    x = np.arange(-10.24,10.242,0.02)
    y = np.arange(-10.24,10.242,0.02)

    x_small = np.linspace(-10.24,10.24,npx)
    y_small = np.linspace(-10.24,10.24,npx)

    S0_small = stokes_total[:,0,:].reshape(npx,npx,len(wavelength)) #x,y,lambda
    S1_small = stokes_total[:,1,:].reshape(npx,npx,len(wavelength)) #x,y,lambda
    S2_small = stokes_total[:,2,:].reshape(npx,npx,len(wavelength))#x,y,lambda
    S3_small = stokes_total[:,3,:].reshape(npx,npx,len(wavelength))#x,y,lambda

    return wavelength, nx, ny, pixel_size, x_small, y_small, S0_small, S1_small, S2_small, S3_small, x, y