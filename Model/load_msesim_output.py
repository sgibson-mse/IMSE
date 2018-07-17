import idlbridge as idl

def load_msesim_spectrum():
    idl.execute(
        "restore, '/home/sam/Desktop/msesim/runs/mast_imse_2d_f85mm/output/data/density2e19_MAST_photron_2d.dat' , /VERBOSE")

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