import idlbridge as idl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp2d
from Model.bbo_model import AlphaBBO

idl.execute("restore, '/home/sam/Desktop/msesim/runs/mast_imse_2d/output/data/density1e19_MAST_photron_2d.dat' , /VERBOSE")

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
R = data["resolution_vector(R)"]
R = R[:,0]

nx = 1024
ny= 1024
pixel_size = 20*10**-3

x = np.arange(-int(nx/2), int(nx/2)+1, 1)*pixel_size
y = np.arange(-int(ny/2), int(ny/2)+1, 1)*pixel_size

x_small = x[::50]
y_small = y[::50]

S0 = stokes_total[:,0,:].reshape(21,21,536) #x,y,lambda
S1 = stokes_total[:,1,:].reshape(21,21,536) #x,y,lambda
S2 = stokes_total[:,2,:].reshape(21,21,536) #x,y,lambda
S3 = stokes_total[:,3,:].reshape(21,21,536) #x,y,lambda

# Make crystals

# Make the delay and displacer plates
delay_plate = AlphaBBO(wavelength=wavelength, thickness=15000, cut_angle=0, alpha=11.3)
displacer_plate = AlphaBBO(wavelength=wavelength, thickness=3000, cut_angle=45, alpha=11.3)

S_total45 = np.zeros((len(x_small), len(y_small), len(wavelength)))
S_total90 = np.zeros((len(x_small), len(y_small), len(wavelength)))

for j in range(len(y_small)):
    for i in range(len(wavelength)):
        S_total45[:,j,i] = S0[:,j,i] + S1[:,j,i]*np.sin((delay_plate.phi_0[i]+displacer_plate.phi_0[i]+displacer_plate.phi_shear[i]*y_small[j])) + S1[:,j,i]*np.cos((delay_plate.phi_0[i]+displacer_plate.phi_0[i]+displacer_plate.phi_shear[i]*y_small[j]))
        S_total90[:,j,i] = S0[:,j,i] + S1[:,j,i]*np.sin((delay_plate.phi_0[i]+displacer_plate.phi_0[i]+displacer_plate.phi_shear[i]*y_small[j])) - S1[:,j,i]*np.cos((delay_plate.phi_0[i]+displacer_plate.phi_0[i]+displacer_plate.phi_shear[i]*y_small[j]))

S_summed45 = np.sum(S_total45, axis=2)
S_summed90 = np.sum(S_total90, axis=2)

plt.figure()
plt.imshow(S_summed45, cmap='gray')
plt.show()

plt.figure()
plt.imshow(S_summed90, cmap='gray')
plt.show()

#plt.gca().invert_xaxis()
# for i in range(len(wavelength)):
#     I_interp = interp2d(x_small, y_small, I[:,:,i], kind='cubic')
#     I_new[:,:,i] = I_interp(x,y)
#
# np.save('S2.npy', I_new)
# np.save('wavelength.npy', wavelength)
# np.save('R.npy', R)













