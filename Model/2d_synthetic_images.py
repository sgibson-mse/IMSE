import idlbridge as idl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp2d
from Model.bbo_model import Crystal

SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

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
wavelength = wavelength/10 #convert to nm

R = data["resolution_vector(R)"]
R = R[:,0]

nx = 1024
ny= 1024
pixel_size = 20*10**-6

x = np.arange(-int(nx/2), int(nx/2)+1, 1)*pixel_size
y = np.arange(-int(ny/2), int(ny/2)+1, 1)*pixel_size

x_small = x[::50]
y_small = y[::50]

x = x[::5]
y = y[::5]

xx, yy = np.meshgrid(x,y)

S0_small = stokes_total[:,0,:].reshape(21,21,536) #x,y,lambda
S1_small = stokes_total[:,1,:].reshape(21,21,536) #x,y,lambda
S2_small = stokes_total[:,2,:].reshape(21,21,536)#x,y,lambda
S3_small = stokes_total[:,3,:].reshape(21,21,536) #x,y,lambda

# Make crystals
# Make the delay and displacer plates

delay_plate = Crystal(wavelength=wavelength, thickness=15000, cut_angle=0, name='lithium_niobate')
displacer_plate = Crystal(wavelength=wavelength, thickness=3000, cut_angle=45, name='alpha_bbo')

print('delay plate', delay_plate.kappa[0], delay_plate.birefringence[0], delay_plate.ne[0], delay_plate.no[0])
print('displacer plate', displacer_plate.kappa[0], displacer_plate.birefringence[0], displacer_plate.ne[0], displacer_plate.no[0])

def interpolate(x,y,wavelength,x_small,y_small, I):

    I_new = np.zeros((len(x), len(y), len(wavelength)))

    for i in range(len(wavelength)):
        I_interp = interp2d(x_small, y_small, I[:, :, i], kind='cubic')
        I_new[:, :, i] = I_interp(x, y)

    return I_new

S0 = interpolate(x,y,wavelength,x_small,y_small, S0_small)
S1 = interpolate(x,y,wavelength,x_small,y_small, S1_small)
S2 = interpolate(x,y,wavelength,x_small,y_small, S2_small)

S_total45 = np.zeros((len(x), len(y), len(wavelength)))
S_total90 = np.zeros((len(x), len(y), len(wavelength)))
phase_shear = np.zeros((len(x), len(y), len(wavelength)))
phase_hyperbolic_delay = np.zeros((len(x), len(y), len(wavelength)))
phase_hyperbolic_displacer = np.zeros((len(x), len(y), len(wavelength)))

for i in range(len(wavelength)):
    phase_shear[:,:,i] = displacer_plate.phi_shear[i] * yy
    phase_hyperbolic_delay[:,:,i] = delay_plate.phi_hyperbolic[i] * ( (3-np.cos(2*delay_plate.cut_angle))*xx**2 - ((3*np.cos(2*delay_plate.cut_angle))-1)*yy**2 )
    phase_hyperbolic_displacer[:,:,i] = displacer_plate.phi_hyperbolic[i] * ( (3-np.cos(2*displacer_plate.cut_angle))*xx**2 - ((3*np.cos(2*displacer_plate.cut_angle))-1)*yy**2 )

    S_total45[:,:,i] = S0[:,:,i] + S1[:,:,i]*np.sin((delay_plate.phi_0[i]+displacer_plate.phi_0[i]+phase_shear[:,:,i]+phase_hyperbolic_delay[:,:,i]+phase_hyperbolic_displacer[:,:,i])) + S2[:,:,i]*np.cos((delay_plate.phi_0[i]+displacer_plate.phi_0[i]+phase_shear[:,:,i]+phase_hyperbolic_delay[:,:,i]+phase_hyperbolic_displacer[:,:,i]))
    #S_total90[:,:,i] = S0[:,:,i] + S1[:,:,i]*np.sin((delay_plate.phi_0[i]+displacer_plate.phi_0[i]+phase_shear[:,:,i]+phase_hyperbolic_delay[:,:,i]+phase_hyperbolic_displacer[:,:,i])) - S2[:,:,i]*np.cos((delay_plate.phi_0[i]+displacer_plate.phi_0[i]+phase_shear[:,:,i]+phase_hyperbolic_delay[:,:,i]+phase_hyperbolic_displacer[:,:,i]))

S_summed45 = np.sum(S_total45, axis=2)
#S_summed90 = np.sum(S_total90, axis=2)

plt.figure()
plt.title('LiNO3')
plt.imshow(phase_hyperbolic_displacer[:,:,100])
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.colorbar()
plt.show()

plt.figure()
plt.title('LiNO3')
plt.imshow(phase_hyperbolic_delay[:,:,100])
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.colorbar()
plt.show()


plt.figure()
ax = plt.subplot(111)
plt.title('LiNO3')
plt.imshow(S_summed45, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
cs = plt.colorbar()
cs.ax.set_ylabel('Intensity (Photons/s)', rotation=90)
plt.show()

# plt.figure()
# ax1 = plt.subplot(111)
# plt.imshow(S_summed90, cmap='gray')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.gca().invert_yaxis()
# plt.gca().invert_xaxis()
# cs2 = plt.colorbar()
# cs2.ax1.set_ylabel('Intensity (Photons/s)', rotation=90)
# plt.show()












