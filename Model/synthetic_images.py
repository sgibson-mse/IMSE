import idlbridge as idl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp2d
from Model.bbo_model import Crystal
import pandas as pd

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

#idl.execute("restore, '/home/sam/Desktop/msesim/runs/mast_imse_2d/output/data/density1e19_MAST_photron_2d.dat' , /VERBOSE")

def calculate_synthetic_image(FLC_state):
    idl.execute("restore, '/home/sam/Desktop/msesim/runs/mast_imse_2d_f85mm/output/data/density2e19_MAST_photron_2d.dat' , /VERBOSE")

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
    pixel_size = 20*10**-6

    x = np.arange(+0.5-int(nx/2), int(nx/2), 1)*pixel_size
    y = np.arange(+0.5-int(ny/2), int(ny/2), 1)*pixel_size

    x_small = x[::50]
    y_small = y[::50]

    xx, yy = np.meshgrid(x,y)

    S0_small = stokes_total[:,0,:].reshape(21,21,len(wavelength)) #x,y,lambda
    S1_small = stokes_total[:,1,:].reshape(21,21,len(wavelength)) #x,y,lambda
    S2_small = stokes_total[:,2,:].reshape(21,21,len(wavelength))#x,y,lambda

    # x=x[::5]
    # y=y[::5]

    S_total = np.zeros((len(x), len(y), len(wavelength)))

    for i in range(len(wavelength)):

        #Make the delay and displacer plate to find the phase shift for a given wavelength

        delay_plate = Crystal(wavelength=wavelength[i], thickness=0.015, cut_angle=0., name='alpha_bbo', nx=nx, ny=ny, pixel_size=pixel_size, orientation=90.)

        displacer_plate = Crystal(wavelength=wavelength[i], thickness=0.003, cut_angle=45., name='alpha_bbo', nx=nx, ny=ny, pixel_size=pixel_size, orientation=90.)

        #Interpolate our small 200x200 msesim grid to the real image size, 1024x1024

        S0_interp = interp2d(x_small, y_small, S0_small[:, :, i], kind='cubic')
        S0 = S0_interp(x,y)

        S1_interp = interp2d(x_small, y_small, S1_small[:, :, i], kind='cubic')
        S1 = S1_interp(x,y)

        S2_interp = interp2d(x_small, y_small, S2_small[:, :, i], kind='cubic')
        S2 = S2_interp(x,y)

        #Calculate the total intensity for a given wavelength, propagating stokes components through a delay plate and a displacer plate.

        if FLC_state == 1:
            S_total[:,:,i] = S0 + S1*np.sin((delay_plate.phi_total + displacer_plate.phi_total)) + S2*np.cos((delay_plate.phi_total + displacer_plate.phi_total))
        if FLC_state == 2:
            S_total[:,:,i] = S0 + S1*np.sin((delay_plate.phi_total + displacer_plate.phi_total)) - S2*np.cos((delay_plate.phi_total + displacer_plate.phi_total))

    synthetic_image = np.sum(S_total, axis=2)

    return synthetic_image

def save_image(image, filename):

    x = pd.HDFStore(filename)

    x.append("a", pd.DataFrame(image)) # <-- This will take a while.
    x.close()

    return

image_1 = calculate_synthetic_image(FLC_state=1)
print('Made image 1! Saving...')
save_image(image_1, filename="image_FLC1.hdf")

print('Making image 2...')
image_2 = calculate_synthetic_image(FLC_state=2)
print('Image 2 complete! Saving...')
save_image(image_2, filename="image_FLC2.hdf")


# gamma_profile = np.sum(polarisation_angle, axis=2)
#
# plt.figure()
# ax = plt.subplot(111)
# plt.imshow(gamma_profile*(180./np.pi))
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.gca().invert_yaxis()
# plt.gca().invert_xaxis()
# cs = plt.colorbar()
# cs.ax.set_ylabel('Intensity (Photons/s)', rotation=90)
# plt.show()

#S_summed90 = np.sum(S_total90, axis=2)

# plt.figure()
# plt.imshow(phase_hyperbolic_displacer[:,:,100])
# plt.gca().invert_yaxis()
# plt.gca().invert_xaxis()
# plt.colorbar()
# plt.show()
#
# plt.figure()
# plt.imshow(phase_hyperbolic_delay[:,:,100])
# plt.gca().invert_yaxis()
# plt.gca().invert_xaxis()
# plt.colorbar()
# plt.show()
#
#
# plt.figure()
# ax = plt.subplot(111)
# plt.imshow(S_summed45, cmap='gray')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.gca().invert_yaxis()
# plt.gca().invert_xaxis()
# cs = plt.colorbar()
# cs.ax.set_ylabel('Intensity (Photons/s)', rotation=90)
# plt.show()

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












