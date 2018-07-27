import idlbridge as idl
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp2d
import pandas as pd

from Model.Crystal import Crystal

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

def get_msesim_output():
    idl.execute("restore, '/home/sgibson/PycharmProjects/msesim/runs/imse_2d_32x32_centerpixel/output/data/MAST_18501_imse.dat' , /VERBOSE")

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

    nx =1024
    ny= 1024
    pixel_size = 20*10**-6

    npx = 32

    x = np.arange(-10.23, 10.25, 0.02)
    y = np.arange(-10.23, 10.25, 0.02)

    x_small = np.linspace(-10.23, 10.23, npx)
    y_small = np.linspace(-10.23, 10.23, npx)

    S0_small = stokes_total[:,0,:].reshape(npx,npx,len(wavelength)) #x,y,lambda
    S1_small = stokes_total[:,1,:].reshape(npx,npx,len(wavelength)) #x,y,lambda
    S2_small = stokes_total[:,2,:].reshape(npx,npx,len(wavelength))#x,y,lambda
    S3_small = stokes_total[:,3,:].reshape(npx,npx,len(wavelength))#x,y,lambda

    return wavelength, nx, ny, pixel_size, x_small, y_small, S0_small, S1_small, S2_small, S3_small, x, y

def calculate_TSH_image(FLC_state):

    """
    Calculate the intensity of an image for the imaging MSE system with an FLC as a function of wavelength.
    :param FLC_state: 1 or 2, switches between 45 degrees and 90 degrees polarisation state
    :return: Intensity on image, summed up over wavelength.
    """

    wavelength, nx, ny, pixel_size, x_small, y_small, S0_small, S1_small, S2_small, S3_small, x, y = get_msesim_output()

    S_total = np.zeros((len(x), len(y), len(wavelength)))

    print('Calculating TSH synthetic image...')

    for i in range(len(wavelength)):
        #Make the delay and displacer plate to find the phase shift for a given wavelength

        delay_plate = Crystal(wavelength=wavelength[i], thickness=0.015, cut_angle=0., name='alpha_bbo', nx=nx, ny=ny, pixel_size=pixel_size, orientation=90., two_dimensional=True)
        displacer_plate = Crystal(wavelength=wavelength[i], thickness=0.003, cut_angle=45., name='alpha_bbo', nx=nx, ny=ny, pixel_size=pixel_size, orientation=90.,two_dimensional=True )

        #Interpolate our small msesim grid to the real image size, 1024x1024

        S0_interp = interp2d(x_small, y_small, S0_small[:, :, i], kind='linear')
        S0 = S0_interp(x,y)

        S1_interp = interp2d(x_small, y_small, S1_small[:, :, i], kind='linear')
        S1 = S1_interp(x,y)

        S2_interp = interp2d(x_small, y_small, S2_small[:, :, i], kind='linear')
        S2 = S2_interp(x,y)

        #Calculate the total intensity for a given wavelength, propagating stokes components through a delay plate and a displacer plate.

        if FLC_state == 1:
            print('FLC state is 1')
            S_total[:,:,i] = S0 + S1*np.sin((delay_plate.phi_total + displacer_plate.phi_total)) + S2*np.cos((delay_plate.phi_total + displacer_plate.phi_total))

        else:
            print('FLC state is 2')
            S_total[:,:,i] = S0 + S1*np.sin((delay_plate.phi_total + displacer_plate.phi_total)) - S2*np.cos((delay_plate.phi_total + displacer_plate.phi_total))

    print('save the image!')
    tsh_image = np.sum(S_total, axis=2)

    return tsh_image

def calculate_ASH_image(circular):
    """
    Calculate the intensity of the image for an amplitude spatial heterodyne system
    :param circular: True/False - Include S3 component to realistically model effect of S3 on carrier amplitudes.
    :return:
    """

    #Get output variables from MSESIM
    wavelength, nx, ny, pixel_size, x_small, y_small, S0_small, S1_small, S2_small, S3_small, x, y = get_msesim_output()

    S_total = np.zeros((len(x), len(y), len(wavelength)))

    for i in range(len(wavelength)):
        savart_1 = Crystal(wavelength=wavelength[i], thickness=0.00210, cut_angle=45., name='alpha_bbo', nx=nx, ny=ny,
                           pixel_size=pixel_size, orientation=45., two_dimensional=True)
        savart_2 = Crystal(wavelength=wavelength[i], thickness=0.00210, cut_angle=45., name='alpha_bbo', nx=nx, ny=ny,
                           pixel_size=pixel_size, orientation=135., two_dimensional=True)
        delay_plate = Crystal(wavelength=wavelength[i], thickness=0.015, cut_angle=0., name='alpha_bbo', nx=nx,
                              ny=ny,
                              pixel_size=pixel_size, orientation=90., two_dimensional=True)
        displacer_plate = Crystal(wavelength=wavelength[i], thickness=0.003, cut_angle=45., name='alpha_bbo', nx=nx,
                                  ny=ny, pixel_size=pixel_size, orientation=90., two_dimensional=True)

        S0_interp = interp2d(x_small, y_small, S0_small[:, :, i], kind='cubic')
        S0 = S0_interp(x, y)

        S1_interp = interp2d(x_small, y_small, S1_small[:, :, i], kind='cubic')
        S1 = S1_interp(x, y)

        S2_interp = interp2d(x_small, y_small, S2_small[:, :, i], kind='cubic')
        S2 = S2_interp(x, y)

        if circular == True:
            S3_interp = interp2d(x_small, y_small, S3_small[:, :, i], kind='cubic')
            S3 = S3_interp(x, y)

            S_total[:, :, i] = 2 * S0 + 2 * S2 * np.cos(delay_plate.phi + displacer_plate.phi) + S1 * (
                    np.cos(displacer_plate.phi + delay_plate.phi + savart_1.phi - savart_2.phi) - np.cos(
                delay_plate.phi + displacer_plate.phi - savart_1.phi + savart_2.phi)) - S3 * (np.sin(
                displacer_plate.phi + delay_plate.phi + savart_1.phi - savart_2.phi) + np.sin(
                displacer_plate.phi + delay_plate.phi - savart_1.phi + savart_2.phi))
        else:
            S_total[:, :, i] = 2 * S0 + 2 * S2 * np.cos(delay_plate.phi + displacer_plate.phi) + S1 * (
                    np.cos(displacer_plate.phi + delay_plate.phi + savart_1.phi - savart_2.phi) - np.cos(
                delay_plate.phi + displacer_plate.phi - savart_1.phi + savart_2.phi))

    print('Generating image...')
    ash_image = np.sum(S_total, axis=2)

    return ash_image

def calculate_field_widened_TSH(FLC_state):

    """
    Calculate intensity on camera for an FLC system, including a field widened delay plate (Two thin delay plates with half waveplate in between. Gives zero net delay, but gives shear delay
    across crystal and reduces higher order effects that manifest as curved interference fringes.
    :param FLC_state: 1 or 2
    :return:
    """

    wavelength, nx, ny, pixel_size, x_small, y_small, S0_small, S1_small, S2_small, S3_small, x, y = get_msesim_output()

    S_total = np.zeros((len(x), len(y), len(wavelength)))

    print('Calculating TSH synthetic image...')

    for i in range(len(wavelength)):
        #Make the delay and displacer plate to find the phase shift for a given wavelength

        delay_plate_1 = Crystal(wavelength=wavelength[i], thickness=0.007, cut_angle=0., name='alpha_bbo', nx=nx, ny=ny, pixel_size=pixel_size, orientation=0., two_dimensional=True)
        delay_plate_2 = Crystal(wavelength=wavelength[i], thickness=0.007, cut_angle=0., name='alpha_bbo', nx=nx, ny=ny, pixel_size=pixel_size, orientation=90., two_dimensional=True)
        displacer_plate = Crystal(wavelength=wavelength[i], thickness=0.003, cut_angle=45., name='alpha_bbo', nx=nx, ny=ny, pixel_size=pixel_size, orientation=90., two_dimensional=True)

        #Interpolate our small msesim grid to the real image size, 1024x1024

        S0_interp = interp2d(x_small, y_small, S0_small[:, :, i], kind='quintic')
        S0 = S0_interp(x,y)

        S1_interp = interp2d(x_small, y_small, S1_small[:, :, i], kind='quintic')
        S1 = S1_interp(x,y)

        S2_interp = interp2d(x_small, y_small, S2_small[:, :, i], kind='quintic')
        S2 = S2_interp(x,y)

        #Calculate the total intensity for a given wavelength, propagating stokes components through a delay plate and a displacer plate.

        if FLC_state == 1:
            print('FLC state is 1')
            S_total[:,:,i] = S0 + S1*np.sin((delay_plate_1.phi_total + delay_plate_2.phi_total + displacer_plate.phi_total)) - S2*np.cos((delay_plate_1.phi_total + delay_plate_2.phi_total + displacer_plate.phi_total))

        else:
            print('FLC state is 2')
            S_total[:,:,i] = S0 - S1*np.sin((delay_plate_1.phi_total + delay_plate_2.phi_total + displacer_plate.phi_total)) + S2*np.cos((delay_plate_1.phi_total + delay_plate_2.phi_total + displacer_plate.phi_total))

    tsh_fw_image = np.sum(S_total, axis=2)

    return tsh_fw_image

def calculate_ideal_TSH(FLC_state):

    """
    Calculate intensity on the image using "cheat" equations ie. already derived using assumptions for the phase.
    :param FLC_state: 1 or 2
    :return:
    """

    wavelength, nx, ny, pixel_size, x_small, y_small, S0_small, S1_small, S2_small, S3_small, x, y = get_msesim_output()

    S_total = np.zeros((len(x), len(y), len(wavelength)))

    print('Calculating TSH synthetic image...')

    for i in range(len(wavelength)):
        #Make the delay and displacer plate to find the phase shift for a given wavelength

        delay_plate = Crystal(wavelength=wavelength[i], thickness=0.015, cut_angle=0., name='alpha_bbo', nx=nx, ny=ny, pixel_size=pixel_size, orientation=90., two_dimensional=True)
        displacer_plate = Crystal(wavelength=wavelength[i], thickness=0.003, cut_angle=45., name='alpha_bbo', nx=nx, ny=ny, pixel_size=pixel_size, orientation=90.,two_dimensional=True )

        #Interpolate our small 200x200 msesim grid to the real image size, 1024x1024

        S0_interp = interp2d(x_small, y_small, S0_small[:, :, i], kind='quintic')
        S0 = S0_interp(x,y)

        S1_interp = interp2d(x_small, y_small, S1_small[:, :, i], kind='quintic')
        S1 = S1_interp(x,y)

        S2_interp = interp2d(x_small, y_small, S2_small[:, :, i], kind='quintic')
        S2 = S2_interp(x,y)

        #Calculate the total intensity for a given wavelength, propagating stokes components through a delay plate and a displacer plate.

        if FLC_state == 1:
            print('FLC state is 1')
            S_total[:,:,i] = np.exp(1j * (delay_plate.phi_total + displacer_plate.phi_total)) * (S1 + (S2/1j))

        else:
            print('FLC state is 2')
            S_total[:,:,i] = np.exp(1j * (delay_plate.phi_total + displacer_plate.phi_total)) * (S1 - (S2/1j))

    ideal_image = np.sum(S_total, axis=2)

    return ideal_image

def save_image(image, filename):

    #save image as a HDF file

    x = pd.HDFStore(filename)

    x.append("a", pd.DataFrame(image))
    x.close()

    return

def make_TSH_1():
    image_1 = calculate_TSH_image(FLC_state=1)
    print('Made image 1! Saving...')
    save_image(image_1, filename="synthetic_image1_32x32.hdf")

def make_TSH_2():
    print('Making image 2...')
    image_2 = calculate_TSH_image(FLC_state=2)
    print('Image 2 complete! Saving...')
    save_image(image_2, filename="synthetic_image2_32x32.hdf")

def make_ASH_images():

    print('making first image')
    ash_nos3 = calculate_ASH_image(circular=False)
    print('saving image')
    save_image(ash_nos3, filename='ash_image_noS3_savart4mm.hdf')

    #Include circular polarisation S3 term. This gives an imbalance in the amplitude of the sum and difference carriers, which carries through as an error on the phase
    print('making second image')
    ash_s3 = calculate_ASH_image(circular=True)
    print('saving image')
    save_image(ash_s3, filename='ash_image_S34mm.hdf')
    return

def make_field_widened_TSH_1():
    print('Making image 1...')
    image_1 = calculate_field_widened_TSH(FLC_state=1)
    print('Made image 1! Saving...')
    save_image(image_1, filename="image_FLC1_field_widened.hdf")

def make_field_widened_TSH_2():
    print('Making image 2...')
    image_2 = calculate_field_widened_TSH(FLC_state=2)
    print('Made image 1! Saving...')
    save_image(image_2, filename="image_FLC2_field_widened.hdf")

def make_ideal_TSH_1():
    print('Making image 1...')
    ideal_1 = calculate_ideal_TSH(FLC_state=1)
    print('Made image 1! Saving...')
    save_image(ideal_1, filename="image_FLC1_ideal.hdf")

def make_ideal_TSH_2():
    print('Making image 2...')
    ideal_2 = calculate_ideal_TSH(FLC_state=2)
    print('Made image 2! Saving...')
    save_image(ideal_2, filename="image_FLC2_ideal.hdf")


make_TSH_1()
make_TSH_2()



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



