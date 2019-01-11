import idlbridge as idl
import numpy as np
from scipy.interpolate import interp2d
import pandas as pd
import matplotlib.pyplot as plt

from Tools.load_msesim import MSESIM
from Model.scratch.Crystal import Crystal
from Model.Observer import Camera

class Image(object):

    def __init__(self):
        self.px = np.arange(-1*camera.sensor_size/2, (camera.sensor_size/2)+camera.pixel_size_mm, camera.pixel_size_mm)
        self.py = np.arange(-1*camera.sensor_size/2, camera.sensor_size/2+camera.pixel_size_mm, camera.pixel_size_mm)

        print(len(self.px))

    def simulate_ideal_TSSH(self, FLC_state):

        """
        Calculate the intensity of an image for the imaging MSE system with a switching FLC as a function of wavelength.
        :param FLC_state: 1 or 2, switches between 45 degrees and 90 degrees polarisation state
        :param Ideal: True or False. True uses assumptions to reduce the phase shift equation to an ideal system ie. don't include higher order effects. False means the phase shift is calculated
        from the whole equation in the Veries paper. Better as it doesn't use any assumptions, but also tricky to demodulate and get a good "reference" demodulation...
        :return: Image intensity (photons/s)
        """

        S_total = np.zeros((len(self.px), len(self.py), len(msesim.wavelength)))

        print('Calculating TSH synthetic image...')

        for i in range(len(msesim.wavelength)):
            # Make the delay and displacer plate to find the phase shift for a given wavelength

            delay_plate = Crystal(wavelength=msesim.wavelength[i], thickness=0.015, cut_angle=0., name='alpha_bbo', nx=len(self.px),
                                  ny=len(self.py), pixel_size=camera.pixel_size, orientation=90., two_dimensional=True)

            displacer_plate = Crystal(wavelength=msesim.wavelength[i], thickness=0.003, cut_angle=45., name='alpha_bbo', nx=len(self.px),
                                      ny=len(self.py), pixel_size=camera.pixel_size, orientation=90., two_dimensional=True)

            # Interpolate our small msesim grid to the real image size, 1024x1024

            S0_interp = interp2d(msesim.x_coords, msesim.y_coords, msesim.S0[:, :, i], kind='linear')
            S0 = S0_interp(self.px, self.py)

            S1_interp = interp2d(msesim.x_coords, msesim.y_coords, msesim.S1[:, :, i], kind='linear')
            S1 = S1_interp(self.px, self.py)

            S2_interp = interp2d(msesim.x_coords, msesim.y_coords, msesim.S2[:, :, i], kind='linear')
            S2 = S2_interp(self.px, self.py)

            # Calculate the total intensity for a given msesim.wavelength, propagating stokes components through a delay plate and a displacer plate.

            print('Generating *ideal* TSSH image!')
            if FLC_state == 1:
                print('FLC state is 1')
                S_total[:, :, i] = S0 + S1 * np.sin((delay_plate.phi_0 + delay_plate.phi_shear + displacer_plate.phi_0 + displacer_plate.phi_shear)) + S2 * np.cos(
                    (delay_plate.phi_0 + delay_plate.phi_shear + displacer_plate.phi_0 + displacer_plate.phi_shear))

            elif FLC_state == 2:
                print('FLC state is 2')
                S_total[:, :, i] = S0 + S1 * np.sin((delay_plate.phi_0 + delay_plate.phi_shear + displacer_plate.phi_0 + displacer_plate.phi_shear)) - S2 * np.cos(
                    (delay_plate.phi_0 + delay_plate.phi_shear + displacer_plate.phi_0 + displacer_plate.phi_shear))
            else:
                print('FLC_state must be either 1 or 2!')

        tsh_image = np.sum(S_total,axis=2)

        return tsh_image

    def simulate_non_ideal_TSSH(self, FLC_state):

        S_total = np.zeros((len(self.px), len(self.py), len(msesim.wavelength)))

        print('Calculating non-ideal TSH synthetic image...')

        for i in range(len(msesim.wavelength)):
            # Make the delay and displacer plate to find the phase shift for a given wavelength

            delay_plate = Crystal(wavelength=msesim.wavelength[i], thickness=0.015, cut_angle=0., name='alpha_bbo',
                                  nx=len(self.px),
                                  ny=len(self.py), pixel_size=camera.pixel_size, orientation=90., two_dimensional=True)

            displacer_plate = Crystal(wavelength=msesim.wavelength[i], thickness=0.003, cut_angle=45., name='alpha_bbo',
                                      nx=len(self.px),
                                      ny=len(self.py), pixel_size=camera.pixel_size, orientation=90.,
                                      two_dimensional=True)

            # Interpolate our small msesim grid to the real image size, 1024x1024

            S0_interp = interp2d(msesim.x_coords, msesim.y_coords, msesim.S0[:, :, i], kind='linear')
            S0 = S0_interp(self.px, self.py)

            S1_interp = interp2d(msesim.x_coords, msesim.y_coords, msesim.S1[:, :, i], kind='linear')
            S1 = S1_interp(self.px, self.py)

            S2_interp = interp2d(msesim.x_coords, msesim.y_coords, msesim.S2[:, :, i], kind='linear')
            S2 = S2_interp(self.px, self.py)

            print('Generating non-ideal TSSH image!')
            if FLC_state == 1:
                print('FLC state is 1')
                S_total[:, :, i] = S0 + S1 * np.sin((delay_plate.phi_total + delay_plate.phi_total + displacer_plate.phi_total + displacer_plate.phi_total)) + S2 * np.cos(
                    (delay_plate.phi_total + delay_plate.phi_total + displacer_plate.phi_total + displacer_plate.phi_total))

            elif FLC_state == 2:
                print('FLC state is 2')
                S_total[:, :, i] = S0 + S1 * np.sin((delay_plate.phi_total + delay_plate.phi_total + displacer_plate.phi_total + displacer_plate.phi_total)) - S2 * np.cos(
                    (delay_plate.phi_total + delay_plate.phi_total + displacer_plate.phi_total + displacer_plate.phi_total))
            else:
                print('FLC_state must be either 1 or 2!')

        tsh_image = np.sum(S_total, axis=2)

        return tsh_image


    def simulate_field_widened_TSSH(self, FLC_state):

        """
        Calculate intensity on camera for an FLC system, including a field widened delay plate (Two thin delay plates with half waveplate in between. Gives zero net delay, but gives shear delay
        across crystal and reduces higher order effects that manifest as curved interference fringes.
        :param FLC_state: 1 or 2
        :return:
        """

        S_total = np.zeros((len(self.px), len(self.py), len(msesim.wavelength)))

        print('Calculating TSH synthetic image...')

        for i in range(len(msesim.wavelength)):
            # Make the delay and displacer plate to find the phase shift for a wavelength

            delay_plate_1 = Crystal(wavelength=msesim.wavelength[i], thickness=0.007, cut_angle=0., name='alpha_bbo', nx=len(self.px),
                                    ny=len(self.py), pixel_size=camera.pixel_size, orientation=0., two_dimensional=True)
            delay_plate_2 = Crystal(wavelength=msesim.wavelength[i], thickness=0.007, cut_angle=0., name='alpha_bbo', nx=len(self.px),
                                    ny=len(self.py), pixel_size=camera.pixel_size, orientation=90., two_dimensional=True)
            displacer_plate = Crystal(wavelength=msesim.wavelength[i], thickness=0.003, cut_angle=45., name='alpha_bbo', nx=len(self.px),
                                      ny=len(self.py), pixel_size=camera.pixel_size, orientation=90., two_dimensional=True)

            # Interpolate our small msesim grid to the real image size, 1024x1024

            S0_interp = interp2d(msesim.x_coords, msesim.y_coords, msesim.S0[:, :, i], kind='linear')
            S0 = S0_interp(self.px, self.py)

            S1_interp = interp2d(msesim.x_coords, msesim.y_coords, msesim.S1[:, :, i], kind='linear')
            S1 = S1_interp(self.px, self.py)

            S2_interp = interp2d(msesim.x_coords, msesim.y_coords, msesim.S2[:, :, i], kind='linear')
            S2 = S2_interp(self.px, self.py)

            # Calculate the total intensity for a given msesim.wavelength, propagating stokes components through a delay plate and a displacer plate.

            if FLC_state == 1:
                print('FLC state is 1')
                S_total[:, :, i] = S0 + S1 * np.sin((delay_plate_1.phi_total + delay_plate_2.phi_total + displacer_plate.phi_total)) - S2 * np.cos((delay_plate_1.phi_total + delay_plate_2.phi_total + displacer_plate.phi_total))

            else:
                print('FLC state is 2')
                S_total[:, :, i] = S0 - S1 * np.sin((delay_plate_1.phi_total + delay_plate_2.phi_total + displacer_plate.phi_total)) + S2 * np.cos((delay_plate_1.phi_total + delay_plate_2.phi_total + displacer_plate.phi_total))

        tsh_fw_image = np.sum(S_total, axis=2)

        return tsh_fw_image

    def simulate_ASH_non_ideal(self, circular):
        """
        Calculate the intensity of the image for an amplitude spatial heterodyne system
        :param circular: True/False - Include S3 component to realistically model effect of S3 on carrier amplitudes.
        :return:
        """

        S_total = np.zeros((len(self.px), len(self.py), len(msesim.wavelength)))

        for i in range(len(msesim.wavelength)):
            savart_1 = Crystal(wavelength=msesim.wavelength[i], thickness=0.00210, cut_angle=45., name='alpha_bbo', nx=len(self.px),
                               ny=len(self.py),
                               pixel_size=camera.pixel_size, orientation=45., two_dimensional=True)
            savart_2 = Crystal(wavelength=msesim.wavelength[i], thickness=0.00210, cut_angle=45., name='alpha_bbo', nx=len(self.px),
                               ny=len(self.py),
                               pixel_size=camera.pixel_size, orientation=135., two_dimensional=True)
            delay_plate = Crystal(wavelength=msesim.wavelength[i], thickness=0.015, cut_angle=0., name='alpha_bbo', nx=len(self.px),
                                  ny=len(self.px),
                                  pixel_size=camera.pixel_size, orientation=90., two_dimensional=True)
            displacer_plate = Crystal(wavelength=msesim.wavelength[i], thickness=0.003, cut_angle=45., name='alpha_bbo', nx=len(self.px),
                                      ny=len(self.px), pixel_size=camera.pixel_size, orientation=90., two_dimensional=True)

            S0_interp = interp2d(msesim.x_coords, msesim.y_coords, msesim.S0[:, :, i], kind='linear')
            S0 = S0_interp(self.px, self.py)

            S1_interp = interp2d(msesim.x_coords, msesim.y_coords, msesim.S1[:, :, i], kind='linear')
            S1 = S1_interp(self.px, self.py)

            S2_interp = interp2d(msesim.x_coords, msesim.y_coords, msesim.S2[:, :, i], kind='linear')
            S2 = S2_interp(self.px, self.py)

            if circular == True:
                S3_interp = interp2d(msesim.x_coords, msesim.y_coords, msesim.S3[:, :, i], kind='linear')
                S3 = S3_interp(self.px, self.py)

                S_total[:, :, i] = 2 * S0 + 2 * S2 * np.cos(delay_plate.phi_total + displacer_plate.phi_total) + S1 * (
                        np.cos(displacer_plate.phi_total + delay_plate.phi_total + savart_1.phi_total - savart_2.phi_total) - np.cos(
                    delay_plate.phi_total + displacer_plate.phi_total - savart_1.phi_total + savart_2.phi_total)) - S3 * (np.sin(
                    displacer_plate.phi_total + delay_plate.phi_total + savart_1.phi_total - savart_2.phi_total) + np.sin(
                    displacer_plate.phi_total + delay_plate.phi_total - savart_1.phi_total + savart_2.phi_total))
            else:
                S_total[:, :, i] = 2 * S0 + 2 * S2 * np.cos(delay_plate.phi_total + displacer_plate.phi_total) + S1 * (
                        np.cos(displacer_plate.phi_total + delay_plate.phi_total + savart_1.phi_total - savart_2.phi_total) - np.cos(
                    delay_plate.phi_total + displacer_plate.phi_total - savart_1.phi_total + savart_2.phi_total))

        print('Generating image...')
        ash_image = np.sum(S_total, axis=2)

        return ash_image

    def save(self, image, filename):

        # save image as a HDF file

        x = pd.HDFStore(filename)

        x.append("a", pd.DataFrame(image))
        x.close()

        return

    def load(self, filename):

        image_file = pd.HDFStore(filename)
        print(image_file)
        image = image_file['/a']

        return image

    def plot(self, image):

        plt.figure(1)
        plt.imshow(image, cmap='gray')
        plt.gca().invert_xaxis()
        plt.xlabel('x')
        plt.ylabel('y')
        cbar = plt.colorbar()
        cbar.set_label('Intensity ($\gamma$/s)', rotation=90.)
        plt.show()

        return

idl.execute("restore, '/home/sgibson/PycharmProjects/msesim/runs/imse_2d_32x32_f80mm/output/data/MAST_18501_imse.dat', /VERBOSE")
camera = Camera(photron=True)
msesim = MSESIM(nx=32,ny=32)

image = Image()

im1 = image.simulate_ideal_TSSH(FLC_state=1)
# image.save(im1, filename=filename)
# ideal_image1 = image.load(filename='/home/sgibson/PycharmProjects/IMSE/Model/TSSH_ideal1.hdf')
# image.plot(ideal_image1)


