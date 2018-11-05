import numpy as np
import idlbridge as idl
import matplotlib.pyplot as plt

from Tools.Plotting.graph_format import plot_format
from Tools.calculate_1d_contrast import get_stokes_components, find_optimal_crystal, calc_fringe_frequency, angle_of_incidence, design_filter

plot_format()

# FULL ENERGY COMPONENT
idl.execute(
    "restore, '/home/sam/Desktop/msesim/runs/mast_imse_photron/output/data/density2e19m3_MAST_photron.dat', /VERBOSE")

delay_thickness = 15*10**-3 #np.arange(3*10**-3, 23*10**-3, 0.1*10**-3)
delay_cut_angle = 0.
displacer_cut_angle = np.arange(0,90.,1.)
displacer_thickness = 3*10**-3
tilt_angle = 0.
scale_fwhm = 3.5
lambda_0 = 660.5

major_radius, stokes, wavelength_vector, linearly_polarised = get_stokes_components()

wavelength_nm = wavelength_vector/10

contrasts, total_polarised, phi_total = find_optimal_crystal(wavelength_nm, stokes, delay_thickness, delay_cut_angle, displacer_thickness, displacer_cut_angle)

pixels_per_fringe = calc_fringe_frequency(displacer_cut_angle)

alpha, beta, pixel_x = angle_of_incidence()

transmittance = design_filter(wavelength_nm, alpha, tilt_angle, scale_fwhm, lambda_0)

class System:

    def __init__(self):
        self.R = major_radius                           # Major radii
        self.S = stokes                                 # Full energy stokes components
        self.wavelength = wavelength_vector/10          # Wavelength vector
        self.I_linear = linearly_polarised              # Intensity of linearly polarised emission
        self.contrast = contrasts                       # Contrast (ie. ratio of polarised to total emission intensity)
        self.I_total_polarised = total_polarised        # Intensity of polarised light
        self.delay = phi_total                          # Phase shift in number of waves, as a function of wavelength
        self.delay_thickness = delay_thickness          # Thickness of the delay plate
        self.delay_cut_angle = delay_cut_angle          # Cut angle of the delay plate
        self.displacer_cut_angle = displacer_cut_angle  # Cut angle of the displacer plate
        self.displacer_thickness = displacer_thickness  # Thickness of the displacer plate
        self.pixels_per_fringe = pixels_per_fringe      # Number of pixels per fringe, for a given focal length lens
        self.alpha = alpha                              # Angle of incidence on the sensor
        self.beta = beta                                # Polar co-ordinate to determine position on sensor
        self.x_pixel = pixel_x                          # Position on sensor in mm (across horizontal)
        self.transmittance = transmittance              # Transmittance function for the filter

    def plot_contrast(self):

        pp, ll = np.meshgrid(self.delay_thickness, self.R)

        levels = np.arange(0, 0.43, 0.05)

        plt.figure()
        plt.title('Displacer plate thickness fixed {} mm'.format(displacer_thickness*1000))
        CS = plt.pcolormesh(ll, pp*1000, self.contrast*100, shading='gouraud')
        #plt.contour(ll, pp*1000, self.contrast, colors='k', levels=levels)
        cbar = plt.colorbar(CS)
        cbar.set_label('Contrast [%]', rotation=90)
        plt.xlabel('Major Radius [m]')
        plt.ylabel('Delay plate thickness [mm]')
        plt.show()

        return

    def plot_delay(self):

        plt.figure()
        plt.title('15mm delay plate, 3mm displacer plate')
        plt.plot(self.R, self.contrast[:,109]*100)
        plt.xlabel('Major radius(m)')
        plt.ylabel('Contrast [%]')
        plt.show()

        return

    def plot_cut_angle(self):
        pp, ll = np.meshgrid(self.displacer_cut_angle, self.R)

        levels = np.arange(20,45, 5)
        plt.figure()
        plt.title('Displacer plate = 3mm, Delay = 15mm')
        CS = plt.pcolormesh(ll,pp,self.contrast*100, shading='gouraud')
        cbar = plt.colorbar(CS)
        cont = plt.contour(ll, pp, self.contrast*100, colors='black', levels=levels)
        plt.clabel(cont, inline=1, fontsize=24)
        cbar.set_label('Contrast', rotation=90)
        plt.xlabel('Major Radius [m]')
        plt.ylabel('Cut angle $\Theta$ [degrees]')
        plt.show()

        return

    def plot_fringe_frequency(self):
        'plot of pixels per fringe for a given focal length lens, varying displacer cut angle'

        plt.figure()
        plt.plot(self.displacer_cut_angle, self.pixels_per_fringe[:, 0], label='$f$ = 50mm')
        plt.plot(self.displacer_cut_angle, self.pixels_per_fringe[:, 1], label='$f$ = 85mm')
        plt.legend()
        plt.ylim(5, 20)
        plt.xlabel('Displacer cut angle [degrees]')
        plt.ylabel('Number of pixels per fringe')
        plt.legend()
        plt.ylim(5, 20)
        plt.xlabel('Displacer cut angle [degrees]')
        plt.show()

        return

    def plot_angle_of_incidence(self):

        plt.figure()
        plt.title('Angle of incidence of light to normal of the filter ')
        plt.plot(self.x_pixel, self.alpha * (180. / np.pi))
        plt.xlabel('Pixel position from center of the sensor [mm]')
        plt.ylabel('Angle of Incidence [deg]')
        plt.show()

        return

    def plot_filter_function(self):

        ll, rr = np.meshgrid(self.wavelength, self.R)

        plt.figure()
        cont = plt.contour(ll, rr, self.transmittance, cmap='gray')
        plt.show()

        return

    def plot_filtered_energy_component(self):

        I_polarised = np.sqrt(self.S[:,1,:]**2+self.S[:,2,:]**2)/np.max(abs(np.sqrt(self.S[:,1,:]**2+self.S[:,2,:]**2)))

        ll, rr = np.meshgrid(self.wavelength, self.R)

        plt.figure(1)
        cont = plt.contour(ll, rr, self.transmittance, cmap='gray')
        plt.clabel(cont, inline=1, fontsize=10)
        plt.pcolormesh(ll, rr, I_polarised)
        cs = plt.colorbar()
        cs.set_label('Intensity [Arb Units]', rotation=90.)
        plt.ylabel('Major radius [m]')
        plt.xlabel('Wavelength of spectrum [nm]')
        plt.show()

        return

system = System()

#system.plot_cut_angle()
#system.plot_fringe_frequency()
#system.plot_angle_of_incidence()
#system.plot_filter_function()
#system.plot_filtered_energy_component()