import numpy as np
from scipy.interpolate import RectBivariateSpline

from zoidberg_progress.zoidberg_progress import zoidberg_progress as zp

from Model.Light import Light
from Model.Optic import Lens, Crystal, Polarizer, Filter
from Model.Observer import Camera

def retard(delay, displacer, light, FLC):

    S_out = np.zeros((len(camera.x), len(camera.y), len(light.wavelength)))

    for i, wavelength in enumerate(light.wavelength):

        zp(i/len(light.wavelength), bar_length=40, ascii=False, pad=False, food='><>', woop=False)

        interp_S0 = RectBivariateSpline(camera.x[0::32], camera.y[0::32], light.S0[:,:,i], kx = 3, ky=3)
        interp_S1 = RectBivariateSpline(camera.x[0::32], camera.y[0::32], light.S1[:,:,i], kx = 3, ky=3)
        interp_S2 = RectBivariateSpline(camera.x[0::32], camera.y[0::32], light.S2[:,:,i], kx = 3, ky=3)

        S0 = interp_S0(camera.x, camera.y)
        S1 = interp_S1(camera.x, camera.y)
        S2 = interp_S2(camera.x, camera.y)

        delay.calculate_birefringence(light.wavelength[i])

        phase_delay = delay.get_phase(light.alpha, light.beta, light.wavelength[i])

        displacer.calculate_birefringence(light.wavelength[i])

        phase_displacer = displacer.get_phase(light.alpha, light.beta, light.wavelength[i])

        phase = phase_delay + phase_displacer


        #delay_phi_c, delay_phi_s, delay_phi_h = delay.approximate_phase_delay(light.wavelength[i], light.x, light.y, lens)

        #displacer_phi_c, displacer_phi_s, displacer_phi_h = displacer.approximate_phase_delay(light.wavelength[i], light.x, light.y, lens)

        #phase = delay_phi_c + delay_phi_s + delay_phi_h + displacer_phi_c + displacer_phi_s + displacer_phi_h

        if FLC==45:
            S_out[:,:,i] = light.interact(phase, S0, S1, S2)
        if FLC==90:
            S_out[:,:,i] = light.interact(phase-np.pi, S0, S1, S2)

    return S_out

def transmit(photons, lens, delay, displacer, polariser, filter):
    return photons*lens.transmission*lens.transmission*delay.transmission*displacer.transmission*lens.transmission*polariser.transmission*filter.transmission*lens.transmission

#Define some necessary values for the system

camera_name = 'photron-sa4'
camera_exposure = 1*10**-3

focal_length = 85*10**-3

delay_thickness = 15*10**-3
delay_cut_angle = 0.
delay_orientation= 90.
delay_material='alpha_bbo'

displacer_thickness = 3*10**-3
displacer_cut_angle = 45.
displacer_orientation = 90.
displacer_material='alpha_bbo'

polariser_orientation = 45.

filter_fwhm = 3*10**-9
filter_cwl = 660*10**-9

#Make an instance of the light class with MSESIM data
light = Light(filepath='/work/sgibson/msesim/runs/imse_2d_32x32_magneticsequi_edgecurrent/output/data/MAST_24763_imse.dat')

#Make a camera
camera = Camera(name=camera_name)

#Make a polariser
polariser = Polarizer(polariser_orientation)

#Make a filter
filter = Filter(fwhm=filter_fwhm, cwl=filter_cwl)

#Make a lens
lens = Lens(focal_length, aperture=None, diameter=None, f=None)

#Project the light onto the camera sensor
light.project(lens, camera)

#Make the crystals
delay = Crystal(delay_thickness, delay_cut_angle, delay_orientation, delay_material)
displacer = Crystal(displacer_thickness, displacer_cut_angle, displacer_orientation, displacer_material)

#Find the light intensity after passing through the crystals
S_out = retard(delay, displacer, light, FLC=45)

#Transmit light through the components to find the output number of photons
intensity = transmit(S_out, lens, delay, displacer, polariser, filter)

#Observe the light with the camera
image = camera.observe(intensity, camera_exposure, ideal=False)

#Digitize the image
digitized_image = camera.digitize(image)

#Plot the digitized image
camera.plot(digitized_image)
