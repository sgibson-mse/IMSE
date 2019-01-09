from Tools.load_msesim import MSESIM
import numpy as np
import xarray as xr
from scipy.interpolate import RectBivariateSpline

class Light():

    def __init__(self, filepath):

        msesim = MSESIM(filepath=str(filepath))

        self.stokes_vector = msesim.stokes_vector
        self.S0 = msesim.S0
        self.S1 = msesim.S1
        self.S2 = msesim.S2
        self.S3 = msesim.S3

        self.wavelength = msesim.wavelength*10**-9

    def project(self, lens, camera):

        #Find out where the light will focus onto the sensor.
        #We need to do this first as the shift between ordinary/extraordinary rays exiting the crystal depends on
        #the incident angle when the optical axis is not parallel to the plate surface.

        self.x, self.y = np.meshgrid(camera.x, camera.y)

        self.alpha = np.arctan(np.sqrt(self.x ** 2 + self.y ** 2) / lens.focal_length)

        self.beta = np.arctan2(self.y, self.x)

        return self.alpha, self.beta

    def interact(self, phi, S0, S1, S2):
        """Find the output stokes vector after light has interacted with the crystal."""
        return S0 + S1*np.sin(phi) + S2*np.cos(phi)

if __name__ == "__main__":

    light = Light(filepath='/work/sgibson/msesim/runs/imse_2d_32x32_magneticsequi_edgecurrent/output/data/MAST_24763_imse.dat')

    from Model.Optic import Lens, Crystal
    from Model.Observer import Camera

    def retard(delay, displacer, light):

        s_out = []

        #Calculate the birefringence and phase delay for a given crystal

        for i in range(len(light.wavelength)):

            interp_S0 = RectBivariateSpline(camera.x[0::32], camera.y[0::32], light.S0[:,:,i], kx = 3, ky=3)
            interp_S1 = RectBivariateSpline(camera.x[0::32], camera.y[0::32], light.S1[:,:,i], kx = 3, ky=3)
            interp_S2 = RectBivariateSpline(camera.x[0::32], camera.y[0::32], light.S2[:,:,i], kx = 3, ky=3)

            S0 = interp_S0(camera.x, camera.y)
            S1 = interp_S1(camera.x, camera.y)
            S2 = interp_S2(camera.x, camera.y)

            delay.calculate_birefringence(light.wavelength[i])
            delay_shift = delay.get_phase(light.alpha, light.beta, light.wavelength[i])

            displacer.calculate_birefringence(light.wavelength[i])
            displacer_shift = delay.get_phase(light.alpha, light.beta, light.wavelength[i])

            S0_out = light.interact(delay_shift+displacer_shift, S0, S1, S2)

            s_out.append(xr.Dataset({'S0': (['y', 'x'], S0_out)}))

        s_out = xr.concat(s_out, dim='wavelength')

        return s_out

    camera = Camera(name='photron-sa4')

    focal_length = 85*10**-3

    delay_thickness = 5*10**-3
    delay_cut_angle = 0.
    delay_orientation=90.
    delay_material='alpha_bbo'

    displacer_thickness = 3*10**-3
    displacer_cut_angle = 45.
    displacer_orientation = 90.
    displacer_material='lithium_niobate'

    lens = Lens(focal_length, aperture=None, diameter=None, f=None)
    light.project(lens, camera)

    delay = Crystal(delay_thickness, delay_cut_angle, delay_orientation, delay_material)
    displacer = Crystal(displacer_thickness, displacer_cut_angle, displacer_orientation, displacer_material)

    s_out = retard(delay, displacer, light)



    #
    # phi_displacer = light.retard(displacer)
    #
    # phi = phi_delay + phi_displacer
    #
    # #FLC in state 1
    # s_out = light.interact(phi)
    # camera.observe(s_out)

    #FLC in state 2
    #light.interact(np.pi - phi)




# light.interact(crystal)
# light.interact(polarizer)
#polariser = Polarizer(theta=45.)
#crystal = Crystal('params in here')




        #np.matmul(self.stokes_vector, component.matrix)

        #interact with an optical component
        #optical component has: a name, and an associated muller matrix





# # Calculate emission - MSESIM
#
# # Set up experiment
# source = Source()
#
# crystal1 = Crystal(pos, type)
# camera = Camera(pos, )
#
# # Produce light at source
# light = source.emit(data='./MSESIM.nc') # -> Read MSESIM output Stokes vector
#
# # Pass light through crystal
# light = crystal1.refract(light)
#
# # Pass light through crystal2
#
#
# # Detect light on camera
# camera.observe() #sum along wavelength


# class Source:
#
#     """Class defines a source object - in this case it's light output from MSESIM. In The future this will be a source from cherab..."""
#
#     def __init__(self, filepath):
#
#         self.load_file = self.load_file(filepath)
#
#     def load_file(self, filepath):
#         msesim = MSESIM(filepath=str(filepath))
#         return msesim