import numpy as np
import matplotlib.pyplot as plt

from Tools.Plotting.graph_format import plot_format

plot_format()

class Camera(object):

    def __init__(self, photron):

        if photron:
            self.photron_specs()
        else:
            self.pco_specs()

    def photron_specs(self):
        self.n_pixels = 1024
        self.pixel_size = 20*10**-6 #m
        self.sensor_size = 20.48*10**-3 #mm
        self.dark_noise = 3.55 #electrons
        self.dark_noise_error = 0.02
        self.read_out = 41.2
        self.fill_factor = 0.52
        self.fullwell_capacity = 45000
        self.quantum_efficiency = 0.4
        self.gain = 0.0862

    def pco_specs(self):
        self.dark_noise = 2 #electrons
        self.read_out = 2.2 #electrons
        self.fill_factor = 1 # how much of the pixel is actually available for light collection
        self.fullwell_capacity = 30000 #electrons
        self.pixel_size = 6.5 #um
        self.gain = 0.46 #electrons/count
        self.quantum_efficiency = 0.54



# Calculate signal to noise ratio as a function of integration time for the PCO edge camera vs the photron sa-4.
#
# def signal_to_noise(intensity, read, dark, shot):
#     SN = intensity / np.sqrt(read**2 + dark + shot)
#     return SN
#
# sa4 = Camera(photron=True)
#
# I_photons = 4.33*10**7 #photons/second - minimum emitted intensity output from msesim (average = 4.34*10^7) (min = 1.22E+06)
# integration_time_sa4 = np.arange(1*10**-3, 20.5*10**-3, 0.5*10**-3)
# integration_time_pco = np.arange(5*10**-3, 20.5*10**-3, 0.5*10**-3)
#
# intensity_sa4 = I_photons * sa4.fill_factor * sa4.quantum_efficiency * integration_time_sa4 * sa4.gain
# shot_sa4 = I_photons * integration_time_sa4 * sa4.quantum_efficiency * sa4.gain**2
# dark_sa4 = sa4.dark_noise * integration_time_sa4
#
# SN_sa4 = signal_to_noise(intensity_sa4, sa4.read_out, sa4.dark_noise, shot_sa4)
#
# print(SN_sa4)
#
# plt.figure()
# plt.plot(integration_time_sa4*1000, SN_sa4, '--', label='sa4')
# plt.xlim(0,10)
# plt.xlabel('Integration time [ms]')
# plt.ylabel('Signal to Noise')
# plt.legend()
# plt.show()




# pco = Camera(photron=False)
# intensity_pco = I_photons * pco.fill_factor * pco.quantum_efficiency * integration_time_pco * pco.gain
# shot_pco = I_photons * integration_time_pco * pco.quantum_efficiency * pco.gain**2
# dark_pco = pco.dark_noise * integration_time_pco
#
# SN_pco = signal_to_noise(intensity_pco, pco.read_out, dark_pco, shot_pco)

