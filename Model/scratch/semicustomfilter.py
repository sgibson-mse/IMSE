import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt

#Joe Allcock's snazzy filter class

class SemiCustomFilter(object):

    def __init__(self, centre, fwhm, peak_tx, type, tilt_angle):
        """ set the normalised transmission values, and the corresponding fwhm coefficients based on info from:
        https://www.andovercorp.com/technical/bandpass-filter-fundamentals/filter-types/

        :param centre: centre wavelength [nm]
        :param fwhm: full width half max. wavelength [nm]
        :param peak_tx: transmission at centre, as a fraction.
        :param type: type of semi-custom filter, according to Andover, basically, number of cavities.
        :param tilt_angle: [deg]
        """

        self.centre = centre
        self.fwhm = fwhm
        self.peak_tx = peak_tx
        self.type = type
        self.tilt_angle = tilt_angle

        self.name = 'centre_' + str(centre) + 'nm_fwhm_' + str(fwhm) + 'nm_type_' + str(type)
        self.n = 2  # refractive index

        # account for filter tilt
        self.tilt_angle_rad = self.tilt_angle * (np.pi / 180)

        self.centre = centre * np.sqrt(1 - (np.sin(self.tilt_angle_rad) / self.n) ** 2)

        self._norm_transmission_values = np.array([1.e-5, 1.e-4, 1.e-3, 1.e-2, 1.e-1, 5.e-1, 9.e-1]) * self.peak_tx

        if type == 2:
            self.fwhm_coeff = np.array([45, 15, 6.3, 3.5, 2, 1, 0.5])
        elif type == 3:
            self.fwhm_coeff = np.array([15, 5.4, 3.2, 2.2, 1.5, 1, 0.65])
        elif type == 4:
            self.fwhm_coeff = np.array([12, 4.25, 2.25, 1.8, 1.3, 1, 0.8])
        else: raise Exception("'type' must be 2, 3 or 4.")

        self.wl_axis, self.tx = self.get_raw_profile()
        self.wl_axis, self.tx = self.get_interp_profile()

    def get_raw_profile(self):
        """ """

        norm_transmission = np.concatenate([self._norm_transmission_values, [1 * self.peak_tx], self._norm_transmission_values[::-1]])
        wl_axis = np.zeros_like(norm_transmission)

        for i in range(0, len(self._norm_transmission_values)):
            half_width = (self.fwhm_coeff[i] * self.fwhm) / 2
            wl_axis[i] = self.centre - half_width
            wl_axis[-(i + 1)] = self.centre + half_width

        wl_axis[len(self._norm_transmission_values)] = self.centre

        return wl_axis, norm_transmission

    def add_to_plot(self, ax,  **kwargs):
        """ """

        ax.plot(self.wl_axis, self.tx, label=self.tilt_angle, **kwargs)
        return

    def get_interp_profile(self):
        """ interpolate the few points provided by the manufacturer onto a finer wavelength grid."""

        no_fwhm = 10
        wl_lo = self.centre - (no_fwhm * self.fwhm)
        wl_hi = self.centre + (no_fwhm * self.fwhm)

        interp_wavelength_axis = np.linspace(wl_lo, wl_hi, 1000)

        f = scipy.interpolate.InterpolatedUnivariateSpline(self.wl_axis, self.tx, k=1, ext='zeros')
        interp_transmission = f(interp_wavelength_axis)

        return interp_wavelength_axis, interp_transmission