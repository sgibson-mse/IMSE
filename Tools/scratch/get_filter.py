from Model.Constants import Constants
from Model.Observer import Camera
from Model.scratch.semicustomfilter import SemiCustomFilter

import matplotlib.pyplot as plt
import idlbridge as idl
import numpy as np
import pyuda

from Tools.MSESIM import MSESIM


from scipy.interpolate import interp1d

def non_axial_rays(focal_length):

    x = np.linspace(-camera.sensor_size / 2, camera.sensor_size / 2, 32)
    y = np.linspace(-camera.sensor_size / 2, camera.sensor_size / 2, 32)

    xx,yy = np.meshgrid(x,y)


    alpha = np.arctan(np.sqrt((xx) ** 2 + (yy) ** 2) / (focal_length))

    beta = np.arctan2(yy, xx)

    return alpha, beta

client = pyuda.Client()

nx, ny = 32, 32
idl.execute("restore, '/home/sgibson/PycharmProjects/msesim/runs/imse_2d_32x32_f80mm_opticaxis/output/data/MAST_18501_imse.dat', /VERBOSE")
msesim = MSESIM(nx, ny)
data = msesim.load_msesim_spectrum()
constants = Constants()
camera = Camera(photron=True)

alpha, beta = non_axial_rays(focal_length=85*10**-3)

alpha = alpha * (180./np.pi)

vals = np.arange(0, 30, 1)

filter_2d = []
filter_2d_wl = []

tx_interp = []

alpha_mid = alpha[int(len(alpha)/2),:]

for i in range(len(alpha_mid)):
    filter = SemiCustomFilter(660.3, 3, 0.7, 3, tilt_angle=alpha_mid[i])
    tx_finterp = interp1d(filter.wl_axis, filter.tx)
    tx_interp.append(tx_finterp(msesim.wavelength/10))

levels = np.arange(0,1,0.1)
spectrum = np.sqrt(msesim.S1**2 + msesim.S2**2)
spectrum = spectrum/np.max(spectrum)

wl, rr, = np.meshgrid(msesim.wavelength/10, msesim.major_radius)

plt.figure()
CS1 = plt.pcolormesh(wl,rr,spectrum[16,:,:]/np.max(spectrum[16,:,:]), shading='gourand', rasterized=True)
CS = plt.contour(wl, rr, tx_interp, colors='white', levels=levels)
cbar = plt.colorbar(CS1)
plt.clabel(CS, inline=True, fontsize=14, inline_spacing=10, manual=True)
cbar.ax.set_ylabel('Intensity (Arb. Units)')
cbar.add_lines(CS)
plt.xlabel('Wavelength $\lambda$ (nm)')
plt.ylabel('R (m)')
plt.show()