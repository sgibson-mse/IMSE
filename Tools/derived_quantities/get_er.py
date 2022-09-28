import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

def load_er(filename):
    data = np.loadtxt(filename)
    Er_ds = xr.Dataset( {'rho': ('rho', data[:,0]),
                         'v_perp': ('rho', data[:,1]),
                         'v_toroidal': ('rho', data[:,2]),
                         'Er': ('rho', data[:,3])})
    return Er_ds

filename = '/home/sfreethy/code/logbook/Er profile calculations/M9/30166_200.dat'
Er_ds = load_er(filename)

fig, (ax1, ax2, ax3) = plt.subplots(3)
Er_ds['Er'].plot(ax=ax1)
Er_ds['v_perp'].plot(ax=ax2)
Er_ds['v_toroidal'].plot(ax=ax3)
plt.show()

