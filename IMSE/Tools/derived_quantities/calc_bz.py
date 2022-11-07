import pyuda
from scipy.interpolate import interp2d, interp1d
from scipy.io import readsav

import idlbridge as idl
import xarray as xr
import matplotlib.pyplot as plt

client = pyuda.Client()

import numpy as np




def create_gamma_dataset():

    wavelength = xr.DataArray(data=idl.get('lambda'), dims=('wavelength'), name='lambda')
    wavelength.attrs = {'units': 'Angstrom', 'long_name': 'Wavelength'}

    r_resolution = xr.DataArray(data=idl.get('r_res'), dims=('channel', 'nr'), name='r_resolution')
    r_resolution.attrs = {'units': 'm', 'long_name': 'Spatial resolution'}

    r_coords = r_resolution.isel(nr=2)

    r_coords = xr.DataArray(data=r_coords, name='r_coords')

    #Calculate the polarisation angle from the total stokes vector
    total_stokes = xr.DataArray(data=idl.get('stokes'), dims=('channel', 'stokes', 'wavelength'), name='total_stokes')

    polarisation_angle = 0.5*np.arctan(np.sum(total_stokes.isel(stokes=2),axis=1)/np.sum(total_stokes.isel(stokes=1),axis=1))

    polarisation_angle = xr.DataArray(data=polarisation_angle, dims=('channel'), name='gamma')

    gamma_dataset = xr.merge([wavelength, polarisation_angle, r_coords])

    gamma_dataset = gamma_dataset.assign_coords(channel=np.arange(40))

    return gamma_dataset


def create_dataset(fname):

    idl.execute("restore, '{0}' , /VERBOSE".format(fname))
    gamma_dataset = create_gamma_dataset()

    return gamma_dataset

def calc_current(fname, eq):

    gamma_dataset = create_dataset(fname)
    # get the A coefficients from the database from a previous calibration

    acoeffs = client.get('AMS_ACOEFF', 24409).data
    rpos = client.get('AMS_RPOS', 24409).data[0, :]

    ams_a0 = acoeffs[0, 0, :]
    ams_a1 = acoeffs[0, 1, :]
    ams_a2 = acoeffs[0, 2, :]
    ams_a3 = acoeffs[0, 3, :]
    ams_a4 = acoeffs[0, 4, :]
    ams_a5 = acoeffs[0, 5, :]



    # interp bfield
    Bphi = eq['bfld'][65,:,2]

    bphi_interp = interp1d(eq['r'], eq['bfld'][65,:,2])

    bphi_new = bphi_interp(gamma_dataset['r_coords'][0:36])

    polarisation_angle = gamma_dataset['gamma']

    # calculate Bz from the polarisation angle and the toroidal field from the equilibrium - fix up a sign error from the equilibrium (plasma current was +ve, not -ve for this equilibrium)

    Bz = (np.tan(-1 * polarisation_angle[0:36]) * ams_a5 - ams_a2) * bphi_new / (ams_a0 - np.tan(-1 * polarisation_angle[0:36]) * ams_a3)

    j_phi = np.gradient(Bz)/ np.gradient(gamma_dataset['r_coords'][0:36])

    r_coords = gamma_dataset['r_coords'][0:36]

    return r_coords, j_phi


mastu_1ma = '/work/sgibson/msesim/runs/conventional_mse_mastu_fiesta1MA/output/data/conventional_mse_mastu.dat'
mastu_k25 = '/work/sgibson/msesim/runs/conventional_mse_mastu_K25_scenario/output/data/conventional_mse_mastu_k25_scenario.dat'
mastu_mastlike = '/work/sgibson/msesim/runs/conventional_mse_mastu_mastlike/output/data/conventional_mse_mastu.dat'


eq_1ma = readsav('/work/sgibson/msesim/equi/equi_MASTU_1MA_P4_CATIA.sav')
eq_k25 = readsav('/work/sgibson/msesim/equi/equi_MASTU_k25_scenario_centre.sav')
eq_mastlike = readsav('/work/sgibson/msesim/equi/equi_MASTU_mastlike.sav')

r_1ma, jphi_1ma = calc_current(mastu_1ma, eq_1ma)
r_k25, jphi_k25 = calc_current(mastu_k25, eq_k25)
r_ml, jphi_ml = calc_current(mastu_mastlike, eq_mastlike)

plt.figure()
plt.plot(r_1ma, jphi_1ma, label='3-4 beam MAST-U plasma', color='black')
plt.plot(r_k25, jphi_k25, label='2 beam MAST-U plasma', color='red')
plt.plot(r_ml, jphi_ml, label='MAST-like plasma', color='blue')
plt.ylabel('Toroidal Current Density (MA/m$^{2}$)', fontsize=28)
plt.xlabel('R (m)', fontsize=28)
plt.legend(loc=2, prop={'size': 20})
plt.xlim(0.9,1.35)
plt.show()