from __future__ import division

"""
Class to read and store useful information concerning an equilibrium in a manner
that abstracts the user from the data source.

Nick Walkden, James Harrison, Anthony field June 2015
John Omotani April 2018
"""

import numpy as np
from copy import deepcopy as copy
from collections import namedtuple

fluxSurface = namedtuple('fluxSurface', 'R Z')
try:
    # Try to import Afields Point class
    from pyDivertor.Utilities.utilities import Point
except:
    # Otherwise just use a named tuple
    Point = namedtuple('Point', 'r z')

from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline


def interp2d(R, Z, field):
    return RectBivariateSpline(R, Z, np.transpose(field))


class equilibriumField(object):
    """
    Container for fields of the equilibrium.

    An equilibrium field can be accessed either by indexing, ie

        data = myfield[i,j]

    or as a function, ie

        data = myfield(R,Z)

    NOTE:
        At present, operations on two equilibriumField objects, i.e

        equilibriumField + equilibriumField

        return the value of their data only as a numpy array, and do not return
        an equilibriumField object
    """

    def __init__(self, data, function):

        self._data = data
        self._func = function

    def __getitem__(self, inds):
        return self._data[inds]

    def __len__(self):
        return len(self._data)

    def __add__(self, other):
        if type(other) is type(self):
            return self._data + other._data
        else:
            return self._data + other

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if type(other) is type(self):
            return self._data - other._data
        else:
            return self._data - other

    def __truediv__(self, other):
        if type(other) is type(self):
            return self._data / other._data
        else:
            return self._data / other

    def __itruediv__(self, other):
        if type(other) is type(self):
            return other._data / self._data
        else:
            return other / self._data

    def __rsub__(self, other):
        return -1.0 * (self.__sub__(other))

    def __mul__(self, other):
        if type(other) is type(self):
            return self._data * other._data
        else:
            return self._data * other

    def __rmul__(self, other):
        if type(other) is type(self):
            return other._data * self._data
        else:
            return other * self._data

    def __pow__(self, power):
        return self._data ** power

    def __setitem__(self, inds, values):
        self._data[inds] = values

    def __call__(self, *args, **kwargs):
        if len(self._data.shape) > 1:
            return np.transpose(self._func(*args, **kwargs))
        else:
            return self._func(*args, **kwargs)


class equilibrium(object):
    """ Equilibrium object to be passed to
    other codes. Holds useful information about the equilbrium in
    a general way

    Abstracts user from the data source

    Currently the object contains
        R       1d array    Major radius grid points
        Z       1d array    Vertical height grid points
        psi     2d field    poloidal flux as a function of (R,Z)
        psiN        2d field    produce normalized psi for a given R,Z
        dpsidR      2d field    dpsi/dR
        dpsidZ      2d field    dpsi/dZ
        BR      2d field    Radial magnetic field
        BZ      2d field    Vertical magnetic field
        Bp      2d field    poloidal magnetic field
        Bt      2d field    toroidal magnetic field
        B       2d field    total magnetic field
        fpol        1d field    Toroidal flux function, f = R*Bt, as a function of psi
        fpolRZ      2d field    Toroidal flux function in RZ space
        psi_bnd     float       value of the separatrix poloidal flux
        psi_axis    float       value of the poloidal flux on axis
        sigBp       int     Sign of the plasma current (determines sign of poloidal field)
        nr      int     Number of grid points in R
        nz      int     number of grid points in Z
        Btcent      float       Vacuum toroidal field at magnetic axis
        Rcent       float       Major radius at magnetic axis
        wall        dict        Dictionary containing the R and Z coordinates of the wall
        wallshadow_psiN  float  psiN value of the flux surface that first touches a wall apart from the divertor
        nxpt        int     Number of Xpoints
        xpoint      list        list of Xpoint locations as Point types (see above)
        axis        Point       position of the magnetic axis as a Point type (see above)
    """

    def __init__(self, device=None, shot=None, time=None, gfile=None, with_bfield=True, verbose=False, mdsport=None):
        self.psi = None
        self.psiN = None
        self._loaded = False
        self._time = None
        self.R = None
        self.Z = None
        self.dpsidR = None
        self.dpsidZ = None
        self.BR = None
        self.BZ = None
        self.Bp = None
        self.Bt = None
        self.B = None
        self.fpol = None
        self.fpolRZ = None
        self.psi_bnd = None
        self.psi_axis = None
        self.sigBp = None
        self.nr = 0
        self.nz = 0
        self.Btcent = 0.0
        self.Rcent = 0.0
        self.wall = {}
        self.wallshadow_psiN = None
        self.nxpt = 0.0
        self.xpoint = []
        self.axis = None

        if shot is not None and time is not None:
            if device is 'MAST':
                self.load_MAST(shot, time, with_bfield=with_bfield, verbose=verbose)
            elif device is 'JET':
                self.load_JET(shot, time, with_bfield=with_bfield, verbose=verbose)
            elif device is 'TCV':
                self.load_TCV(shot, time, port=mdsport, with_bfield=with_bfield, verbose=verbose)
            elif device is 'AUG':
                self.load_AUG(shot, time, with_bfield=with_bfield, verbose=verbose)
            elif gfile is not None:
                self.load_geqdsk(gfile, with_bfield=with_bfield, verbose=verbose, fiesta_geqdsk=fiesta_geqdsk)
            else:
                return
        else:
            if gfile is not None:
                self.load_geqdsk(gfile, with_bfield=with_bfield, verbose=verbose)
            else:
                return

    def load_geqdsk(self, gfile, with_bfield=True, verbose=False, rev_Bt=False):
        """
        load equilibrium data from an efit geqdsk file

        arguments:
            gfile = file or str file to load data from
                        can be either a file type object
                        or a string containing the file name
        """
        try:
            from .geqdsk import Geqdsk
        except:
            raise ImportError("No geqdsk module found. Cannot load from gfile.")

        ingf = Geqdsk(gfile)
        self.machine = None

        if verbose: print("\n Loading equilibrium from gfile " + str(gfile) + "\n")
        self.nr = ingf['nw']
        self.nz = ingf['nh']

        self.R = np.arange(ingf['nw']) * ingf['rdim'] / float(ingf['nw'] - 1) + ingf['rleft']
        self.Z = np.arange(ingf['nh']) * ingf['zdim'] / float(ingf['nh'] - 1) + ingf['zmid'] - 0.5 * ingf['zdim']

        self._Rinterp, self._Zinterp = np.meshgrid(self.R, self.Z)
        psi = ingf['psirz']
        psi_func = interp2d(self.R, self.Z, psi)
        self.psi = equilibriumField(psi, psi_func)

        self.psi_axis = ingf['simag']
        self.psi_bnd = ingf['sibry']
        self.sigBp = (abs(ingf['current']) / ingf['current'])

        self.axis = Point(ingf['rmaxis'], ingf['zmaxis'])

        fpol = -ingf['fpol']
        if rev_Bt:
            fpol = -fpol
        psigrid = np.linspace(self.psi_axis, self.psi_bnd, len(fpol))
        fpol_func = None  # np.vectorize(interp1d(psigrid,fpol))
        self.fpol = equilibriumField(fpol, fpol_func)

        self._loaded = True
        self.Btcent = ingf['bcentr']
        self.Rcent = ingf['rcentr']
        if rev_Bt:
            self.Btcent = -self.Btcent

        psiN_func = interp2d(self.R, self.Z, (self.psi[:] - self.psi_axis) / (self.psi_bnd - self.psi_axis))
        self.psiN = equilibriumField((self.psi[:] - self.psi_axis) / (self.psi_bnd - self.psi_axis), psiN_func)

        R = ingf['rlim']
        Z = ingf['zlim']
        # if ingf['case'][0]=='Fiesta':
        # Assume it is an SXD case so need to specify wall

        self.wall = {'R': R, 'Z': Z}
        self.xpoints = {}
        self.spoints = None

        self.calc_bfield()
        self._loaded = True

    def dump_geqdsk(self, filename="equilibrium.g"):
        from geqdsk import Geqdsk
        if self._loaded:
            print("Writing gfile: " + filename)
            outgf = Geqdsk()
            outgf.set('nw', self.nr)
            outgf.set('nh', self.nz)
            outgf.set('rleft', np.min(self.R))
            outgf.set('rdim', np.max(self.R) - np.min(self.R))
            outgf.set('zdim', np.max(self.Z) - np.min(self.Z))
            outgf.set('zmid', 0.5 * (np.max(self.Z) + np.min(self.Z)))
            outgf.set('psirz', self.psi[:])
            outgf.set('simag', self.psi_axis)
            outgf.set('sibry', self.psi_bnd)
            outgf.set('current', self.sigBp)
            outgf.set('rmaxis', self.axis.r)
            outgf.set('zmaxis', self.axis.z)
            outgf.set('xdum', 0.00000)
            outgf.set('pres', np.zeros(self.nr))
            outgf.set('pprime', np.zeros(self.nr))
            outgf.set('ffprime', np.zeros(self.nr))
            outgf.set('qpsi', np.zeros(self.nr))
            if self.fpol is not None:
                outgf.set('fpol', self.fpol[:])
            else:
                outgf.set('fpol', np.zeros(self.nr) + self.Rcent * self.Btcent)
            outgf.set('bcentr', self.Btcent)
            outgf.set('rcentr', self.Rcent)
            outgf.set('rlim', self.wall['R'])
            outgf.set('zlim', self.wall['Z'])
            boundary = self.get_fluxsurface(1.0)
            outgf.set('rbbbs', boundary.R)
            outgf.set('zbbbs', boundary.Z)
            outgf.set('nbbbs', len(list(boundary.R)))
            outgf.dump(filename)
        else:
            print("WARNING: No equilibrium loaded, cannot write gfile")

    def load_MAST(self, shot, time, with_bfield=True, verbose=False):
        """ Read in data from MAST IDAM

        arguments:
            shot = int  shot number to read in
            time = float    time to read in data at
        """

        try:
            import idam
            idam.setHost("idam1")
        except:
            raise ImportError("No Idam module found, cannot load MAST shot")

        if verbose: print("\nLoading equilibrium from MAST shot " + str(shot) + "\n")

        self.machine = 'MAST'
        self._shot = shot
        if not self._loaded:
            self._psi = idam.Data("efm_psi(r,z)", shot)
            self._r = idam.Data("efm_grid(r)", shot)
            self._z = idam.Data("efm_grid(z)", shot)
            self._psi_axis = idam.Data("efm_psi_axis", shot)
            self._psi_bnd = idam.Data("efm_psi_boundary", shot)
            self._cpasma = idam.Data("efm_plasma_curr(C)", shot)
            self._bphi = idam.Data("efm_bvac_val", shot)
            self._xpoint1r = idam.Data("efm_xpoint1_r(c)", shot)
            self._xpoint1z = idam.Data("efm_xpoint1_z(c)", shot)
            self._xpoint2r = idam.Data("efm_xpoint2_r(c)", shot)
            self._xpoint2z = idam.Data("efm_xpoint2_z(c)", shot)
            self._axisr = idam.Data("efm_magnetic_axis_r", shot)
            self._axisz = idam.Data("efm_magnetic_axis_z", shot)

        tind = np.abs(self._psi.time - time).argmin()
        self.R = self._r.data[0, :]
        self.Z = self._z.data[0, :]

        psi_func = interp2d(self.R, self.Z, self._psi.data[tind, :, :])
        self.psi = equilibriumField(self._psi.data[tind, :, :], psi_func)

        self.nr = len(self.R)
        self.nz = len(self.Z)

        tind_ax = np.abs(self._psi_axis.time - time).argmin()
        self.psi_axis = self._psi_axis.data[tind_ax]
        tind_bnd = np.abs(self._psi_bnd.time - time).argmin()
        self.psi_bnd = self._psi_bnd.data[tind_bnd]
        self.Rcent = 1.0  # Hard-coded(!)
        tind_Bt = np.abs(self._bphi.time - time).argmin()
        self.Btcent = self._bphi.data[tind_Bt]
        tind_sigBp = np.abs(self._cpasma.time - time).argmin()
        self.sigBp = (abs(self._cpasma.data[tind_sigBp]) / self._cpasma.data[tind_sigBp])

        self.nxpt = 2
        tind_xpt = np.abs(self._xpoint1r.time - time).argmin()
        self.xpoints = {'xp1': Point(self._xpoint1r.data[tind_xpt], self._xpoint1z.data[tind_xpt]),
                        'xp2': Point(self._xpoint2r.data[tind_xpt], self._xpoint2z.data[tind_xpt])}
        self.spoints = None
        self.axis = Point(self._axisr.data[tind_xpt], self._axisz.data[tind_xpt])

        self.fpol = None

        self._loaded = True
        self._time = self._psi.time[tind]

        psiN_func = interp2d(self.R, self.Z, (self.psi[:] - self.psi_axis) / (self.psi_bnd - self.psi_axis))
        self.psiN = equilibriumField((self.psi[:] - self.psi_axis) / (self.psi_bnd - self.psi_axis), psiN_func)

        # Previously included version of MAST wall, without structures for poloidal field coils
        # R = [
        # 0.195, 0.195, 0.280, 0.280, 0.280, 0.175,
        # 0.175, 0.190, 0.190, 0.330, 0.330, 0.535,
        # 0.535, 0.755, 0.755, 0.755, 1.110, 1.655,
        # 1.655, 1.655, 2.000, 2.000, 1.975, 2.000,
        # 2.000, 1.655, 1.655, 1.655, 1.110, 0.755,
        # 0.755, 0.755, 0.535, 0.535, 0.330, 0.330,
        # 0.190, 0.190, 0.175, 0.175, 0.280, 0.280,
        # 0.280, 0.195, 0.195]
        #
        # Z = [
        # 0.000, 1.080, 1.220, 1.450, 1.670, 1.670,
        # 1.720, 1.820, 1.905, 2.150, 2.190, 2.190,
        # 2.165, 2.165, 1.975, 1.826, 1.826, 1.826,
        # 1.975, 2.165, 2.165, 0.300, 0.000,-0.300,
        # -2.165,-2.165,-1.975,-1.826,-1.826,-1.826,
        # -1.975,-2.165,-2.165,-2.190,-2.190,-2.150,
        # -1.905,-1.820,-1.720,-1.670,-1.670,-1.450,
        # -1.220,-1.080, 0.000]

        R = np.array([
            1.9, 1.5551043, 1.5551043, 1.4079306, 1.4079306,
            1.0399311, 1.0399311, 1.9, 1.9, 0.56493068,
            0.56493068, 0.78350002, 0.78350002, 0.58259028, 0.4165,
            0.28, 0.28, 0.1952444, 0.1952444, 0.28,
            0.28, 0.4165, 0.58259028, 0.78350002, 0.78350002,
            0.56493068, 0.56493068, 1.9, 1.9, 1.0399311,
            1.0399311, 1.4079306, 1.4079306, 1.5551043, 1.5551043,
            1.9, 1.9])

        Z = np.array([
            0.405, 0.405, 0.82250023, 0.82250023, 1.0330003,
            1.0330003, 1.1950001, 1.1950001, 1.825, 1.825,
            1.7280816, 1.7280816, 1.7155817, 1.5470001, 1.5470001,
            1.6835001, 1.2290885, 1.0835, -1.0835, -1.2290885,
            -1.6835001, -1.5470001, -1.5470001, -1.7155817, -1.7280816,
            -1.7280816, -1.825, -1.825, -1.1950001, -1.1950001,
            -1.0330003, -1.0330003, -0.82250023, -0.82250023, -0.405,
            -0.405, 0.405])

        self.wall = {'R': R, 'Z': Z}
        if with_bfield: self.calc_bfield()

    def load_JET(self, shot, time, sequence=0, with_bfield=True, verbose=False):
        """ Read in data from MAST IDAM

        arguments:
            shot = int  shot number to read in
            time = float    time to read in data at
        """

        try:
            from ppf import ppfget, ppfgo, ppferr, ppfuid
        except:
            raise ImportError("No ppf module found, cannot load JET shot")

        if verbose: print("\nLoading equilibrium from JET shot " + str(shot) + "\n")

        ppfuid('JETPPF', 'r')

        self._shot = shot
        self._sequence = sequence
        self.machine = 'JET'

        def ppf_error_check(func, err):
            if err != 0:
                msg, ierr = ppferr(func, err)
                exit(func + ": error code : " + msg)

        err = ppfgo(self._shot, self._sequence)
        ppf_error_check('PPFGO', err)

        if not self._loaded:
            # Only store things we will need again as attributes
            (c_bnd, m_bnd, self._bnd_data, x_bnd, t_bnd, error_bnd) = ppfget(shot, "EFIT", "fbnd")
            ppf_error_check('PPFGET', error_bnd)
            (c_axs, m_axs, self._axs_data, x_axs, t_axs, error_axs) = ppfget(shot, "EFIT", "faxs")
            ppf_error_check('PPFGET', error_axs)
            (c_psi, m_psi, self._psi_data, x_psi, self._t_psi, error_psi) = ppfget(shot, "EFIT", "psi")
            ppf_error_check('PPFGET', error_psi)
            (c_psir, m_psir, self._psi_r, x_psir, t_psir, error_psir) = ppfget(shot, "EFIT", "psir")
            ppf_error_check('PPFGET', error_psir)
            (c_psiz, m_psiz, self._psi_z, x_psiz, t_psiz, error_psiz) = ppfget(shot, "EFIT", "psiz")
            ppf_error_check('PPFGET', error_psiz)
            (c_rbphi, m_rbphi, self._r_bphi, x_rbphi, t_rbphi, error_rbphi) = ppfget(shot, "EFIT", "bvac")
            ppf_error_check('PPFGET', error_rbphi)
            (c_rsim, m_rxpm, self._rxpm, x_rxpm, t_rxpm, error_rxpm) = ppfget(shot, "EFIT", "rxpm")
            ppf_error_check('PPFGET', error_rxpm)
            (c_zxpm, m_zxpm, self._zxpm, x_zxpm, t_zxpm, error_zxpm) = ppfget(shot, "EFIT", "zxpm")
            ppf_error_check('PPFGET', error_zxpm)
            (c_rxpl, m_rxpl, self._rxpl, x_rxpl, t_rxpl, error_rxpl) = ppfget(shot, "EFIT", "rxpl")
            ppf_error_check('PPFGET', error_rxpl)
            (c_rxpl, m_zxpl, self._zxpl, x_zxpl, t_zxpl, error_zxpl) = ppfget(shot, "EFIT", "zxpl")
            ppf_error_check('PPFGET', error_zxpl)
            (c_rxpu, m_rxpu, self._rxpu, x_rxpu, t_rxpu, error_rxpu) = ppfget(shot, "EFIT", "rxpu")
            ppf_error_check('PPFGET', error_rxpu)
            (c_zxpu, m_zxpu, self._zxpu, x_zxpu, t_zxpu, error_zxpu) = ppfget(shot, "EFIT", "zxpu")
            ppf_error_check('PPFGET', error_zxpu)
            (c_rsim, m_rsim, self._rsim, x_rsim, t_rsim, error_rsim) = ppfget(shot, "EFIT", "rsim")
            ppf_error_check('PPFGET', error_rsim)
            (c_zsim, m_zsim, self._zsim, x_zsim, t_zsim, error_zsim) = ppfget(shot, "EFIT", "zsim")
            ppf_error_check('PPFGET', error_zsim)
            (c_rsil, m_rsil, self._rsil, x_rsil, t_rsil, error_rsil) = ppfget(shot, "EFIT", "rsil")
            ppf_error_check('PPFGET', error_rsil)
            (c_zsil, m_zsil, self._zsil, x_zsil, t_zsil, error_zsil) = ppfget(shot, "EFIT", "zsil")
            ppf_error_check('PPFGET', error_zsil)
            (c_rsiu, m_rsiu, self._rsiu, x_rsiu, t_rsiu, error_rsiu) = ppfget(shot, "EFIT", "rsiu")
            ppf_error_check('PPFGET', error_rsiu)
            (c_zsiu, m_zsiu, self._zsiu, x_zsiu, t_zsiu, error_zsiu) = ppfget(shot, "EFIT", "zsiu")
            ppf_error_check('PPFGET', error_zsiu)
            (c_rsom, m_rsom, self._rsom, x_rsom, t_rsom, error_rsom) = ppfget(shot, "EFIT", "rsom")
            ppf_error_check('PPFGET', error_rsom)
            (c_zsom, m_zsom, self._zsom, x_zsom, t_zsom, error_zsom) = ppfget(shot, "EFIT", "zsom")
            ppf_error_check('PPFGET', error_zsom)
            (c_rsol, m_rsol, self._rsol, x_rsol, t_rsol, error_rsol) = ppfget(shot, "EFIT", "rsol")
            ppf_error_check('PPFGET', error_rsol)
            (c_zsol, m_zsol, self._zsol, x_zsol, t_zsol, error_zsol) = ppfget(shot, "EFIT", "zsol")
            ppf_error_check('PPFGET', error_zsol)
            (c_rsou, m_rsou, self._rsou, x_rsou, t_rsou, error_rsou) = ppfget(shot, "EFIT", "rsou")
            ppf_error_check('PPFGET', error_rsou)
            (c_zsou, m_zsou, self._zsou, x_zsou, t_zsou, error_zsou) = ppfget(shot, "EFIT", "zsou")
            ppf_error_check('PPFGET', error_zsou)
            (c_axisr, m_axisr, self._axisr, x_axisr, t_axisr, error_axisr) = ppfget(shot, "EFIT", "rmag")
            ppf_error_check('PPFGET', error_axisr)
            (c_axisz, m_axisz, self._axisz, x_axisz, t_axisz, error_axisz) = ppfget(shot, "EFIT", "zmag")
            ppf_error_check('PPFGET', error_axisz)
            (c_nbnd, m_nbnd, self._nbnd, x_nbnd, t_nbnd, error_nbnd) = ppfget(shot, "EFIT", "nbnd")
            ppf_error_check('PPFGET', error_nbnd)
            (c_rbnd, m_rbnd, d_rbnd, x_rbnd, t_rbnd, error_rbnd) = ppfget(shot, "EFIT", "rbnd")
            ppf_error_check('PPFGET', error_rbnd)
            self._rbnd_data = np.reshape(d_rbnd, (len(t_rbnd), len(x_rbnd)))
            (c_zbnd, m_zbnd, d_zbnd, x_zbnd, t_zbnd, error_zbnd) = ppfget(shot, "EFIT", "zbnd")
            ppf_error_check('PPFGET', error_zbnd)
            self._zbnd_data = np.reshape(d_zbnd, (len(t_zbnd), len(x_zbnd)))

        # Specify radius where toroidal field is specified
        self._r_at_bphi = 2.96

        self._psi_data = np.reshape(self._psi_data, (len(self._t_psi), len(self._psi_r), len(self._psi_z)))

        tind = np.abs(np.asarray(self._t_psi) - time).argmin()

        # Define the wall geometry
        wall_r = np.array([
            3.2900, 3.1990, 3.1940, 3.0600, 3.0110, 2.9630, 2.9070, 2.8880, 2.8860, 2.8900, 2.9000, \
            2.8840, 2.8810, 2.8980, 2.9870, 2.9460, 2.8700, 2.8340, 2.8140, 2.7000, 2.5760, 2.5550, \
            2.5500, 2.5220, 2.5240, 2.4370, 2.4060, 2.4180, 2.4210, 2.3980, 2.4080, 2.4130, 2.4130, \
            2.4050, 2.3600, 2.2950, 2.2940, 2.2000, 2.1390, 2.0810, 1.9120, 1.8210, 1.8080, 1.8860, \
            1.9040, 1.9160, 2.0580, 2.1190, 2.1780, 2.2430, 2.3810, 2.5750, 2.8840, 2.9750, 3.1490, \
            3.3000, 3.3980, 3.4770, 3.5620, 3.6400, 3.6410, 3.7430, 3.8240, 3.8855, 3.8925, 3.8480, \
            3.7160, 3.5780, 3.3590, 3.3090, 3.2940, 3.2900])

        wall_z = np.array([
            -1.1520, -1.2090, -1.2140, -1.2980, -1.3351, -1.3350, -1.3830, -1.4230, -1.4760, -1.4980, \
            -1.5100, -1.5820, -1.6190, -1.6820, -1.7460, -1.7450, -1.7130, -1.7080, -1.6860, -1.6510, \
            -1.6140, -1.6490, -1.6670, -1.7030, -1.7100, -1.7110, -1.6900, -1.6490, -1.6020, -1.5160, \
            -1.5040, -1.4740, -1.4290, -1.3860, -1.3340, -1.3341, -1.3200, -1.2440, -1.0970, -0.9580, \
            -0.5130, -0.0230, 0.4260, 1.0730, 1.1710, 1.2320, 1.6030, 1.7230, 1.8390, 1.8940, 1.9670, \
            2.0160, 1.9760, 1.9410, 1.8160, 1.7110, 1.6430, 1.5720, 1.4950, 1.4250, 1.2830, 1.0700, \
            0.8290, 0.4950, 0.2410, -0.0960, -0.4980, -0.7530, -1.0310, -1.0800, -1.1110, -1.1520])

        # Interpolate the equilibrium onto a finer mesh
        psi_func = interp2d(self._psi_r, self._psi_z, self._psi_data[tind, :, :])
        self.psi = equilibriumField(self._psi_data[tind, :, :], psi_func)
        self.nr = len(self._psi_r)
        self.nz = len(self._psi_z)
        self.R = self._psi_r
        self.Z = self._psi_z

        self.psi_axis = self._axs_data[tind]
        self.psi_bnd = self._bnd_data[tind]

        self.Btcent = self._r_bphi[tind]
        self.Rcent = self._r_at_bphi
        self.sigBp = 1.0

        self.nxpt = 1
        # print(self._rxpm)
        self.xpoints = {"xpm": Point(self._rxpm[tind], self._zxpm[tind]),
                        "xpl": Point(self._rxpl[tind], self._zxpl[tind]),
                        "xpu": Point(self._rxpu[tind], self._zxpu[tind])}

        self.spoints = {"sim": Point(self._rsim[tind], self._zsim[tind]),
                        "som": Point(self._rsom[tind], self._zsom[tind]),
                        "sil": Point(self._rsil[tind], self._zsil[tind]),
                        "sol": Point(self._rsol[tind], self._zsol[tind]),
                        "siu": Point(self._rsiu[tind], self._zsiu[tind]),
                        "sou": Point(self._rsou[tind], self._zsou[tind])}

        self.axis = Point(self._axisr[tind], self._axisz[tind])

        self.fpol = None

        tmp = (self.psi[:] - self.psi_axis) / (self.psi_bnd - self.psi_axis)

        psiN_func = interp2d(self.R, self.Z, (self.psi[:] - self.psi_axis) / (self.psi_bnd - self.psi_axis))
        self.psiN = equilibriumField((self.psi[:] - self.psi_axis) / (self.psi_bnd - self.psi_axis), psiN_func)

        self._loaded = True
        self._time = self._t_psi[tind]

        self.wall = {'R': wall_r, 'Z': wall_z}

        if with_bfield: self.calc_bfield()

    def load_TCV(self, shot, time, port=None, with_bfield=True, verbose=False):
        try:
            import MDSplus as mds
        except:
            raise ImportError("Error: No MDSplus module found. \nPlease see www.mdsplus.org to install MDSplus\n")

        if port is None:
            port = raw_input("Please enter port for ssh tunnel to TCV LAC: ")
        try:
            conn = mds.Connection('localhost:' + str(port))
            self._mdsport = port
        except:
            print("---------------------------------------------------------------------")
            print("Have you created an ssh tunnel to the TCV LAC?")
            print("Create a tunnel by typing into the command line")
            print("      ssh username@lac.911.epfl.ch -L XYZ:tcvdata.epfl.ch:8000")
            print("where XYV is your port number (i.e 1600)")
            print("---------------------------------------------------------------------")
            raise RuntimeError("Error: Could not connect through port")

        # Open tree to Lique data for equilibrium info

        self.machine = 'TCV'
        self._shot = shot
        if not self._loaded:
            # Tree = mds.Tree('tcv_shot',shot)
            conn.openTree('tcv_shot', int(shot))
            # Load in required data from Lique off the MDS server
            # mds_str = '\\results::{}'
            mds_str = 'tcv_eq2("{}","liuqe.m")'

            self._psi = conn.get(mds_str.format("psi")).data() / (2.0 * np.pi)
            self._time_array = conn.get('dim_of(' + mds_str.format("psi_axis") + ')').data()
            nr = self._psi.shape[2]
            nz = self._psi.shape[1]
            self._r = np.linspace(conn.get('\\results::parameters:ri').data(),
                                  conn.get('\\results::parameters:ro').data(), nr)
            z0 = conn.get('\\results::parameters:zu').data()
            self._z = np.linspace(-z0, z0, nz)
            self._psi_axis = conn.get(mds_str.format('psi_axis')).data()
            self._psi_bnd = conn.get(mds_str.format('psi_xpts')).data()
            self._bphi = conn.get('\\magnetics::rbphi').data()
            self._bt_time_array = conn.get('dim_of(\\magnetics::rbphi)').data()
            self._xpointr = conn.get(mds_str.format('r_xpts')).data()
            self._xpointz = conn.get(mds_str.format('z_xpts')).data()
            self._axisr = conn.get(mds_str.format('r_axis')).data()
            self._axisz = conn.get(mds_str.format('z_axis')).data()
            conn.closeTree('tcv_shot', int(shot))
            self._loaded = True

        tind = np.abs(self._time_array - time).argmin()
        # print(tind)
        # print(self._time_array.shape)
        # print(self._psi.shape)
        # print(tind)
        # print "\n\n",tind,"\n\n"
        self.R = self._r  # .data[0,:]
        self.Z = self._z  # .data[0,:]
        psi_func = interp2d(self.R, self.Z, self._psi[tind])
        self.psi = equilibriumField(self._psi[tind], psi_func)
        self.nr = len(self.R)
        self.nz = len(self.Z)
        self.axis = Point(self._axisr[tind], self._axisz[tind])
        tind_ax = tind  # np.abs(self._psi_axis.time - time).argmin()
        self.psi_axis = self.psi(self.axis.r, self.axis.z)
        tind_bnd = tind  # np.abs(self._psi_bnd.time - time).argmin()
        self.psi_bnd = 0.0  # np.max(self._psi_bnd[tind_bnd])
        self.Rcent = 0.88  # Hard-coded(!)
        tind_Bt = np.abs(self._bt_time_array - time).argmin()
        self.Btcent = self._bphi[0, tind_Bt]
        tind_sigBp = tind  # np.abs(self._cpasma.time - time).argmin()
        self.sigBp = 1.0  # -(abs(self._cpasma.data[tind_sigBp])/self._cpasma.data[tind_sigBp])

        self.nxpt = 2
        tind_xpt = tind  # np.abs(self._xpoint1r.time - time).argmin()
        self.xpoints = {'xp1': Point(self._xpointr[tind_xpt][0], self._xpointz[tind_xpt][0]),
                        'xp2': Point(self._xpointr[tind_xpt][1], self._xpointz[tind_xpt][1])}
        self.spoints = None
        # self.xpoint.append(Point(self._xpoint2r.data[tind_xpt],self._xpoint2z.data[tind_xpt]))

        self.fpol = None

        self._loaded = True
        self._time = 0.0  # self._psi.time[tind]

        psiN_func = interp2d(self.R, self.Z, (self.psi[:] - self.psi_axis) / (self.psi_bnd - self.psi_axis))
        self.psiN = equilibriumField((self.psi[:] - self.psi_axis) / (self.psi_bnd - self.psi_axis), psiN_func)

        R = np.array(
            [0.624, 0.624, 0.666, 0.672, 0.965, 0.971, 1.136, 1.136, 0.971, 0.965, 0.672, 0.666, 0.624, 0.624, 0.624])
        Z = np.array(
            [0.697, 0.704, 0.75, 0.75, 0.75, 0.747, 0.55, -0.55, -0.747, -0.75, -0.75, -0.75, -0.704, -0.697, 0.697])

        self.wall = {'R': R, 'Z': Z}
        if with_bfield: self.calc_bfield()

    def load_AUG(self, shot, time, with_bfield=True, verbose=False):
        self.machine = 'AUG'
        self._shot = shot
        if not self._loaded:
            Eqm = map_equ.equ_map()
            status = Eqm.Open(self._shot, diag='EQH')
            Eqm._read_scalars()
            Eqm._read_profiles()
            Eqm._read_pfm()
            # Load in required data
            # The transpose is needed since in this way we have dimension of
            # the form (#samples, Rgrid, ZGrid)
            self._psi = Eqm.pfm.transpose()
            self._time_array = Eqm.t_eq
            nr = self._psi.shape[0]
            nz = self._psi.shape[1]
            self._r = Eqm.Rmesh
            self._z = Eqm.Zmesh
            self._psi_axis = Eqm.psi0
            self._psi_bnd = Eqm.psix
            # get the fpol in similar way
            # as done in eqtools
            self._jpol = Eqm.jpol
            # these are the lower xpoints
            self._rxpl = Eqm.ssq['Rxpu']
            self._zxpl = Eqm.ssq['Zxpu']
            # read also the upper xpoint
            self._rxpu = Eqm.ssq['Rxpo']
            self._zxpu = Eqm.ssq['Zxpo']
            # R magnetic axis
            self._axisr = Eqm.ssq['Rmag']
            # Z magnetic axis
            self._axisz = Eqm.ssq['Zmag']
            # eqm does not load the RBphi on axis
            Mai = dd.shotfile('MAI', self._shot)
            self.Rcent = 1.65
            # we want to interpolate on the same time basis
            Spl = UnivariateSpline(Mai('BTF').time, Mai('BTF').data, s=0)
            self._bphi = Spl(self._time_array) * self.Rcent
            Mai.close()
            Mag = dd.shotfile('MAG', self._shot)
            Spl = UnivariateSpline(Mag('Ipa').time, Mag('Ipa').data, s=0)
            self._cplasma = Spl(self._time_array)
            # we want to load also the plasma curent
            self._loaded = True

        tind = np.abs(self._time_array - time).argmin()
        self.R = self._r  # .data[0,:]
        self.Z = self._z  # .data[0,:]
        psi_func = interp2d(self.R, self.Z, self._psi[tind])
        self.psi = equilibriumField(self._psi[tind], psi_func)
        self.nr = len(self.R)
        self.nz = len(self.Z)
        self.psi_axis = self._psi_axis[tind]
        self.psi_bnd = self._psi_bnd[tind]
        self.Btcent = self._bphi[tind]
        self.sigBp = np.sign(self._cplasma[tind])
        fpol = self._jpol[:, tind] * 2e-7
        fpol = fpol[:np.argmin(np.abs(fpol))]
        psigrid = np.linspace(self.psi_axis, self.psi_bnd, len(fpol))
        self.fpol = equilibriumField(fpol, UnivariateSpline(psigrid, fpol, s=0))
        self.nxpt = 2
        tind_xpt = tind  # np.abs(self._xpoint1r.time - time).argmin()
        self.xpoints = {'xpl': Point(self._rxpl[tind], self._zxpl[tind]),
                        'xpu': Point(self._rxpu[tind], self._zxpu[tind])}
        self.spoints = None
        # self.xpoint.append(Point(self._xpoint2r.data[tind_xpt],self._xpoint2z.data[tind_xpt]))
        self.axis = Point(self._axisr[tind], self._axisz[tind])
        self.fpol = None

        self._loaded = True
        self._time = self._time_array[tind]

        psiN_func = interp2d(self.R, self.Z, (self.psi[:] - self.psi_axis) / (self.psi_bnd - self.psi_axis))
        self.psiN = equilibriumField(
            (self.psi[:] - self.psi_axis) / (self.psi_bnd - self.psi_axis), psiN_func)

        VesselFile = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'augVesseldata.txt')
        x, y = np.loadtxt(VesselFile, unpack=True)
        self.wall = {'R': x, 'Z': y}
        if with_bfield: self.calc_bfield()

    def set_time(self, time):
        if self._loaded and self.machine is not None:
            if self.machine == 'MAST':
                self.load_MAST(self._shot, time)
            elif self.machine == 'JET':
                self.load_JET(self._shot, time)
            elif self.machine == 'TCV':
                self.load_TCV(self._shot, time, port=self._mdsport)
            elif self.machine == 'AUG':
                self.load_AUG(self._shot, time)

    def plot_flux(self, col_levels=None, Nlines=20, axes=None, wall_shadow=False, label_contours=False,
                  normalized_psi=False, show=True, title=None, colorbar=True, cmap=None, savename=None):
        """
        Contour plot of the equilibrium poloidal flux

        keywords:
            col_levels = []     array storing contour levels for color plot
            Nlines = int        number of contour lines to display
            axes                if given, add the plot to already existing Matplotlib axes
            wall_shadow = bool  toggle plotting of position of wall shadow (default True)
            label_contours = bool   toggle labels for psi contours (default False)
            normalized_psi = bool   toggle normalized psi (0 at axis, 1 at separatrix) (default False)
            show = bool         toggle display of plot (default True)
            title = string      set title for plot
            colorbar = bool     toggle colorbar for plot (default True)
            cmap                if given, custom colormap for plot
            savename            if given, filename for saved copy of plot, passed to savefig (default None)
        """
        import matplotlib.pyplot as plt
        # Set contour levels
        if self.psi is not None:
            if normalized_psi:
                psi = self.psiN
                psilabel = "$\psi_\mathrm{N}$"
            else:
                psi = self.psi
                psilabel = "$\psi$"
            if col_levels is None:
                col_levels = np.linspace(np.min(psi), np.max(psi), 100)

            if axes is None:
                axes = plt.gca()
            else:
                show = False
            im = axes.contourf(self.R, self.Z, psi, levels=col_levels, cmap=cmap)
            if colorbar: cbar = plt.colorbar(im, format='%.3f', ax=axes)
            psi_contours = axes.contour(self.R, self.Z, psi, Nlines, colors='k', linestyles='solid')
            if label_contours:
                plt.clabel(psi_contours, fontsize='xx-small', fmt='%1.2f')

            axes.set_xlim(np.min(self.R), np.max(self.R))
            axes.set_ylim(np.min(self.Z), np.max(self.Z))
            if self.wall is not None:
                axes.plot(self.wall['R'], self.wall['Z'], '-k', linewidth=4)
            if self.xpoints:
                for xpoint in self.xpoints:
                    try:
                        axes.plot(self.xpoints[xpoint].r, self.xpoints[xpoint].z, 'rx')
                        axes.contour(self.R, self.Z, self.psi,
                                     [self.psi(self.xpoints[xpoint].r, self.xpoints[xpoint].z)], colors='r',
                                     linewidth=2.5, linestyles='solid')
                    except:
                        pass

            if self.axis is not None:
                try:
                    axes.plot(self.axis.r, self.axis.z, 'ro')
                except:
                    pass
            if wall_shadow:
                if self.wallshadow_psiN is None:
                    self.find_wall_shadow()
                wall_contour = axes.contour(self.R, self.Z, self.psiN, [self.wallshadow_psiN], colors='b',
                                            linewidth=1.7, linestyles='solid')
                if label_contours:
                    plt.clabel(wall_contour, fontsize='xx-small', fmt='%1.2f')

            axes.set_xlabel('R (m)')
            axes.set_ylabel('Z (m)')
            if title is not None: axes.set_title(title)
            if colorbar: cbar.ax.set_ylabel(psilabel)

            axes.set_aspect('equal')

            if savename is not None:
                plt.gcf().savefig(savename, bbox_inches='tight')
            if show: plt.show()

        else:
            print("ERROR: No poloidal flux found. Please load an equilibrium before plotting.\n")

    def get_fluxsurface(self, psiN, Rref=1.5, Zref=0.0):
        """
        Get R,Z coordinates of a flux surface at psiN
        """
        try:
            import matplotlib.pyplot as plt
        except:
            print("ERROR: matplotlib required for flux surface construction, returning")
            return

        if type(psiN) is list:
            surfaces = []
            for psiNval in psiN:
                psi_cont = plt.contour(self.R, self.Z, (self.psi - self.psi_axis) / (self.psi_bnd - self.psi_axis),
                                       levels=[0, psiN], alpha=0.0)
                paths = psi_cont.collections[1].get_paths()
                # Figure out which path to use
                i = 0
                old_dist = 100.0
                for path in paths:
                    dist = np.min(((path.vertices[:, 0] - Rref) ** 2.0 + (path.vertices[:, 1] - Zref) ** 2.0) ** 0.5)
                    if dist < old_dist:
                        true_path = path
                        old_dist = dist

                R, Z = true_path.vertices[:, 0], true_path.vertices[:, 1]
                surfaces.append(fluxSurface(R=R, Z=Z))
            return surfaces
        else:
            psi_cont = plt.contour(self.R, self.Z, (self.psi - self.psi_axis) / (self.psi_bnd - self.psi_axis),
                                   levels=[0, psiN], alpha=0.0)
            paths = psi_cont.collections[1].get_paths()
            old_dist = 100.0
            for path in paths:
                dist = np.min(((path.vertices[:, 0] - Rref) ** 2.0 + (path.vertices[:, 1] - Zref) ** 2.0) ** 0.5)
                if dist < old_dist:
                    true_path = path
                    old_dist = dist
            R, Z = true_path.vertices[:, 0], true_path.vertices[:, 1]
            # plt.clf()
            return fluxSurface(R=R, Z=Z)

    def plot_var(self, var, title=None, col_levels=None, with_wall=False, axes=None, show=True, colorbar=True,
                 cbar_label=None, with_flux=False, cmap=None):
        import matplotlib.pyplot as plt
        if axes is None:
            axes = plt.gca()

        if col_levels == None:
            col_levels = np.linspace(np.min(var), np.max(var), 100)
        im = axes.contourf(self.R, self.Z, var, levels=col_levels, cmap=cmap)
        if colorbar: cbar = plt.colorbar(im, format='%.3f', ax=axes)
        if cbar_label is not None: cbar.ax.set_ylabel(cbar_label)
        if self.wall is not None and with_wall:
            axes.plot(self.wall['R'], self.wall['Z'], '-k', linewidth=4)
        if with_flux:
            axes.contour(self.R, self.Z, self.psi, 20, colors='k', linestyles='solid', alpha=0.5)
            if self.xpoints:
                for xpoint in self.xpoints:
                    axes.contour(self.R, self.Z, self.psi, [self.psi(self.xpoints[xpoint].r, self.xpoints[xpoint].z)],
                                 colors='k', linestyles='dashed', alpha=0.5)
        axes.set_xlabel('R (m)')
        axes.set_ylabel('Z (m)')
        axes.set_xlim(np.min(self.R), np.max(self.R))
        axes.set_ylim(np.min(self.Z), np.max(self.Z))
        if title != None:
            axes.set_title(title)
        axes.set_aspect('equal')
        if show: plt.show()

    def count_wall_intersections(self, R, Z):
        """
        Check for an intersection between the line whose points are given by the numpy arrays (R,Z) and the wall
        [note, based on wall_intersection method of pyFieldlineTracer, but converted to work on whole arrays at once
        """
        xa = np.array(R[:-1])
        ya = np.array(Z[:-1])
        xb = np.array(R[1:])
        yb = np.array(Z[1:])

        wallxc = self.wall['R'][:-1]
        wallyc = self.wall['Z'][:-1]
        wallxd = self.wall['R'][1:]
        wallyd = self.wall['Z'][1:]

        # First check for intersection
        def ccw_line_line_wall(xa, ya, xb, yb, xc, yc):
            """
            check if points are counterclockwise
            xa, ya, xb, yb all have the same dimensions (points from the line)
            xc, yc have the same dimensions (points from the wall)
            returns 2d array with dimensions (len(xa), len(xc))
            """
            xa = xa[:, np.newaxis]
            ya = ya[:, np.newaxis]
            xb = xb[:, np.newaxis]
            yb = yb[:, np.newaxis]
            xc = xc[np.newaxis, :]
            yc = yc[np.newaxis, :]
            return (yc - ya) * (xb - xa) > (yb - ya) * (xc - xa)

        def ccw_line_wall_wall(xa, ya, xc, yc, xd, yd):
            """
            check if points are counterclockwise
            xa, ya have the same dimensions (points from the line)
            xc, yc, xd, yd have the same dimensions (points from the wall)
            returns 2d array with dimensions (len(xa), len(xd))
            """
            xa = xa[:, np.newaxis]
            ya = ya[:, np.newaxis]
            xc = xc[np.newaxis, :]
            yc = yc[np.newaxis, :]
            xd = xd[np.newaxis, :]
            yd = yd[np.newaxis, :]
            return (yd - ya) * (xc - xa) > (yc - ya) * (xd - xa)

        # Note: multiplying arrays of booleans is equivalent to an elementwise 'AND' operation
        intersections = (
                (ccw_line_wall_wall(xa, ya, wallxc, wallyc, wallxd, wallyd) != ccw_line_wall_wall(xb, yb, wallxc,
                                                                                                  wallyc, wallxd,
                                                                                                  wallyd))
                * (ccw_line_line_wall(xa, ya, xb, yb, wallxc, wallyc) != ccw_line_line_wall(xa, ya, xb, yb, wallxd,
                                                                                            wallyd))
        )
        # Now each 'True' in intersections represents an intersection between an element of the line and an element of the wall
        return intersections.sum()

    def find_wall_shadow(self, region="outboard"):
        """
        Find the first flux surface or surfaces that intersect the wall other than at the divertor.
        Check the surfaces in the main SOL (i.e. outboard SOL for a double-null)

        - region - {'outboard', 'inboard'} selects which region to search for
          the wall shadow in. Should make no difference in single null plasma
          but may be useful for a double null. Defaults to 'outboard'.
        """
        tolerance = 0.00001  # acceptable error in psiN space on which flux surface first intersects the wall
        dpsi = 0.05  # Initial step for psiN search
        psiNguess = 1.01  # Initial value for guess of psiN of flux surface that intersects wall

        def updateRZguess(R, Z, closed=False):
            if closed:
                # Select a point, chosen according to the value of the region
                # option, near which to search for the next flux surface
                if region == "outboard":
                    # Search near the point with the largest major radius
                    i = R.argmax()
                elif region == "inboard":
                    # Search near the point with the smallest major radius
                    i = R.argmin()
                else:
                    raise ValueError("Error in find_wall_shadow(), unrecognized region argument: " + str(
                        region) + "\ngood values are: outboard, inboard")
            else:
                # Already on an open field line, look for a point near the midplane
                if region == "outboard":
                    # Only look for the midplane outboard of the magnetic axis
                    test = R > self.axis.r
                    testZ = Z.copy()
                    testZ[test] = float("inf")
                    i = abs(testZ).argmin()
                elif region == "inboard":
                    test = R < self.axis.r
                    testZ = Z.copy()
                    testZ[test] = float("inf")
                    i = abs(testZ).argmin()
                else:
                    raise ValueError("Error in find_wall_shadow(), unrecognized region argument: " + str(
                        region) + "\ngood values are: outboard, inboard")
            return R[i], Z[i]

        # Want to start from the outboard midplane, so first find a closed flux surface near the separatrix
        closedR, closedZ = self.get_fluxsurface(.9, Rref=self.axis.r, Zref=self.axis.z)
        # Find the point to search around
        Rguess, Zguess = updateRZguess(closedR, closedZ, closed=True)
        # ...find a nearby open flux surface just outside the separatrix
        openR, openZ = self.get_fluxsurface(1.1, Rref=Rguess, Zref=Zguess)
        # Update the R and Z to search from
        Rguess, Zguess = updateRZguess(openR, openZ)

        # Simple binary search algorithm
        needToUpdateRZguess = False
        haveFoundIntersection = False
        while dpsi > tolerance:
            R, Z = self.get_fluxsurface(psiNguess + dpsi, Rref=Rguess, Zref=Zguess)
            if needToUpdateRZguess:
                Rguess, Zguess = updateRZguess(R, Z)
            if self.count_wall_intersections(R, Z) > 2:
                # found a surface with wall intersection in addition to divertor
                # refine search...
                dpsi = dpsi / 2.
                haveFoundIntersection = True
            else:
                # surface with wall intersection in addition to divertor is further out
                # keep searching...
                psiNguess = psiNguess + dpsi
                # update R and Z to search for flux surface from
                needToUpdateRZguess = True
                if haveFoundIntersection:
                    # Full step at dpsi would go back to a surface where we already found an intersection
                    # so decrease dpsi as well
                    dpsi = dpsi / 2.

        # Take a final value at the half way point between the surface that doesn't intersect the wall and the one that does
        psiNresult = psiNguess + dpsi / 2.

        self.wallshadow_psiN = psiNresult
        return psiNresult

    def dump_equ(self, filename="equilibrium"):
        """
        Saves the equilibrium in a .equ file
        """

        f = open(filename + '.equ', 'w')
        f.write('    jm   :=  no. of grid points in radial direction;\n')
        f.write('    km   :=  no. of grid points in vertical direction;\n')
        f.write('    r    :=  radial co-ordinates of grid points  [m];\n')
        f.write('    z    :=  vertical co-ordinates of grid points  [m];\n')
        f.write('    psi  :=  flux per radian at grid points      [Wb/rad];\n')
        f.write('    psib :=  psi at plasma boundary              [Wb/rad];\n')
        f.write('    btf  :=  toroidal magnetic field                  [T];\n')
        f.write('    rtf  :=  major radius at which btf is specified   [m];\n')
        f.write('\n\n')
        f.write('    jm    = ' + str(len(self.R)) + ' ;\n')
        f.write('    km    = ' + str(len(self.Z)) + ' ;\n')
        f.write('    psib  = ' + str(self.psi_bnd) + ' Wb/rad;\n')
        f.write('    btf   = ' + str(self.Btcent) + ' T;\n')
        f.write('    rtf   = ' + str(self.Rcent) + ' m;\n')
        f.write('\n')
        f.write('    r(1:jm);\n')
        ctr = 0.0
        tmp = ''
        for i in np.arange(len(self.R)):
            if (ctr < 4):
                tmp = tmp + str('%17.8f' % self.R[i])
                ctr = ctr + 1.0
            else:
                tmp = tmp + str('%17.8f\n' % self.R[i])
                f.write(tmp)
                ctr = 0.0
                tmp = ''
        if (ctr != 0.0):
            f.write(tmp + '\n')
        f.write('')

        f.write('   z(1:km);\n')
        ctr = 0.0
        tmp = ''
        for i in np.arange(len(self.Z)):
            if (ctr < 4):

                tmp = tmp + str('%17.8f' % self.Z[i])
                ctr = ctr + 1.0
            else:
                tmp = tmp + str('%17.8f\n' % self.Z[i])
                f.write(tmp)
                ctr = 0.0
                tmp = ''
        if (ctr != 0.0):
            f.write(tmp + '\n')
        f.write('')

        f.write('     ((psi(j,k)-psib,j=1,jm),k=1,km)\n')
        ctr = 0.0
        tmp = ''
        for i in np.arange(len(self.Z)):
            for j in np.arange(len(self.R)):
                if (ctr < 4):
                    tmp = tmp + str('%17.8f' % (self.psi[i, j] - self.psi_bnd))
                    ctr = ctr + 1.0
                else:
                    tmp = tmp + str('%17.8f\n' % (self.psi[i, j] - self.psi_bnd))
                    f.write(tmp)
                    ctr = 0.0
                    tmp = ''
        if (ctr != 0.0):
            f.write(tmp + '\n')

        f.close()

    def __calc_psi_deriv(self, method='CD'):
        """
        Calculate the derivatives of the poloidal flux on the grid in R and Z
        """
        if self.psi is None or self.R is None or self.Z is None:
            print("ERROR: Not enough information to calculate grad(psi). Returning.")
            return

        if method is 'CD':  # 2nd order central differencing on an interpolated grid
            R = np.linspace(np.min(self.R), np.max(self.R), 200)
            Z = np.linspace(np.min(self.Z), np.max(self.Z), 200)
            Rgrid, Zgrid = R, Z  # np.meshgrid(R,Z)
            psi = self.psi(Rgrid, Zgrid)
            deriv = np.gradient(psi)  # gradient wrt index
            # Note np.gradient gives y derivative first, then x derivative
            ddR = deriv[1]
            # ddR = self.psi(Rgrid,Zgrid,dx=1)
            ddZ = deriv[0]
            # ddZ = self.psi(Rgrid,Zgrid,dy=1)
            dRdi = 1.0 / np.gradient(R)
            dZdi = 1.0 / np.gradient(Z)
            dpsidR = ddR * dRdi[np.newaxis, :]  # Ensure broadcasting is handled correctly
            dpsidZ = ddZ * dZdi[:, np.newaxis]
            dpsidR_func = interp2d(R, Z, dpsidR)
            dpsidZ_func = interp2d(R, Z, dpsidZ)

            RR, ZZ = self.R, self.Z
            self.dpsidR = equilibriumField(np.transpose(dpsidR_func(RR, ZZ)), dpsidR_func)
            self.dpsidZ = equilibriumField(np.transpose(dpsidZ_func(RR, ZZ)), dpsidZ_func)
        else:
            print("ERROR: Derivative method not implemented yet, reverting to CD")
            self.calc_psi_deriv(method='CD')

    def calc_bfield(self):

        """Calculate magnetic field components"""

        self.__calc_psi_deriv()

        BR = -1.0 * self.dpsidZ / self.R[np.newaxis, :]
        BZ = self.dpsidR / self.R[np.newaxis, :]
        Bp = self.sigBp * (BR ** 2.0 + BZ ** 2.0) ** 0.5

        self.__get_fpolRZ()
        Bt = self.fpolRZ / self.R[np.newaxis, :]
        B = (BR ** 2.0 + BZ ** 2.0 + Bt ** 2.0) ** 0.5

        BR_func = interp2d(self.R, self.Z, BR)
        BZ_func = interp2d(self.R, self.Z, BZ)
        Bp_func = interp2d(self.R, self.Z, Bp)
        Bt_func = interp2d(self.R, self.Z, Bt)
        B_func = interp2d(self.R, self.Z, B)

        self.BR = equilibriumField(BR, BR_func)
        self.BZ = equilibriumField(BZ, BZ_func)
        self.Bp = equilibriumField(Bp, Bp_func)
        self.Bt = equilibriumField(Bt, Bt_func)
        self.B = equilibriumField(B, B_func)

    def __get_fpolRZ(self, plasma_response=False):
        """
        Generate fpol on the RZ grid given fpol(psi) and psi(RZ)
        fpol(psi) is given on an evenly spaced grid from psi_axis to psi_bnd. This means that
        some regions of psi will exceed the range that fpol is calculated over.
        When this occurs (usually in the SOL) we assume that the plasma contribution
        is negligable and instead just take the vacuum assumption for the B-field and
        reverse engineer

        """
        from scipy.interpolate import interp1d

        fpolRZ = np.zeros((self.nz, self.nr))

        if plasma_response and self.fpol is not None:
            psigrid = np.linspace(self.psi_axis, self.psi_bnd, len(self.fpol))
            for i in np.arange(self.nr):
                for j in np.arange(self.nz):
                    if self.psi[i, j] < psigrid[-1] and self.psi[i, j] > psigrid[0]:
                        fpolRZ[i, j] = self.fpol(self.psi[i, j])
                    else:
                        fpolRZ[i, j] = self.Btcent * self.Rcent

        else:
            fpolRZ[:, :] = self.Btcent * self.Rcent

        fpolRZ_func = interp2d(self.R, self.Z, fpolRZ)
        self.fpolRZ = equilibriumField(fpolRZ, fpolRZ_func)


if __name__ == '__main__':
    # Load a MAST shot
    eq = equilibrium(device='MAST', shot=24409, time=0.35)

    # plot flux surfaces
    #eq.plot_flux()

    #eq.plot_var(eq.dpsidR)

    eq.plot_var(eq.dpsidZ)

    # # Change the shot time
    # eq.set_time(0.4)
    #
    # # Replot flux surfaces
    # eq.plot_flux()
    #
    # # plot B-field on efit grid
    # eq.plot_var(eq.B)

    # plot B-field on refined grid
    RR, ZZ = np.linspace(np.min(eq.R), np.max(eq.R), 100), np.linspace(np.min(eq.Z), np.max(eq.Z), 200)
    import matplotlib.pyplot as plt

    plt.contourf(RR, ZZ, eq.B(RR, ZZ))
    plt.show()

    plt.contour(RR, ZZ, eq.psiN(RR, ZZ))
    plt.xlim(-2, 2)
    plt.show()

    eq.dump_geqdsk("g.test")
    eq2 = equilibrium(gfile="g.test")
    eq2.plot_flux()

