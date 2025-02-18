"""
    This module generates an interpolation function for d_c(z).
"""
import numpy as np
from scipy import integrate, interpolate
from numba.core.decorators import jit
from numba import vectorize, float64,float32
from parameters import *

@jit()
def E(z):
    """ 
    The inverse of the Hubble parameter normalized by H_0 and as a function of z.
    """
    return 1./np.sqrt((1. - OmDE)*(1. + z)**3 + OmDE)

@vectorize(["float32(float32)", "float64(float64)"], forceobj=True)
def d_c(z):
    """ 
    Comoving distance as a function of z, it
    recieves a list and outputs a list.
    """
    inte=integrate.quad(E,0,z)
    return d_H0 * inte[0]

ztable = np.linspace(zmin,zmax,nz)
d_ctable = d_c(ztable)
tck = interpolate.splrep(ztable, d_ctable, s=0)

def dc_interpol(z):
    """ Interpolation function for d_c(z) outputs a real number.
    """
    return interpolate.splev(z, tck, der=0)

