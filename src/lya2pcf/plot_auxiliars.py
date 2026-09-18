import scipy as sp
from numba.core.decorators import jit
import numpy as np

@jit(nopython=True)
def weight_2d_wedge(mumin_wedge, mumax_wedge, rmin_wedge, rmax_wedge, shape_hist , rpmax, rtmax):
    coarse_weight = np.zeros(shape_hist)
    precision = 100
    fine_shape = np.array(shape_hist) * precision
    fine_rp = (np.arange(fine_shape[0]) + 0.5) / fine_shape[0] * rpmax
    fine_rt = (np.arange(fine_shape[1]) + 0.5) / fine_shape[1] * rtmax
    for i in range(fine_shape[0]):
        for j in range(fine_shape[1]):
            r = np.sqrt(fine_rp[i]**2 + fine_rt[j]**2)
            mu = fine_rp[i]/r
            if r < rmax_wedge and r > rmin_wedge and mu < mumax_wedge and mu > mumin_wedge:
                bini = int(i / precision)
                binj = int(j / precision)
                coarse_weight[bini,binj] += 1
    coarse_weight = coarse_weight / np.sum(coarse_weight)
    return coarse_weight

# Using weights from covariance
def wedges_weighted_co(mumin, mumax, shape_hist, points, rmax, rmaxp, rmaxt, correlation, covariance):
    rr = np.zeros(points)
    geometric_weight_wedge = []
    for i in range(points):
        rmin_bin = i / points * rmax
        rmax_bin = (i + 1) / points * rmax
        rr[i] = 0.5*(rmin_bin + rmax_bin)
        geometric_weight_wedge.append((weight_2d_wedge(mumin,mumax,rmin_bin,rmax_bin,shape_hist,rmaxp,rmaxt)).flatten())
    geometric_weight_wedge = np.array(geometric_weight_wedge)
    correlation_wedge, covariance_wedge = wedge(geometric_weight_wedge, correlation.flatten(), covariance)
    error2 = sp.diagonal(covariance_wedge)
    return rr, correlation_wedge, error2

# This function was adapted from PICCA picca/py/picca/wedgize.py
def wedge(geometric_weight,da,co):
    we = 1/sp.diagonal(co)
    w = geometric_weight.dot(we)
    Wwe = geometric_weight*we
    mask = w>0
    Wwe[mask,:]/=w[mask,None]
    d = Wwe.dot(da)
    return d,Wwe.dot(co).dot(Wwe.T)