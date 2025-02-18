import numpy as np
import time
from parameters import *
import math
from numba.core.decorators import jit

def init(data_aux, log_file_aux, shape_hist_aux, angmax_aux):
    global data
    global log_file
    global shape_hist
    global angmax

    data = data_aux
    log_file = log_file_aux
    shape_hist = shape_hist_aux
    angmax = angmax_aux


def two_point_per_pixel(pixel, **kargs):
    """ This function computes the weighted sum of w and delta*w for all pairs of data
    and stores them in histograms to prepare for the correlation function. The
    histograms are stored by healpix pixel of the first element in the pair.
    Parammeters:
    pixel   int
            The healpix pixel of the first element in the pair.
    angmax real
            Maximum angle between to forests to fit in the histogram.
    shape_hist  array int (np, nt)
            Shape of the histogram in bits
    """
    # Preparing data structure for the partial histograms
    w_hist  = np.zeros(shape_hist)
    dw_hist = np.zeros(shape_hist)

    for forest1 in data[pixel]:
        # Looking for neighbors
        neighbors = forest1.neighborhood(data, angmax)

        for forest2 in neighbors:
                w_hist_tmp, dw_hist_tmp = pair_correlation(angmax, forest1.ra,forest1.dec,forest1.we,forest1.dw,forest1.pl,forest1.dc,forest1.fib,forest2.ra,forest2.dec,forest2.we,forest2.dw,forest2.pl,forest2.dc,forest2.fib)
                w_hist += w_hist_tmp
                dw_hist += dw_hist_tmp
    return (w_hist,dw_hist)


@jit(nopython=True, nogil=True)
def pair_correlation(angmax, ra1,dec1,w1,dw1,pl1,dc1,fib1,ra2,dec2,w2,dw2,pl2,dc2,fib2):
    """ Computes the sum of w and delta*w for a pair of forests and stores it in
    a histogram according to their distance.
    Parammeters:
    angmax: Real           Maximum angle between forests to be considered in the histograms.
    ra1, dec1:  Real            Right assention and declination of the first forest.
    w1, dw1:    Array(Real)     Weight and delta times weight of the forest.
    pl1:    Int                 Plate id of the first forest, it is not used anymore.
    fib1:   Int                 Fib id of the first forest, it is not used anymore.
    Same parammeters for the second forest
    """
    w_hist  = np.zeros(shape_hist)
    dw_hist = np.zeros(shape_hist)
    len_this = len(dw1)
    if abs(ra1-ra2)<chiquito and abs(dec1-dec2)<chiquito:
        delta_theta = np.sqrt(((ra1-ra2)*np.cos(dec1))**2+(dec1-dec2)**2)
    else:
        delta_theta=np.arccos(np.sin(dec1)*np.sin(dec2)+np.cos(dec1)*np.cos(dec2)*np.cos(ra1-ra2))

    cos = np.cos(0.5*delta_theta)
    sin = np.sin(0.5*delta_theta)

    if delta_theta < angmax:
        len_that = len(dw2)
        for i in range(len_this):
            for j in range(len_that):
                rp = np.abs(dc1[i] - dc2[j])*cos
                rt = (dc1[i] + dc2[j])*sin
                binp = int(rp/rpmax*numpix_rp)
                bint = int(rt/rtmax*numpix_rt)

                w12 = w1[i]*w2[j]
                dw12 = dw1[i]*dw2[j]

                if binp < numpix_rp and bint < numpix_rt:
                    w_hist[binp, bint] += w12
                    dw_hist[binp, bint] += dw12
    return w_hist, dw_hist

