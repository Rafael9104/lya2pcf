"""
    This program takes the histogram files computed from 2pla.py and outputs the
    correlation, covariance matrix and plots.
"""
import glob
import argparse
import numpy as np
from parameters import *
from numba.core.decorators import jit

@jit(nopython=True, nogil=True)
def bin_coordinates(numpix_r_p, numpix_r_t):
    r_p_l = list()
    r_t_l = list()

    for label in range(1, numpix_r_p + 1, 1):
        for label2 in range(1, numpix_r_t + 1, 1):
            r_p_auxiliar = label
            r_t_auxiliar = label2
            r_t_l.append(r_t_auxiliar)
            r_p_l.append(r_p_auxiliar)
        
    r_p = np.array(r_p_l)
    r_t = np.array(r_t_l)

    return r_p, r_t

@jit(nopython=True, nogil=True)
def cov(xi, weight):
    num_elements = len(xi[0])
    xi_weight = xi*weight

    means_l = list()
    for i in range(len(xi[0])):
        mean_auxiliar = np.sum(xi_weight[:,i]) / np.sum(weight[:,i])
        means_l.append(mean_auxiliar)
    means = np.array(means_l)

    covariance_matrix = np.zeros((num_elements, num_elements))
    for label in range(len(xi[0])):
        bins_product = (weight[:,label]*(xi[:,label] - means[label]))*(weight[:,label]*(xi[:,label] - means[label]))
        covariance_matrix[label, label] = np.sum(bins_product)/(np.sum(weight[:,label])*np.sum(weight[:,label]))
        for label2 in range(label + 1, len(xi[0])):
            bins_product2 = (weight[:,label]*(xi[:,label] - means[label]))*(weight[:,label2]*(xi[:,label2] - means[label2]))
            covariance_matrix[label, label2] = np.sum(bins_product2)/(np.sum(weight[:,label])*np.sum(weight[:,label2]))
            covariance_matrix[label2, label] = covariance_matrix[label, label2]
            
    print("Computed covariance matrix...")
    
    return covariance_matrix, means

def cov_smooth(xi, weight, r_p, r_t, covariance_matrix):
        
    variance = np.diagonal(covariance_matrix)
    if np.any(variance == 0.):
        print('WARNING: Data has some empty bins. Returning the unsmoothed covariance')
        return covariance_matrix

    normalization_factor = np.sqrt(np.outer(variance, variance)) 
    correlation_matrix = covariance_matrix / normalization_factor
    
    num_elements = variance.shape[0]
    num_bins = np.sqrt(len(xi[0]))
    
    correlation_smooth = np.zeros((num_elements, num_elements))
    correlation_sum = np.zeros((int(num_bins), int(num_bins)))
    correlation_counts = np.zeros((int(num_bins), int(num_bins)))

    #add together the elements of correlation matrix with similar separations from parallel and
    #perpendicular distances. We consider those separations like parallel and perpendicular steps.

    for label in range(num_elements):
        for label2 in range(label + 1, num_elements):
            step_r_p = round(abs(r_p[label2] - r_p[label]))
            step_r_t = round(abs(r_t[label2] - r_t[label]))
            correlation_sum[step_r_p, step_r_t] += correlation_matrix[label, label2]
            correlation_counts[step_r_p, step_r_t] += 1

    for label in range(num_elements):
        correlation_smooth[label, label] = correlation_matrix[label, label]
        for label2 in range(label + 1, num_elements):
            step_r_p = round(abs(r_p[label2] - r_p[label]))
            step_r_t = round(abs(r_t[label2] - r_t[label]))
            correlation_smooth[label, label2] = (correlation_sum[step_r_p, step_r_t] / correlation_counts[step_r_p, step_r_t])
            correlation_smooth[label2, label] = correlation_smooth[label, label2]

    covariance_smooth = correlation_smooth * normalization_factor
    
    return  covariance_smooth

# For large matrices
def diag_error(da,we):

    mda = (da*we).sum(axis=0)
    swe = we.sum(axis=0)
    w = swe>0.
    mda[w] /= swe[w]

    wda = we*(da-mda)

    print("Computing errors...")

    error2 = (wda**2).sum(axis=0)
    error = np.sqrt(error2)
    error[w] /= swe[w]

    return error, mda


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='This program takes the histogram files computed from 2pla.py and outputs the correlation.')
    parser.add_argument('--write-coordinates', action = 'store_true', required = False,
        help = 'Write arrays with the central values of the coordinates to each point of the correlation function.')
    parser.add_argument('--diagonal-error', action = 'store_true', required = False,
        help = 'For large number of bins, specially in the three-point correlation, the errors are estimated without the full covariance.')
    args = parser.parse_args()

    shape_hist = (numpix_rp, numpix_rt)
    name_partials = '2d_histogram_pixel_*'
    # Name of the outputs
    cor_name_file = 'correlation_2d'
    error_name_file = 'error_2d'


    print('Looking for histogram files in ' + corr_dir)
    histogram_files = glob.glob(corr_dir + name_partials)
    total_files = len(histogram_files)
    print('In total ' + str(total_files) + ' files were found.')


    we_list = []
    cor_list = []
    counter = 0
    for file in histogram_files:
        w_hist, dw_hist = np.load(file)
        if counter == 0:
            shape = w_hist.shape

        w_hist = w_hist.flatten()
        dw_hist = dw_hist.flatten()
        we_list.append(w_hist)
        # Computing the partial correlation for one pixel
        cor_list.append(np.divide(dw_hist, w_hist, out = np.zeros_like(dw_hist), where = w_hist != 0))
        counter += 1
        print('Reading progress ' + str(int(counter/total_files*100)) + '%')
    da = np.array(cor_list)
    we = np.array(we_list)
    if args.diagonal_error:
        print('We are computing diagonal error, not the full covariance.')
        error, correlation = diag_error(da, we)
    else:
        covariance_not_smooth, correlation = cov(da, we)
        bin_r_p, bin_r_t = bin_coordinates(numpix_rp, numpix_rt)
        covariance = cov_smooth(da, we, bin_r_p, bin_r_t, covariance_not_smooth)
        error = np.sqrt(np.diagonal(covariance))
        np.save(corr_dir + 'covariance', covariance)
        print('The smoothed covariance was saved.')
        np.save(corr_dir + 'covariance_not_smooth', covariance_not_smooth)
        print('The unsmoothed covariance was saved.')

    # We reshape the correlation and error arrays to their original shape.
    correlation = np.reshape(correlation, shape)
    error = np.reshape(error, shape)
    np.save(corr_dir + cor_name_file, correlation)
    np.save(corr_dir + error_name_file, error)
    print('The correlation and error were saved.')

    # The following coordinates correspond to the center of the bins. They are not the
    # weighted averages because we don't have enough computer power to compute them
    if args.write_coordinates:
        print('Writing the coordinates. They are not necessary, but might be useful if you are doing your own analysis.')
        rp = correlation.copy()
        rt = correlation.copy()
        for i in range(numpix_rp):
            for j in range(numpix_rt):
                rp[i,j]=(i + 0.5)*rpmax / numpix_rp
                rt[i,j]=(j + 0.5)*rtmax / numpix_rt
        np.save(corr_dir + 'rp', rp)
        np.save(corr_dir + 'rt', rt)
 
