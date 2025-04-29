"""
   Computes the distortion matrix using the deltas
"""

import argparse
import glob
import os
import time
from mpi4py import MPI

from parameters import *
from forest_class import quasar
import distortion_procedures_pycuda as distortion


if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()
    cuda_device = str(int(mpi_rank%number_of_cuda_devices + cuda_device_first_number))
    os.environ['CUDA_DEVICE'] = cuda_device

    # Writing log files, one per mpi process
    if not os.path.exists(corr_dir):
        os.makedirs(corr_dir)

    log_filename = corr_dir + 'thread_' + str(mpi_rank) + '_of_' + str(mpi_size) + '_distortion.log'
    log_file = open(log_filename,"w+")

    # global data
    if mpi_rank == 0:
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Takes the data.npy file and computes the distortion matrix of the two point correlation funcion.')

        parser.add_argument('--excluded', default = 0.95, required = False,
            help = 'Fraction of forests pairs excluded from the computation.')

        parser.add_argument('--verbose', action = 'store_true', required = False,
            help = 'Show statistics of computation time. Only computes the distortion matrix for a few forests.')

        args = parser.parse_args()

        kwargs = {}
        if args.verbose:
            kwargs['performance'] = True

    print('Loading extracted file.')
    data = np.load(data_dir + 'data1.npy', allow_pickle=True).item()

    # Moving data dict to the correlation_procedures module

    name_partials = 'distortion_pixel_'

    log_file.flush()

    shape_hist = (numpix_rp, numpix_rt)
    total_bins = np.prod(shape_hist)
    disto = np.zeros((total_bins,total_bins))
    weight_A = np.zeros(total_bins)

    pixels_total=np.array(list(data.keys()))
    lpix = len(pixels_total)
    print('Number of non-empty healpix pixels:', lpix)
    log_file.write('\nNumber of non-empty healpix pixels: ' + str(lpix))

    print('Computing partial histograms for each pixel.')

    # Dividing the total number of pixels between the available mpi kernels
    pixels_partial = np.array_split(pixels_total, mpi_size)
    pixels_list = comm.scatter(pixels_partial, root = 0)

    print("The process number:", mpi_rank, "is going to compute the following pixels:", pixels_list)
    print('Computing the maximum angle that we are interested in.')
    log_file.write('\nComputing the maximum angle that we are interested in.')

    dminlist=[]
    for pixel in data:
        for forest in data[pixel]:
            dminlist.append(forest.dc[0])
    try:
        dmin  = min(dminlist)
    except:
        raise NameError('Empty data. Did you write correctly the input directory?')

    angmax = 2*np.arcsin(0.5*rtmax/dmin)
    # We are not considering the case where \xi and \hat{\xi} have different dimensions
    # All: Read shape from correlation

    print('Minimum comoving distance to a forest (Mpc/h):',dmin)
    print('Maximum angle between pairs of skewers that are used (rad):', angmax)

    distortion.init(data, log_file, shape_hist, angmax, float(args.excluded))

    num_pixels_partial = len(pixels_list)

    lpix = len(pixels_list)
    print('Number of non-empty healpix pixels:', lpix)
    log_file.write('\nNumber of non-empty healpix pixels: ' + str(lpix))

    print('Computing partial histograms for each pixel.')

    # Dividing the total number of pixels between the available mpi kernels
    pixels_partial = np.array_split(pixels_list, mpi_size)
    log_file.write('\nThis process computes ' + str(num_pixels_partial) + ' pixels, which go from ' +
        str(pixels_list[0]) + ' to ' + str(pixels_list[-1]))


    ###############################################################################
    # This is the core of the program, where the distortion matrix is computed    #
    ###############################################################################

    pixel_counter = 0
    for pixel in pixels_list:

        log_file.write('\nComputing pixel ' + str(pixel) + ', completed ' + str(int(pixel_counter/num_pixels_partial*100)) + '%')
        log_file.flush()

        disto_pix, weight_pix = distortion.distortion_per_pixel(data[pixel], **kwargs)
        disto += disto_pix
        weight_A += weight_pix

        pixel_counter += 1
        if args.verbose and pixel_counter > 1:
            print('Exiting early due to --verbose option.')
            break
    #print("Finised data file"+datafile)

    print('Finished distortion computation.')
    if mpi_size > 1:
        distortion_total = comm.reduce(disto)
        weight_total = comm.reduce(weight_A)
    else:
        distortion_total  = disto
        weight_total = weight_A

    if mpi_rank == 0:
            np.save(corr_dir + 'distortion', distortion_total/weight_total[:, None])
