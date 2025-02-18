"""
    Takes delta files and computes the correlation
"""

import argparse
import glob
import os
import time

from parameters import *
from forest_class import quasar

from mpi4py import MPI


if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()
    cuda_device = str(int(mpi_rank%number_of_cuda_devices + cuda_device_first_number))
    os.environ['CUDA_DEVICE'] = cuda_device


    # from correlation_procedures import *
    # Writing log files, one per mpi process
    if not os.path.exists(corr_dir):
        os.makedirs(corr_dir)
    log_filename = corr_dir + 'thread_' + str(mpi_rank) + '_of_' + str(mpi_size) + '.log'
    log_file = open(log_filename,"w+")

    # global data
    if mpi_rank == 0:
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Takes the data.npy file and computes the two or three point correlation function.')

        group2 = parser.add_mutually_exclusive_group(required=True)
        group2.add_argument('--cpu',action='store_true',required=False,
            help='Compute the forest correlation using the cpu.')
        group2.add_argument('--gpu',action='store_true',required=False,
            help='Compute the forest correlation with the help of a GPU.')


        parser.add_argument('--verbose', action = 'store_true', required = False,
            help = 'Show statistics of computation time. Only computes the correlation for a few forests.')

        args = parser.parse_args()

        kwargs = {}
        if args.verbose:
            kwargs['performance'] = True

        print('Loading extracted file.')
        data = np.load(data_dir + 'data.npy', allow_pickle=True).item()


        ####################################################################
        #  Computing some important parameters and broadcasting the data  #
        #  to all the nodes if we are using several nodes.
        ####################################################################

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
        shape_hist = (numpix_rp, numpix_rt)

        print('Minimum comoving distance to a forest (Mpc/h):',dmin)
        print('Maximum angle between pairs of skewers that are used (rad):', angmax)

        pixels_total=np.array(list(data.keys()))
        lpix = len(pixels_total)
        print('Number of non-empty healpix pixels:', lpix)
        log_file.write('\nNumber of non-empty healpix pixels: ' + str(lpix))

        print('Computing partial histograms for each pixel.')

        # Dividing the total number of pixels between the available mpi kernels
        pixels_partial = np.array_split(pixels_total, mpi_size)

        if distributed_memory:

            # TODO: The data dictionary is too big to be broadcasted to other nodes as a single object
            # this is a dirty solution

            chunks = 2
            pixels_for_broadcast = np.array_split(pixels_total, chunks)
            data_for_broadcast_0 = {key: data[key] for key in pixels_for_broadcast[0]}
            data_for_broadcast_1 = {key: data[key] for key in pixels_for_broadcast[1]}

    else:
        angmax = None
        pixels_partial = None
        pixels_total = None
        data = None
        args = None
        kwargs = None
        shape_hist = None
        if distributed_memory:
            data_for_broadcast_0 = None
            data_for_broadcast_1 = None


    # Moving the relevant data to the other nodes
    if distributed_memory:
        data_for_broadcast_0 = comm.bcast(data_for_broadcast_0, root = 0)
        data_for_broadcast_1 = comm.bcast(data_for_broadcast_1, root = 0)
        data = {**data_for_broadcast_0, **data_for_broadcast_1}
    else:
        data = comm.bcast(data, root = 0)

    pixels_total = comm.bcast(pixels_total, root = 0)
    args = comm.bcast(args, root = 0)
    angmax = comm.bcast(angmax, root = 0)
    shape_hist = comm.bcast(shape_hist, root = 0)
    kwargs = comm.bcast(kwargs, root = 0)
    pixels_partial = comm.scatter(pixels_partial, root = 0)




    # Moving data dict to the correlation_procedures module
    if args.cpu:
        import correlation_procedures_cpu as correlations
    else:
        import correlation_procedures_pycuda as correlations
    correlations.init(data, log_file, shape_hist, angmax)

    num_pixels_partial = len(pixels_partial)
    log_file.write('\nThis process computes ' + str(num_pixels_partial) + ' pixels, which go from ' +
        str(pixels_partial[0]) + ' to ' + str(pixels_partial[-1]))

    name_partials = '2d_histogram_pixel_'
    log_file.write('\nComputing 2 point correlation with \nrt_max = ' + str(rtmax) +
     '\nrp_max = ' + str(rpmax) + '\npixels in t = ' + str(numpix_rt) + '\npixels in p = ' + str(numpix_rp))

    log_file.flush()

    ###############################################################################
    # This is the core of the program, where the correlation function is computed #
    ###############################################################################
    
    histo = []
    pixel_counter = 0

    for pixel in pixels_partial:
      # if pixel == 6088:

        log_file.write('\nComputing pixel ' + str(pixel) + ', completed ' + str(int(pixel_counter/num_pixels_partial*100)) + '%')
        log_file.flush()

        histo = correlations.two_point_per_pixel(pixel, **kwargs)

        np.save(corr_dir + name_partials + str(pixel), histo)
        pixel_counter += 1
        if args.verbose and pixel_counter > 1:
            print('Exiting early due to --verbose option.')
            break

    print('Finished correlation computation.')
