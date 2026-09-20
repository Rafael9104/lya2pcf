"""
    Takes delta files and computes the correlation. Works the same way
    whether the extraction wrote one data*.npy file or many
    (delta_reader.py --split-number): pixels are always split across MPI
    ranks, and each rank loads only the files its own pixels and their
    neighbour buffer actually need -- see pixel_partition.py.
"""

import argparse
import os
import time

import numpy as np
from mpi4py import MPI

from . import parameters as params
from .forest_class import quasar
from . import pixel_partition


def main():

    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()
    cuda_device = str(int(mpi_rank%params.number_of_cuda_devices + params.cuda_device_first_number))
    os.environ['CUDA_DEVICE'] = cuda_device
    print('worker'+str(mpi_rank)+'will be using gpu number' +cuda_device)

    # Writing log files, one per mpi process
    os.makedirs(params.corr_dir, exist_ok=True)
    log_filename = os.path.join(params.corr_dir, 'thread_' + str(mpi_rank) + '_of_' + str(mpi_size) + '.log')
    log_file = open(log_filename,"w+")

    if mpi_rank == 0:
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description='Takes the data*.npy files and computes the two point correlation function.')

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
    else:
        args = None
        kwargs = None

    args = comm.bcast(args, root = 0)
    kwargs = comm.bcast(kwargs, root = 0)

    ####################################################################
    #  Splitting the pixels between the available mpi ranks, and       #
    #  figuring out which files (own pixels + neighbour buffer) this   #
    #  rank actually needs to load.                                    #
    ####################################################################

    index = pixel_partition.load_index(params.data_dir)
    owned_pixels = pixel_partition.assign_pixels(index['pixel_file'], mpi_size, mpi_rank)
    angmax = 2*np.arcsin(0.5*params.rtmax/index['min_distance'])
    shape_hist = (params.numpix_rp, params.numpix_rt)

    print('Maximum angle between pairs of skewers that are used (rad):', angmax)
    print('Rank', mpi_rank, 'owns', len(owned_pixels), 'pixels.')

    name_partials = '2d_histogram_pixel_'
    log_file.write('\nComputing 2 point correlation with \nrt_max = ' + str(params.rtmax) +
     '\nrp_max = ' + str(params.rpmax) + '\npixels in t = ' + str(params.numpix_rt) + '\npixels in p = ' + str(params.numpix_rp))
    log_file.write('\nThis rank owns ' + str(len(owned_pixels)) + ' pixels.')

    if len(owned_pixels) == 0:
        log_file.write('\nNo pixels assigned to this rank; nothing to do.')
        print('Rank', mpi_rank, 'has no pixels to compute; exiting.')
        return

    buffer_pixels = pixel_partition.find_buffer_pixels(owned_pixels, angmax, set(index['pixel_file']))
    log_file.write('\nFound a buffer of ' + str(len(buffer_pixels)) + ' neighbouring pixels from other files.')

    # The CPU path needs every forest's arrays in memory, so it loads them all. The GPU path
    # streams them to the GPU one data file at a time (streaming_upload.py) and keeps only
    # the light per-forest metadata in host memory.
    if args.cpu:
        from . import correlation_procedures_cpu as correlations
        data = pixel_partition.load_rank_data(params.data_dir, owned_pixels, buffer_pixels, index['pixel_file'])
        correlations.init(data, log_file, shape_hist, angmax)
    else:
        from . import correlation_procedures_pycuda as correlations
        plan = pixel_partition.plan_rank_data(params.data_dir, index, owned_pixels, buffer_pixels)
        log_file.write('\nStreaming ' + str(plan.count_forests) + ' forests from ' + str(len(plan.files)) + ' data files to the GPU.')
        correlations.init(None, log_file, shape_hist, angmax, plan = plan)

    num_pixels_partial = len(owned_pixels)
    log_file.write('\nThis process computes ' + str(num_pixels_partial) + ' pixels, which go from ' +
        str(owned_pixels[0]) + ' to ' + str(owned_pixels[-1]))

    log_file.flush()

    ###############################################################################
    # This is the core of the program, where the correlation function is computed #
    ###############################################################################

    histo = []
    pixel_counter = 0

    for pixel in owned_pixels:

        log_file.write('\nComputing pixel ' + str(pixel) + ', completed ' + str(int(pixel_counter/num_pixels_partial*100)) + '%')
        log_file.flush()

        histo = correlations.two_point_per_pixel(pixel, **kwargs)

        np.save(os.path.join(params.corr_dir, name_partials + str(pixel)), histo)
        pixel_counter += 1
        if args.verbose and pixel_counter > 1:
            print('Exiting early due to --verbose option.')
            break

    print('Finished correlation computation.')


if __name__ == '__main__':
    main()
