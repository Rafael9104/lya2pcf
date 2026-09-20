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

        parser.add_argument('--chunks', type=int, default = None, required = False,
            help = 'Number of chunks the pixels are divided into. Each rank processes its chunks one '
            'after the other, loading only that chunk\'s files and freeing them before the next, '
            'so with one rank (one GPU) the chunks run sequentially and the memory needed is one '
            'chunk\'s, not the whole dataset\'s. Default: one chunk per MPI rank.')

        parser.add_argument('--only-chunk', type=int, default = None, required = False,
            help = 'Process just this one chunk (0 to chunks-1) and exit. Useful to time or debug a single chunk.')

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
    num_chunks = args.chunks if args.chunks is not None else mpi_size
    if num_chunks < 1:
        raise ValueError('--chunks must be at least 1, got %d' % num_chunks)
    my_chunks = pixel_partition.rank_chunks(num_chunks, mpi_rank, mpi_size)
    if args.only_chunk is not None:
        my_chunks = [c for c in my_chunks if c == args.only_chunk]
    angmax = 2*np.arcsin(0.5*params.rtmax/index['min_distance'])
    shape_hist = (params.numpix_rp, params.numpix_rt)

    print('Maximum angle between pairs of skewers that are used (rad):', angmax)
    print('Rank', mpi_rank, 'processes chunks', list(my_chunks), 'of', num_chunks)

    name_partials = '2d_histogram_pixel_'
    log_file.write('\nComputing 2 point correlation with \nrt_max = ' + str(params.rtmax) +
     '\nrp_max = ' + str(params.rpmax) + '\npixels in t = ' + str(params.numpix_rt) + '\npixels in p = ' + str(params.numpix_rp))
    log_file.write('\nThis rank processes chunks ' + str(list(my_chunks)) + ' of ' + str(num_chunks) + '.')

    if args.cpu:
        from . import correlation_procedures_cpu as correlations
    else:
        from . import correlation_procedures_pycuda as correlations

    # Wall time per phase, to see where a run actually spends its time.
    # load: reading + unpickling the data*.npy files; upload: packing the
    # forests and copying them to the GPU; compute: the pair loop (neighbour
    # search on the CPU plus the kernels); save: writing the histograms.
    timers = {'load': 0., 'upload': 0., 'compute': 0., 'save': 0.}
    clock = time.perf_counter
    finished = False
    for chunk in my_chunks:

        owned_pixels = pixel_partition.assign_pixels(index['pixel_file'], num_chunks, chunk)
        log_file.write('\nChunk ' + str(chunk) + ' owns ' + str(len(owned_pixels)) + ' pixels.')
        if len(owned_pixels) == 0:
            log_file.write('\nNo pixels in this chunk; skipping.')
            continue

        t0 = clock()
        buffer_pixels = pixel_partition.find_buffer_pixels(owned_pixels, angmax, set(index['pixel_file']))
        log_file.write('\nLoaded a buffer of ' + str(len(buffer_pixels)) + ' neighbouring pixels from other files.')
        data = pixel_partition.load_rank_data(params.data_dir, owned_pixels, buffer_pixels, index['pixel_file'])
        t1 = clock()
        correlations.init(data, log_file, shape_hist, angmax)
        t2 = clock()
        timers['load'] += t1 - t0
        timers['upload'] += t2 - t1

        num_pixels_partial = len(owned_pixels)
        log_file.write('\nThis chunk computes ' + str(num_pixels_partial) + ' pixels, which go from ' +
            str(owned_pixels[0]) + ' to ' + str(owned_pixels[-1]))
        log_file.flush()

        ###############################################################################
        # This is the core of the program, where the correlation function is computed #
        ###############################################################################

        pixel_counter = 0

        for pixel in owned_pixels:

            log_file.write('\nChunk ' + str(chunk) + ': computing pixel ' + str(pixel) + ', completed ' + str(int(pixel_counter/num_pixels_partial*100)) + '%')
            log_file.flush()

            t3 = clock()
            histo = correlations.two_point_per_pixel(pixel, **kwargs)
            t4 = clock()
            np.save(os.path.join(params.corr_dir, name_partials + str(pixel)), histo)
            timers['compute'] += t4 - t3
            timers['save'] += clock() - t4
            pixel_counter += 1
            if args.verbose and pixel_counter > 1:
                print('Exiting early due to --verbose option.')
                finished = True
                break

        # Free this chunk's device and host memory before loading the next one.
        correlations.release()
        del data
        if finished:
            break

    total = sum(timers.values())
    summary = 'Time by phase: ' + ', '.join(
        '%s %.1f s (%.0f%%)' % (name, seconds, 100*seconds/total if total > 0 else 0)
        for name, seconds in timers.items()) + ', total %.1f s' % total
    print(summary)
    log_file.write('\n' + summary)
    log_file.flush()
    print('Finished correlation computation.')


if __name__ == '__main__':
    main()
