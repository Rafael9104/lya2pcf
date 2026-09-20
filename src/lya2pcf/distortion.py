"""
    Computes the distortion matrix using the deltas. Works the same way
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
from . import distortion_procedures_pycuda as distortion


def main():

    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()
    cuda_device = str(int(mpi_rank%params.number_of_cuda_devices + params.cuda_device_first_number))
    os.environ['CUDA_DEVICE'] = cuda_device

    # Writing log files, one per mpi process
    os.makedirs(params.corr_dir, exist_ok=True)
    log_filename = os.path.join(params.corr_dir, 'thread_' + str(mpi_rank) + '_of_' + str(mpi_size) + '_distortion.log')
    log_file = open(log_filename,"w+")

    if mpi_rank == 0:
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                description='Takes the data*.npy files and computes the distortion matrix of the two point correlation funcion.')

        parser.add_argument('--excluded', default = 0.95, required = False,
                help = 'Fraction of forests pairs excluded from the computation.')

        parser.add_argument('--chunks', type=int, default = None, required = False,
                help = 'Number of chunks the pixels are divided into. Each rank processes its chunks one '
                'after the other, loading only that chunk\'s files and freeing them before the next, '
                'so with one rank (one GPU) the chunks run sequentially and the memory needed is one '
                'chunk\'s, not the whole dataset\'s. Default: one chunk per MPI rank.')

        parser.add_argument('--only-chunk', type=int, default = None, required = False,
                help = 'Process just this one chunk (0 to chunks-1) and exit. Useful to test whether the '
                'largest chunk fits on the GPU, or to debug a single chunk.')

        parser.add_argument('--verbose', action = 'store_true', required = False,
                help = 'Show statistics of computation time. Only computes the distortion matrix for a few forests.')

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
    total_bins = np.prod(shape_hist)
    disto = np.zeros((total_bins,total_bins))
    weight_A = np.zeros(total_bins)

    print('Maximum angle between pairs of skewers that are used (rad):', angmax)
    print('Rank', mpi_rank, 'processes chunks', list(my_chunks), 'of', num_chunks)
    log_file.write('\nThis rank processes chunks ' + str(list(my_chunks)) + ' of ' + str(num_chunks) + '.')

    # disto and weight_A keep accumulating across chunks, so the result is the
    # same however the pixels are grouped.
    finished = False
    for chunk in my_chunks:

        owned_pixels = pixel_partition.assign_pixels(index['pixel_file'], num_chunks, chunk)
        log_file.write('\nChunk ' + str(chunk) + ' owns ' + str(len(owned_pixels)) + ' pixels.')
        if len(owned_pixels) == 0:
            log_file.write('\nNo pixels in this chunk; skipping.')
            continue

        buffer_pixels = pixel_partition.find_buffer_pixels(owned_pixels, angmax, set(index['pixel_file']))
        log_file.write('\nLoaded a buffer of ' + str(len(buffer_pixels)) + ' neighbouring pixels from other files.')
        data = pixel_partition.load_rank_data(params.data_dir, owned_pixels, buffer_pixels, index['pixel_file'])

        distortion.init(data, log_file, shape_hist, angmax, float(args.excluded))

        ###############################################################################
        # This is the core of the program, where the distortion matrix is computed    #
        ###############################################################################
        num_pixels_partial = len(owned_pixels)
        log_file.write('\nThis chunk computes ' + str(num_pixels_partial) + ' pixels, which go from ' +
            str(owned_pixels[0]) + ' to ' + str(owned_pixels[-1]))
        log_file.flush()

        pixel_counter = 0
        for pixel in owned_pixels:

            log_file.write('\nChunk ' + str(chunk) + ': computing pixel ' + str(pixel) + ', completed ' + str(int(pixel_counter/num_pixels_partial*100)) + '%')
            log_file.flush()

            disto_pix, weight_pix = distortion.distortion_per_pixel(data[pixel], **kwargs)
            disto += disto_pix
            weight_A += weight_pix

            pixel_counter += 1
            if args.verbose and pixel_counter > 1:
                print('Exiting early due to --verbose option.')
                finished = True
                break

        # Free this chunk's device and host memory before loading the next one.
        distortion.release()
        del data
        if finished:
            break

    if distortion.forests_seen > 0:
        clamp_message = ('%d of %d forests (%.2f%%) had more neighbours than number_of_neighs=%d after the '
            'exclusion and were capped.' % (distortion.clamped_forests, distortion.forests_seen,
            100.*distortion.clamped_forests/distortion.forests_seen, params.number_of_neighs))
        print('Rank', mpi_rank, clamp_message)
        log_file.write('\n' + clamp_message)
        log_file.flush()
    print('Finished distortion computation.')
    if mpi_size > 1:
        distortion_total = comm.reduce(disto)
        weight_total = comm.reduce(weight_A)
    else:
        distortion_total  = disto
        weight_total = weight_A

    if mpi_rank == 0:
            np.save(os.path.join(params.corr_dir, 'distortion'), distortion_total/weight_total[:, None])


if __name__ == '__main__':
    main()
