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
    owned_pixels = pixel_partition.assign_pixels(index['pixel_file'], mpi_size, mpi_rank)
    angmax = 2*np.arcsin(0.5*params.rtmax/index['min_distance'])
    shape_hist = (params.numpix_rp, params.numpix_rt)
    total_bins = np.prod(shape_hist)
    disto = np.zeros((total_bins,total_bins))
    weight_A = np.zeros(total_bins)

    print('Maximum angle between pairs of skewers that are used (rad):', angmax)
    print('Rank', mpi_rank, 'owns', len(owned_pixels), 'pixels.')
    log_file.write('\nThis rank owns ' + str(len(owned_pixels)) + ' pixels.')

    if len(owned_pixels) > 0:
        buffer_pixels = pixel_partition.find_buffer_pixels(owned_pixels, angmax, set(index['pixel_file']))
        log_file.write('\nFound a buffer of ' + str(len(buffer_pixels)) + ' neighbouring pixels from other files.')

        # Streamed to the GPU one data file at a time; see streaming_upload.py.
        plan = pixel_partition.plan_rank_data(params.data_dir, index, owned_pixels, buffer_pixels)
        log_file.write('\nStreaming ' + str(plan.count_forests) + ' forests from ' + str(len(plan.files)) + ' data files to the GPU.')
        data = distortion.init(None, log_file, shape_hist, angmax, float(args.excluded), plan = plan)

        ###############################################################################
        # This is the core of the program, where the distortion matrix is computed    #
        ###############################################################################
        num_pixels_partial = len(owned_pixels)
        log_file.write('\nThis process computes ' + str(num_pixels_partial) + ' pixels, which go from ' +
            str(owned_pixels[0]) + ' to ' + str(owned_pixels[-1]))
        log_file.flush()

        pixel_counter = 0
        for pixel in owned_pixels:

            log_file.write('\nComputing pixel ' + str(pixel) + ', completed ' + str(int(pixel_counter/num_pixels_partial*100)) + '%')
            log_file.flush()

            disto_pix, weight_pix = distortion.distortion_per_pixel(data[pixel], **kwargs)
            disto += disto_pix
            weight_A += weight_pix

            pixel_counter += 1
            if args.verbose and pixel_counter > 1:
                print('Exiting early due to --verbose option.')
                break
    else:
        log_file.write('\nNo pixels assigned to this rank; nothing to do.')
        print('Rank', mpi_rank, 'has no pixels to compute.')

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
