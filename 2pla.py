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

    

    # global data
    if mpi_rank == 0:
        directory_data = glob.glob(data_dir+"/data*.npy")
        directory_split = np.array_split(directory_data, mpi_size)
    else:
        directory_split = None
    directory_split = comm.scatter(directory_split, root = 0)


    # Moving data dict to the correlation_procedures module
    if args.cpu:
        import correlation_procedures_cpu as correlations
    else:
        import correlation_procedures_pycuda as correlations


    name_partials = '2d_histogram_pixel_'
    log_file.write('\nComputing 2 point correlation with \nrt_max = ' + str(rtmax) +
     '\nrp_max = ' + str(rpmax) + '\npixels in t = ' + str(numpix_rt) + '\npixels in p = ' + str(numpix_rp))
    
    log_file.flush()

    for datafile in directory_split:


        data = np.load(datafile, allow_pickle=True).item()
        pixels_list = np.array(list(data.keys()))
        print("I am node number:", mpi_rank, "I have the pixels", pixels_list)
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

        correlations.init(data, log_file, shape_hist, angmax)

        num_pixels_partial = len(pixels_list)
        log_file.write('\nThis process computes ' + str(num_pixels_partial) + ' pixels, which go from ' +
            str(pixels_list[0]) + ' to ' + str(pixels_list[-1]))


        ###############################################################################
        # This is the core of the program, where the correlation function is computed #
        ###############################################################################
        
        histo = []
        pixel_counter = 0

        for pixel in pixels_list:
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

        

        


