"""
This program takes the deltas in flux as computed by Picca or any other means with the same
format 'deltadir/*.fits.gz' and stores the relevant data in a single file data.npy
Parammeters:
   delta_dir Path to the delta files
   data_dir  Directory where the data will be stored

Folowing Picca, the object data to be stored in 'data_dir/data.npy' is a dictionary between
healpix pixels in the sky and a list of the quasars in that region:
data: pixel -> [quasars]
"""

import argparse
import glob
import numpy as np
import os
from multiprocessing import Pool
import fitsio
import warnings

from . import cosmology
from .forest_class import quasar
from . import parameters as params

def record_from_deltas(file):
    """ Extracts all forests data from a single delta file to a list
    of objects of type quasar.

    file - deltafile*.fits.gz from PICCA
    """
    print('Extracting from file ',file)
    list_of_forests = []
    deltafile = fitsio.FITS(file)
    for forest in deltafile[1:]:
        header = forest.read_header()
        forest_data = quasar(header['FIBERID'],
            header['PLATE'],
            header['THING_ID'],
            header['RA'],
            header['DEC'],
            len(forest['WEIGHT'][:]))
        
        loglam = forest['LOGLAM'][:]
        z = np.power(10, (loglam - params.la)) - 1.
        correctionfactor=np.power((z + 1.)/(1. + params.z_ref), params.gammaovertwo)
        forest_data.we = forest['WEIGHT'][:] * correctionfactor
        forest_data.fill_dw(forest['DELTA'][:], loglam, True)
        #forest_data.dw = forest.data['WEIGHT']*forest.data['DELTA']*correctionfactor
        
        comov_distance = cosmology.dc_interpol(z)
        forest_data.dc = comov_distance
        forest_data.rx = forest_data.x * comov_distance
        forest_data.ry = forest_data.y * comov_distance
        forest_data.rz = forest_data.z * comov_distance
        list_of_forests.append(forest_data)
    return list_of_forests

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Takes delta files by picca and stores data in data.npy.')
    parser.add_argument('--delta-dir', type=str, required=True,
        help = 'Path to the delta files.')
    parser.add_argument('--data-dir', type=str, default = params.data_dir,
        help = 'Directory where the data will be stored.')
    args = parser.parse_args()

    if os.path.exists(args.data_dir):
        warnings.warn('The output delta directory already exists. This procedure might mix deltas from a different run.')

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    data = {}
    directory = glob.glob(args.delta_dir + '/*.fits.gz')
    if len(directory) == 0:
        print('No delta files in directory ' + args.delta_dir)

    pool = Pool()
    data_list = pool.map(record_from_deltas, directory)
    min_distance = 1e10
    for list_of_forests in data_list:
        for forest_data in list_of_forests:
            if forest_data.dc[0] < min_distance:
                min_distance = forest_data.dc[0]
            if forest_data.pix in data.keys():
                data[forest_data.pix].append(forest_data)
            else:
                data[forest_data.pix] = [forest_data]
    np.save(os.path.join(args.data_dir, 'data1'), data)

    # See delta_reader.py's data_index.npy for what this is for -- the
    # multi-file drivers need it even for a single-file extraction like
    # this one, so they don't need a separate code path for that case.
    # pixel_count and max_lenght: see delta_reader.py.
    np.save(os.path.join(args.data_dir, 'data_index'),
            {'pixel_file': {int(p): 1 for p in data}, 'min_distance': min_distance,
             'pixel_count': {int(p): len(forests) for p, forests in data.items()},
             'max_lenght': int(max(len(f.dc) for forests in data.values() for f in forests))})


if __name__ == '__main__':
    main()
