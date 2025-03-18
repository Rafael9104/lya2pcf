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

import cosmology
from forest_class import quasar
from parameters import *

def record_from_deltas(file):
    """ Extracts all forests data from a single delta file to a list
    of objects of type quasar.

    file - deltafile*.fits.gz from PICCA
    """
    print('Extracting from file ',file)
    list_of_forests = []
    deltafile = fitsio.FITS(file)
    numberofforests,numberoflambdas = deltafile['DELTA'].get_dims()
    for i in range(numberofforests):
        metadata=deltafile["METADATA"][:]
        forest_data = quasar(metadata["LOS_ID"][i],
            metadata["LOS_ID"][i],
            metadata["TARGETID"][i],
            metadata["RA"][i],
            metadata["DEC"][i],
            numberoflambdas)
        delta1 = deltafile['DELTA'][i,:][0]
        mask = np.isfinite(delta1)
        lambd_list = deltafile['LAMBDA'][:]
        lambd = lambd_list[mask]
        z = lambd/lambdaa - 1
        loglam = np.log10(lambd)
        correctionfactor=np.power((z + 1.)/(1. + z_ref), gammaovertwo)
        weight_list = deltafile['WEIGHT'][i,:][0] 
        forest_data.we = weight_list[mask] * correctionfactor
        delta_list = deltafile['DELTA'][i,:][0]
        forest_data.fill_dw(delta_list[mask], loglam, True)
        #forest_data.dw = forest.data['WEIGHT']*forest.data['DELTA']*correctionfactor
        
        comov_distance = cosmology.dc_interpol(z)
        forest_data.dc = comov_distance
        forest_data.rx = forest_data.x * comov_distance
        forest_data.ry = forest_data.y * comov_distance
        forest_data.rz = forest_data.z * comov_distance
        list_of_forests.append(forest_data)
    return list_of_forests

def substitute_parameter(key,value):
    import fileinput

    for line in fileinput.input("parameters.py", inplace=True):
        if key in line:
            value_str = str(int(value))
            line = key + " = np.int32(" + value_str + ")\n"
        print('{}'.format(line), end='')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description='Takes delta files by picca and stores data in data.npy.')
parser.add_argument('--delta-dir', type=str, required=True,
    help = 'Path to the delta files.')
parser.add_argument('--data-dir', type=str, default = data_dir,
    help = 'Directory where the data will be stored.')
parser.add_argument('--split-number', type=int, default = 1,
    help = 'Number of files to split the data.')
args = parser.parse_args()

if os.path.exists(args.data_dir):
    warnings.warn('The directory deltas_lya2pcf exists. Erase the directory to continum.')
    quit()

if not os.path.exists(args.data_dir):
    os.makedirs(args.data_dir)

data = {}
directory = glob.glob(args.delta_dir + '/*.fits.gz')
if len(directory) == 0:
    print('No delta files in directory ' + delta_dir)

pool = Pool()
data_list = pool.map(record_from_deltas, directory)
max_lenght = 0
min_distance = 1e10
j=0

sizes = []

for list_of_forests in data_list:
    for forest_data in list_of_forests:
        j+=1
        new_long=len(forest_data.dc)
        if new_long>max_lenght:
            max_lenght = new_long
        if forest_data.dc[0] < min_distance:
            min_distance = forest_data.dc[0]
        if forest_data.pix in data.keys():
            data[forest_data.pix].append(forest_data)
        else:
            data[forest_data.pix] = [forest_data]
        sizes.append(new_long)
del data_list

angmax = 2*np.arcsin(0.5*rtmax/min_distance)
print('Minimum comoving distance to a forest (Mpc/h):',min_distance)
print('Maximum angle between pairs of skewers that are used (rad):', angmax)

list_of_pixels = list(data.keys())
list_of_pixels.sort()


# Here we will search for the neighbors of each forest
neighbors = []
for pix in list_of_pixels:
    for forest in data[pix]:
        min_distance = forest.dc[0]
        angmax = 2*np.arcsin(0.5*rtmax/min_distance)
        neigh_names, neigh_pixels = forest.neighborhood_names(data,angmax)
        forest.neigh_names = neigh_names
        forest.neigh_pixels = neigh_pixels
        number_neighs = len(neigh_names)
        neighbors.append(number_neighs)
np.savetxt("sizes", sizes)
np.savetxt("neighbors", neighbors)

pixels_partial = np.array_split(list_of_pixels, args.split_number)
i=1
for subset in pixels_partial:
    subdata = {x: data[x] for x in subset}
    np.save(args.data_dir+'/data'+str(i),subdata)
    i+=1
    for pixel in subset:
        data.pop(pixel)

substitute_parameter("max_lenght", max_lenght)
print("The largest forest has ", max_lenght, " data points.")
print("The number of forests is:", j)
