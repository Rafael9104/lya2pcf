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
import parameters as params

def missing_keys_message(file, missing, present, what):
    """Report configured key names that the delta file does not have."""
    return (
        "These names from delta_keys in the configuration are not in the "
        "delta file:\n"
        "\n"
        "  file: %s\n"
        "\n"
        "%s"
        "\n"
        "The %s in this file are:\n"
        "  %s\n"
        "\n"
        "Delta files differ between surveys and between blinded and "
        "unblinded productions -- blinded DESI files carry the deltas in "
        "DELTA_BLIND rather than DELTA, for instance. Set the matching "
        "names under delta_keys in parameters.yml."
        % (file,
           "".join("  delta_keys.%-9s = %r  (not found)\n" % (label, params.delta_keys[label])
                   for label in missing),
           what, ", ".join(present)))


def check_keys(deltafile, file):
    """Fail with a readable message if the configured keys do not match.

    Called once up front as well as per file, because an error raised inside
    a multiprocessing worker comes back wrapped in a RemoteTraceback, and
    because a directory can mix productions.
    """
    extensions = [hdu.get_extname() for hdu in deltafile]
    missing = [label for label in params.delta_hdu_keys
               if params.delta_keys[label] not in extensions]
    if missing:
        raise ValueError(missing_keys_message(file, missing,
                                              [e for e in extensions if e], "extensions"))

    columns = deltafile[params.delta_keys['metadata']].get_colnames()
    missing = [label for label in params.delta_column_keys
               if params.delta_keys[label] not in columns]
    if missing:
        raise ValueError(missing_keys_message(
            file, missing, columns,
            "columns in the %r table" % params.delta_keys['metadata']))


def record_from_deltas(file):
    """ Extracts all forests data from a single delta file to a list
    of objects of type quasar.

    file - deltafile*.fits.gz from PICCA
    """
    print('Extracting from file ',file)
    list_of_forests = []
    keys = params.delta_keys
    deltafile = fitsio.FITS(file)
    check_keys(deltafile, file)
    numberofforests,numberoflambdas = deltafile[keys['delta']].get_dims()
    # Each extension is read once per file and then indexed in memory. Reading
    # row by row instead costs a separate cfitsio call per forest, and the
    # deltas were being read twice over. The cost is holding one file's deltas
    # and weights per worker process while it runs.
    metadata = deltafile[keys['metadata']][:]
    lambd_list = deltafile[keys['lambda']][:]
    deltas = deltafile[keys['delta']].read()
    weights = deltafile[keys['weight']].read()
    for i in range(numberofforests):
        forest_data = quasar(metadata[keys['los_id']][i],
            metadata[keys['los_id']][i],
            metadata[keys['targetid']][i],
            metadata[keys['ra']][i],
            metadata[keys['dec']][i],
            numberoflambdas)
        delta_row = deltas[i]
        mask = np.isfinite(delta_row)
        lambd = lambd_list[mask]
        z = lambd/params.lambdaa - 1
        loglam = np.log10(lambd)
        correctionfactor=np.power((z + 1.)/(1. + params.z_ref), params.gammaovertwo)
        forest_data.we = weights[i][mask] * correctionfactor
        forest_data.fill_dw(delta_row[mask], loglam, True)
        #forest_data.dw = forest.data['WEIGHT']*forest.data['DELTA']*correctionfactor
        
        comov_distance = cosmology.dc_interpol(z)
        forest_data.dc = comov_distance
        forest_data.rx = forest_data.x * comov_distance
        forest_data.ry = forest_data.y * comov_distance
        forest_data.rz = forest_data.z * comov_distance
        list_of_forests.append(forest_data)
    return list_of_forests

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description='Takes delta files by picca and stores data in data.npy.')
parser.add_argument('--delta-dir', type=str, required=True,
    help = 'Path to the delta files.')
parser.add_argument('--data-dir', type=str, default = params.data_dir,
    help = 'Directory where the data will be stored.')
parser.add_argument('--split-number', type=int, default = 1,
    help = 'Number of files to split the data.')
args = parser.parse_args()

if os.path.exists(args.data_dir):
    warnings.warn('The output delta directory already exists. This procedure might mix deltas from a different run.')

if not os.path.exists(args.data_dir):
    os.makedirs(args.data_dir)

data = {}
directory = glob.glob(args.delta_dir + '/*.fits.gz')
if len(directory) == 0:
    print('No delta files in directory ' + args.delta_dir)

# Check the configured keys against one file before starting the workers, so a
# mismatch is reported plainly instead of through a RemoteTraceback.
if directory:
    with fitsio.FITS(directory[0]) as first_file:
        check_keys(first_file, directory[0])

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

angmax = 2*np.arcsin(0.5*params.rtmax/min_distance)
print('Minimum comoving distance to a forest (Mpc/h):',min_distance)
print('Maximum angle between pairs of skewers that are used (rad):', angmax)

list_of_pixels = list(data.keys())
list_of_pixels.sort()


# Here we will search for the neighbors of each forest
neighbors = []
for pix in list_of_pixels:
    for forest in data[pix]:
        min_distance = forest.dc[0]
        angmax = 2*np.arcsin(0.5*params.rtmax/min_distance)
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
    np.save(os.path.join(args.data_dir, 'data' + str(i)), subdata)
    i+=1
    for pixel in subset:
        data.pop(pixel)

print("The largest forest has ", max_lenght, " data points.")
print("The number of forests is:", j)
