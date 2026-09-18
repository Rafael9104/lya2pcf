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
import difflib
import glob
import numpy as np
import os
from multiprocessing import Pool
import fitsio
import healpy
import warnings

from . import cosmology
from .forest_class import quasar
from . import parameters as params
from . import pixel_partition

def suggest_name(configured, present):
    """Closest name in the file to the configured one, or None.

    Prefixes are checked before fuzzy matching, so DELTA suggests
    DELTA_BLIND rather than whichever name happens to score well.
    """
    others = [name for name in present if name != configured]
    prefixed = [name for name in others
                if name.startswith(configured) or configured.startswith(name)]
    if prefixed:
        return min(prefixed, key=len)
    close = difflib.get_close_matches(configured, others, n=1, cutoff=0.7)
    return close[0] if close else None


def missing_keys_message(file, missing, present, what):
    """Report configured key names that the delta file does not have."""
    lines = []
    for label in missing:
        configured = params.delta_keys[label]
        suggestion = suggest_name(configured, present)
        lines.append("  delta_keys.%-9s = %-14r not found%s\n"
                     % (label, configured,
                        "; did you mean %r ?" % suggestion if suggestion else ""))
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
        "Set the matching names under delta_keys in parameters.yml. If this "
        "is blinded DESI data, the deltas are in DELTA_BLIND rather than "
        "DELTA and nothing else in the file changes."
        % (file, "".join(lines), what, ", ".join(present)))


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


def record_pixel_only(file):
    """ Reads just the RA/DEC needed to know which output healpix pixel
    each forest in this file belongs to -- none of the delta/weight/lambda
    arrays record_from_deltas reads, so this is cheap enough to run over
    every file up front, before deciding how to split the dataset.

    One input file does not necessarily map to one output pixel at this
    nside (confirmed on the real DR1 set: one file's forests can land in
    several, e.g. one DR1 file spans 4 pixels at the default nside=32) --
    this returns every distinct pixel the file actually contributes to,
    with a forest count each, rather than assuming a single pixel.

    Returns (file, pixel array, per-pixel forest count array).
    """
    keys = params.delta_keys
    deltafile = fitsio.FITS(file)
    check_keys(deltafile, file)
    metadata = deltafile[keys['metadata']][:]
    theta = params.halfpi - metadata[keys['dec']]
    phi = metadata[keys['ra']]
    pix = healpy.ang2pix(params.nside, theta, phi)
    unique_pix, counts = np.unique(pix, return_counts=True)
    return file, unique_pix, counts


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

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Takes delta files by picca and stores data in data.npy.')
    parser.add_argument('--delta-dir', type=str, required=True,
        help = 'Path to the delta files.')
    parser.add_argument('--data-dir', type=str, default = params.data_dir,
        help = 'Directory where the data will be stored.')
    parser.add_argument('--split-number', type=int, default = 1,
        help = 'Number of files to split the data.')
    parser.add_argument('--statistics', action = 'store_true', required = False,
        help = 'Write the sizes and neighbors diagnostic files to the data '
               'directory. Counting neighbours needs a full neighbour search, '
               'which is a large part of the runtime.')
    args = parser.parse_args()

    if os.path.exists(args.data_dir):
        warnings.warn('The output delta directory already exists. This procedure might mix deltas from a different run.')

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    directory = glob.glob(args.delta_dir + '/*.fits.gz')
    if len(directory) == 0:
        print('No delta files in directory ' + args.delta_dir)

    # Check the configured keys against one file before starting the workers, so a
    # mismatch is reported plainly instead of through a RemoteTraceback.
    if directory:
        with fitsio.FITS(directory[0]) as first_file:
            check_keys(first_file, directory[0])

    pool = Pool()

    ####################################################################
    # Pass 1: a cheap RA/DEC-only census (no delta/weight/lambda arrays,
    # no quasar objects) over every file, so the pixel -> output-file
    # split can be decided before extracting anything expensive. Without
    # this, deciding the split needs every forest already extracted,
    # which is exactly the "whole dataset in memory at once" this is
    # for. One input file does not necessarily map to one output pixel
    # (confirmed on real data, see record_pixel_only), so this has to be
    # a real census, not just a per-file lookup.
    ####################################################################
    print('Reading pixel positions to decide how to split the data (pass 1 of 2).')
    census = pool.map(record_pixel_only, directory)

    pixel_forest_count = {}
    file_pixels = {}
    for file, unique_pix, counts in census:
        file_pixels[file] = set(int(p) for p in unique_pix)
        for pix, count in zip(unique_pix, counts):
            pix = int(pix)
            pixel_forest_count[pix] = pixel_forest_count.get(pix, 0) + int(count)
    del census

    list_of_pixels = sorted(pixel_forest_count)
    pixels_partial = np.array_split(list_of_pixels, args.split_number)
    pixel_file = {}
    for i, subset in enumerate(pixels_partial, start=1):
        for pixel in subset:
            pixel_file[int(pixel)] = i

    # Which files does each output chunk actually need? A file that
    # contributes to pixels split across more than one chunk is read
    # again for each one -- wasteful when it happens, but correct
    # either way, and the census above already shows it need not be
    # rare: whether it happens at all depends on how the pixels a file
    # touches happen to fall across --split-number's chunk boundaries.
    chunk_files = {i: set() for i in range(1, args.split_number + 1)}
    for file, pixels in file_pixels.items():
        for pix in pixels:
            chunk_files[pixel_file[pix]].add(file)

    ####################################################################
    # Pass 2: extract and save one output chunk at a time. Only that
    # chunk's own forests are ever in memory at once -- the previous
    # chunk's are already on disk and freed by the time the next one
    # starts, and later chunks' files are not read until their turn.
    ####################################################################
    max_lenght = 0
    min_distance = 1e10
    j = 0

    for chunk_number in range(1, args.split_number + 1):
        files_for_chunk = sorted(chunk_files[chunk_number])
        subdata = {}
        if files_for_chunk:
            print('Extracting chunk %d/%d from %d file(s) (pass 2 of 2).'
                  % (chunk_number, args.split_number, len(files_for_chunk)))
            data_list = pool.map(record_from_deltas, files_for_chunk)
            for list_of_forests in data_list:
                for forest_data in list_of_forests:
                    # A file can feed more than one chunk (see chunk_files
                    # above); only keep the forests this chunk actually owns.
                    if pixel_file.get(forest_data.pix) != chunk_number:
                        continue
                    j += 1
                    new_long = len(forest_data.dc)
                    if new_long > max_lenght:
                        max_lenght = new_long
                    if forest_data.dc[0] < min_distance:
                        min_distance = forest_data.dc[0]
                    subdata.setdefault(forest_data.pix, []).append(forest_data)
            del data_list
        np.save(os.path.join(args.data_dir, 'data' + str(chunk_number)), subdata)
        del subdata

    angmax = 2*np.arcsin(0.5*params.rtmax/min_distance)
    print('Minimum comoving distance to a forest (Mpc/h):',min_distance)
    print('Maximum angle between pairs of skewers that are used (rad):', angmax)

    # Needed by the multi-file drivers (two_point.py, distortion.py) to
    # split pixels across MPI ranks and load only the data*.npy files a
    # rank actually needs -- its own pixels plus a buffer of neighbouring
    # ones -- instead of the whole dataset. min_distance is saved rather
    # than the angmax derived from it, so a driver run with a different
    # rtmax later still gets the angmax that setting actually implies.
    # See IMPROVEMENTS.md #15.
    np.save(os.path.join(args.data_dir, 'data_index'),
            {'pixel_file': pixel_file, 'min_distance': min_distance})

    # The neighbour search is only needed for these diagnostics: the
    # correlation and distortion both call forest.neighborhood() and
    # recompute from scratch. It is a large part of the runtime, so it is
    # off unless asked for. Reads back the just-saved chunks, one at a
    # time plus a buffer of its neighbours (pixel_partition.py -- the
    # same mechanism the correlation drivers use), rather than needing
    # the whole dataset in memory a second time: forest.neighborhood()
    # only sees a given forest's own chunk otherwise, which is exactly
    # the missing-neighbour bug IMPROVEMENTS.md #15 is about.
    if args.statistics:
        print('Computing neighbour statistics (needs a full neighbour search).')
        known_pixels = set(pixel_file)
        sizes = []
        neighbors = []
        for chunk_number in range(1, args.split_number + 1):
            owned = np.array([p for p in list_of_pixels if pixel_file[p] == chunk_number])
            if len(owned) == 0:
                continue
            buffer_pix = pixel_partition.find_buffer_pixels(owned, angmax, known_pixels)
            chunk_data = pixel_partition.load_rank_data(args.data_dir, owned, buffer_pix, pixel_file)
            for pix in owned:
                for forest in chunk_data[pix]:
                    sizes.append(len(forest.dc))
                    forest_angmax = 2*np.arcsin(0.5*params.rtmax/forest.dc[0])
                    neigh_names, _ = forest.neighborhood_names(chunk_data, forest_angmax)
                    neighbors.append(len(neigh_names))
        np.savetxt(os.path.join(args.data_dir, "sizes"), sizes)
        np.savetxt(os.path.join(args.data_dir, "neighbors"), neighbors)

    print("The largest forest has ", max_lenght, " data points.")
    print("The number of forests is:", j)


if __name__ == '__main__':
    main()
