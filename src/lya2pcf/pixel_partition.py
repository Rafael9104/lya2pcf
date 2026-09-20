"""
    Splits pixels across MPI ranks and figures out which data*.npy files
    (and which of their pixels) a rank actually needs -- its own assigned
    pixels, plus a buffer of neighbouring pixels so forest.neighborhood()
    can see every real neighbour even when it lives in a file a different
    rank owns.

    This is the fix for the missing-pair bug in IMPROVEMENTS.md #15:
    delta_reader.py splits files by contiguous ranges of a sorted pixel
    list, so pairs whose two forests land on opposite sides of a file
    boundary were invisible to each other. Loading a buffer of
    neighbouring pixels from other files removes that boundary for
    correctness purposes, without requiring every rank to load the whole
    dataset.

    Buffer pixels are never iterated as the first forest of a pair (see
    each driver's own pixel loop, which only ever loops over its
    *owned* pixels) -- only used as forest.neighborhood()'s candidate
    pool -- so the ordering rule in forest_class.py
    (`self.ra < forest2.ra`) still counts every pair exactly once with no
    de-duplication pass needed. See IMPROVEMENTS.md #15 for the full
    argument.
"""
import os

import healpy
import numpy as np

from . import parameters as params


def load_index(data_dir):
    """The data_index.npy written by delta_reader.py / delta_reader_eboss.py:
    {'pixel_file': {pixel: file_number}, 'min_distance': float}.
    """
    path = os.path.join(data_dir, 'data_index.npy')
    try:
        return np.load(path, allow_pickle=True).item()
    except FileNotFoundError:
        raise FileNotFoundError(
            "%s not found. It is written by lya2pcf-extract[-eboss] "
            "alongside data*.npy and is needed to split pixels across "
            "MPI ranks and load only the files a rank actually needs. "
            "Re-run the extraction step if this data predates it."
            % path) from None


def assign_pixels(pixel_file, mpi_size, mpi_rank):
    """This rank's share of the sorted global pixel list.

    Same contiguous-chunk convention delta_reader.py uses to split pixels
    into files (np.array_split on the sorted pixel list), so a rank's
    assigned pixels usually fall in only a few files, not the whole
    dataset -- though see IMPROVEMENTS.md #15 for why that is not
    guaranteed (RING-ordered healpix indices are not spatially local).
    """
    all_pixels = np.array(sorted(pixel_file))
    return np.array_split(all_pixels, mpi_size)[mpi_rank]


def rank_chunks(num_chunks, mpi_rank, mpi_size):
    """The chunk numbers this rank processes, one after the other.

    Chunk c is the c-th of num_chunks contiguous slices of the sorted pixel
    list (assign_pixels(pixel_file, num_chunks, c)). Rank r takes chunks
    r, r + mpi_size, r + 2*mpi_size, ... so any combination works:
    num_chunks == mpi_size is the original one-slice-per-rank behaviour,
    mpi_size == 1 runs every chunk sequentially in one process (one GPU),
    and anything in between shares the chunks out round-robin.
    """
    return range(mpi_rank, num_chunks, mpi_size)


def find_buffer_pixels(owned_pixels, angmax, known_pixels):
    """Healpix pixels within angmax of any owned pixel, excluding the
    owned pixels themselves and any pixel with no data at all.

    Evaluated from each owned pixel's *center*, with nside's own maximum
    pixel radius added as a margin -- a forest can sit anywhere within
    its pixel, not just at the center, so the search has to be at least
    that much wider than angmax to guarantee no real neighbour is missed.
    """
    if len(owned_pixels) == 0:
        return set()
    owned = set(int(p) for p in owned_pixels)
    margin = healpy.max_pixrad(params.nside)
    search_radius = angmax + margin
    buffer = set()
    for pixel in owned_pixels:
        theta, phi = healpy.pix2ang(params.nside, int(pixel))
        vec = healpy.ang2vec(theta, phi)
        for candidate in healpy.query_disc(params.nside, vec, search_radius, inclusive=True):
            candidate = int(candidate)
            if candidate not in owned and candidate in known_pixels:
                buffer.add(candidate)
    return buffer


def load_rank_data(data_dir, owned_pixels, buffer_pixels, pixel_file):
    """Loads exactly the data*.npy files owned_pixels/buffer_pixels need,
    merged into one dict keyed by pixel -- owned and buffer pixels alike,
    since forest.neighborhood() just needs them all visible. Which
    pixels are "owned" (safe to use as the first forest of a pair) is
    the caller's own owned_pixels list; nothing here distinguishes them.
    """
    needed_pixels = set(int(p) for p in owned_pixels) | set(buffer_pixels)
    needed_files = sorted(set(pixel_file[p] for p in needed_pixels))

    data = {}
    for file_number in needed_files:
        file_path = os.path.join(data_dir, 'data%d.npy' % file_number)
        file_data = np.load(file_path, allow_pickle=True).item()
        for pixel, forests in file_data.items():
            if pixel in needed_pixels:
                data[pixel] = forests
    return data
