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
import argparse
import os
from dataclasses import dataclass

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


@dataclass
class ForestPlan:
    """Where every forest of a rank goes on the GPU, worked out from the
    index alone, before any data*.npy file is read.

    The rank's forests (its own pixels plus the neighbour buffer) are laid
    out pixel by pixel in increasing pixel order: pixel p's forests take
    slots slot_base[p], slot_base[p] + 1, ... Knowing that and max_lenght up
    front is what lets the GPU buffers be allocated at their final size and
    filled one pixel at a time (streaming_upload.py), instead of first
    holding every forest of the rank in host memory.
    """
    data_dir: str
    pixels: list          # sorted pixels needed (owned + buffer)
    slot_base: dict       # pixel -> first slot of its forests
    pixel_count: dict     # pixel -> number of forests in it
    files: dict           # file number -> pixels needed from it
    count_forests: int
    max_lenght: int


def index_stats(index, data_dir=None):
    """pixel_count and max_lenght from the index, or an error saying how to add them."""
    if 'pixel_count' not in index or 'max_lenght' not in index:
        raise KeyError(
            "data_index.npy%s has no 'pixel_count'/'max_lenght'. New extractions "
            "write them; for data extracted before that, run: "
            "python -m lya2pcf.pixel_partition --data-dir %s"
            % (' in ' + data_dir if data_dir else '', data_dir or 'DATA_DIR'))
    return index['pixel_count'], int(index['max_lenght'])


def plan_rank_data(data_dir, index, owned_pixels, buffer_pixels):
    """The ForestPlan for a rank owning owned_pixels, with buffer_pixels around it."""
    pixel_count, max_lenght = index_stats(index, data_dir)
    pixel_file = index['pixel_file']
    pixels = sorted(set(int(p) for p in owned_pixels) | set(int(p) for p in buffer_pixels))
    slot_base = {}
    total = 0
    files = {}
    for pixel in pixels:
        slot_base[pixel] = total
        total += pixel_count[pixel]
        files.setdefault(pixel_file[pixel], []).append(pixel)
    return ForestPlan(data_dir=data_dir, pixels=pixels, slot_base=slot_base,
                      pixel_count={p: pixel_count[p] for p in pixels}, files=files,
                      count_forests=total, max_lenght=max_lenght)


def iter_plan_files(plan):
    """Yields {pixel: [forests]} for the plan's pixels, one data*.npy file at a
    time. Each file is loaded, the plan's pixels taken from it, and the rest
    dropped when the caller moves to the next one, so at most one file's worth
    of forests is in host memory at any moment.
    """
    for file_number in sorted(plan.files):
        file_path = os.path.join(plan.data_dir, 'data%d.npy' % file_number)
        file_data = np.load(file_path, allow_pickle=True).item()
        taken = {}
        for pixel in plan.files[file_number]:
            forests = file_data[pixel]
            if len(forests) != plan.pixel_count[pixel]:
                raise ValueError(
                    "data_index.npy says pixel %d has %d forests but %s has %d. "
                    "Re-run: python -m lya2pcf.pixel_partition --data-dir %s"
                    % (pixel, plan.pixel_count[pixel], file_path, len(forests), plan.data_dir))
            taken[pixel] = forests
        del file_data
        yield taken


def add_index_stats(data_dir):
    """Adds pixel_count and max_lenght to an existing data_index.npy by reading
    every data*.npy once (one file in memory at a time). For data extracted
    before the extraction wrote them.
    """
    index = load_index(data_dir)
    pixel_count = {}
    max_lenght = 0
    for file_number in sorted(set(index['pixel_file'].values())):
        file_data = np.load(os.path.join(data_dir, 'data%d.npy' % file_number),
                            allow_pickle=True).item()
        for pixel, forests in file_data.items():
            pixel_count[int(pixel)] = len(forests)
            for forest in forests:
                max_lenght = max(max_lenght, len(forest.dc))
        del file_data
    index['pixel_count'] = pixel_count
    index['max_lenght'] = int(max_lenght)
    np.save(os.path.join(data_dir, 'data_index'), index)
    print('Wrote pixel_count (%d pixels, %d forests) and max_lenght=%d to %s'
          % (len(pixel_count), sum(pixel_count.values()), max_lenght,
             os.path.join(data_dir, 'data_index.npy')))


def main():
    parser = argparse.ArgumentParser(description='Adds pixel_count and max_lenght to an existing '
        'data_index.npy, needed by the streaming GPU upload.')
    parser.add_argument('--data-dir', type=str, default=params.data_dir,
        help='Directory with the data*.npy files and data_index.npy.')
    add_index_stats(parser.parse_args().data_dir)


if __name__ == '__main__':
    main()
