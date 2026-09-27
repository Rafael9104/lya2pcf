"""
    Copies a rank's forests to the GPU one pixel at a time, without ever
    holding all of them in host memory.

    The GPU upload used to load every forest of the rank (owned pixels plus
    neighbour buffer), pack them into flat host arrays as large as the GPU
    buffers, and only then upload. On a full DR1 slice that peaked at ~14-15 GB
    of host RAM per rank (measured: ~55-58 KB per forest), which does not fit
    when several ranks share a node.

    Here the layout is planned first (pixel_partition.plan_rank_data): every
    forest's GPU slot and max_lenght are known from the index alone. The GPU
    buffers are allocated at their final size and filled pixel by pixel while
    the data*.npy files are read one at a time. Once a forest is on the GPU
    its large arrays are dropped and only what the host loops still need is
    kept: x, y, z, ra, name (forest.neighborhood), index and num_points.
    Peak host RAM is then about one data file plus that light metadata,
    independent of how big the rank's slice is.
"""

import numpy as np
import pycuda.driver as cuda

from . import pixel_partition

# The per-pixel arrays of a forest. Dropped from the host copy once uploaded.
HEAVY_FIELDS = ('dw', 'we', 'dc', 'rx', 'ry', 'rz', 'delta_lambda')


def stream_forests_to_gpu(plan, big_fields, small_fields, dtype):
    """Uploads the forests of `plan` and returns (big, small, data).

    big_fields    forest attributes that are arrays with one value per pixel,
                  stored on the GPU in slots of plan.max_lenght elements
                  (zero padded), like the flat gran_* buffers.
    small_fields  forest attributes that are one number per forest, stored
                  in an array indexed by the forest's slot.
    dtype         precision of the GPU buffers (params.gpu_dtype).

    big and small map each field name to its device buffer. data is the
    {pixel: [forests]} dict with the heavy arrays removed and forest.index
    (the slot) and forest.num_points (number of pixels) set, ready for
    forest.neighborhood() and the kernel launches.
    """
    itemsize = np.dtype(dtype).itemsize
    max_lenght = int(plan.max_lenght)
    count = plan.count_forests
    slot_bytes = max_lenght * itemsize

    big = {}
    for name in big_fields:
        # Not zeroed: every slot is overwritten below. plan_rank_data gives the
        # pixels consecutive, non-overlapping slot ranges that cover the whole
        # buffer, each pixel's block is written in full (its padding is already
        # zero in the block), and iter_plan_files raises if a pixel does not have
        # the number of forests the plan counted.
        big[name] = cuda.mem_alloc(count * slot_bytes)
    small_host = {name: np.zeros(count, dtype=dtype) for name in small_fields}

    data = {}
    for pixel_forests in pixel_partition.iter_plan_files(plan):
        for pixel, forests in pixel_forests.items():
            first_slot = plan.slot_base[pixel]
            block = {name: np.zeros((len(forests), max_lenght), dtype=dtype) for name in big_fields}
            for k, forest in enumerate(forests):
                length = len(forest.we)
                for name in big_fields:
                    block[name][k, :length] = getattr(forest, name)
                for name in small_fields:
                    small_host[name][first_slot + k] = getattr(forest, name)
                forest.index = first_slot + k
                forest.num_points = length
                for name in HEAVY_FIELDS:
                    setattr(forest, name, None)
            for name in big_fields:
                cuda.memcpy_htod(int(big[name]) + first_slot * slot_bytes, block[name])
            data[pixel] = forests

    small = {}
    for name in small_fields:
        small[name] = cuda.mem_alloc(small_host[name].nbytes)
        cuda.memcpy_htod(small[name], small_host[name])
    return big, small, data
