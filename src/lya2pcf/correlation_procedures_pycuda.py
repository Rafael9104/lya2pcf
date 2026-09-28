import numpy as np
from dataclasses import dataclass

import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray

from . import parameters as params
from . import gpu_support
from . import streaming_upload


# The kernel's precision and the dtype of the buffers we upload to it have to
# agree, so both come from the same setting.
myfloat = params.gpu_dtype
mod = gpu_support.compile_kernels()

pair_correlation = mod.get_function("pair_correlation")


@dataclass
class ForestBuffers:
    """Device buffers holding every uploaded forest, as filled by
    upload_forests(). Public so other GPU code over the same forests (a
    three-point correlation, say) can consume it without going through
    TwoPointGPU.

    forest.index into these flat, max_lenght-strided buffers is set by the
    upload, which is how a caller locates a given forest's own slice of them.
    `data` is the {pixel: [forests]} dict of the uploaded forests, with their
    per-pixel arrays already dropped (they are on the GPU now), that
    forest.neighborhood() and the kernel launches work from.
    """
    gran_dc_d: object
    gran_rx_d: object
    gran_ry_d: object
    gran_rz_d: object
    gran_we_d: object
    gran_dw_d: object
    gran_x_d: object
    gran_y_d: object
    gran_z_d: object
    numpix_d: object
    max_lenght: int
    forest_count: int
    data: object


def upload_forests(plan):
    """ Streams the forests of a rank to the GPU: the host-to-device transfer
    any GPU code over the same forests needs, independent of which correlation
    is computed afterwards.

    plan        pixel_partition.ForestPlan
                Which forests (a rank's own pixels plus their neighbour buffer), and
                which GPU slot each one goes to. The data*.npy files are read one at a
                time and copied pixel by pixel, so the forests are never all held in
                host memory (streaming_upload.py). Each forest's per-pixel arrays are
                dropped once uploaded.

    Returns a ForestBuffers with the device pointers, the max_lenght the buffers were
    sized from, and the light forest dict. Also sets forest.index and
    forest.num_points on every uploaded forest.
    """
    count_forests = plan.count_forests
    # The kernel takes this as an int argument, which pycuda can only marshal
    # from a fixed-width type, not a plain Python int.
    max_lenght = np.int32(plan.max_lenght)

    # Checked before anything is allocated, so a slice that does not fit is
    # reported as a whole rather than as whichever allocation happened to fail.
    itemsize = np.dtype(myfloat).itemsize
    forest_bytes = count_forests * int(max_lenght) * itemsize
    gpu_support.require_memory(
        6 * forest_bytes                                  # dc, rx, ry, rz, we, dw
        + 3 * count_forests * itemsize,                   # x, y, z
        "forest data (%d forests, longest %d pixels)" % (count_forests, max_lenght),
        ["split the deltas into more files with the extraction step's "
         "--split-number, and run with more MPI ranks -- each rank only "
         "loads the files its own pixels (plus their neighbour buffer) "
         "actually need, not the whole dataset",
         "coadd/rebin the deltas upstream, which shortens every forest",
         "run on more GPUs: each MPI rank takes a share of the pixels"])

    big, small, light_data = streaming_upload.stream_forests_to_gpu(
        plan, ('dc', 'rx', 'ry', 'rz', 'we', 'dw'), ('x', 'y', 'z'), myfloat)
    numpix_d = gpuarray.to_gpu(np.array([params.numpix_r, params.numpix_mu, params.numpix_theta], dtype = np.int32))

    return ForestBuffers(
        gran_dc_d=big['dc'], gran_rx_d=big['rx'], gran_ry_d=big['ry'], gran_rz_d=big['rz'],
        gran_we_d=big['we'], gran_dw_d=big['dw'], gran_x_d=small['x'], gran_y_d=small['y'], gran_z_d=small['z'],
        numpix_d=numpix_d, max_lenght=max_lenght, forest_count=count_forests, data=light_data)


class TwoPointGPU:
    """ The forests of one MPI rank on the GPU, and the pair_correlation launches over them.

    Copies the forests of `plan` to the GPU once, to avoid the overhead of
    copying them at every call (see upload_forests).
    Parameters:
    plan        pixel_partition.ForestPlan
                The rank's own pixels plus their neighbour buffer.
    shape_hist  shape of the (rp, rt) histogram
    angmax      maximum angle between two forests that are neighbours

    `data` is the {pixel: [forests]} dict two_point_per_pixel works from, with the
    per-pixel arrays already dropped; the device arrays are kept in `buffers`.
    """

    def __init__(self, plan, shape_hist, angmax):
        self.shape_hist = shape_hist
        self.angmax = angmax

        # pair_correlation's per-block histogram (w_hist and dw_hist, both
        # shape_hist-sized) lives in dynamic shared memory -- see the kernel's
        # own comment. Checked once here, at the same size for every launch,
        # since shape_hist is fixed for the run and a raw CUDA "out of shared
        # memory" error at launch time would not say why or suggest a fix.
        self.shared_bytes = 2 * int(np.prod(shape_hist)) * np.dtype(myfloat).itemsize
        max_shared = cuda.Context.get_device().get_attribute(
            cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK)
        if self.shared_bytes > max_shared:
            raise RuntimeError(
                "pair_correlation's per-block histogram needs %d bytes of shared "
                "memory (2 * %d * %d bins * %d bytes for %s), but this GPU only "
                "has %d bytes per block. Use fewer/coarser bins (numpix_rp x "
                "numpix_rt, from rmax and bin_size_r in parameters.yml) to fit."
                % (self.shared_bytes, shape_hist[0], shape_hist[1],
                   np.dtype(myfloat).itemsize, myfloat, max_shared))

        self.buffers = upload_forests(plan)
        self.data = self.buffers.data

    def two_point_per_pixel(self, pixel):
        """ This function computes the weighted sum of w and delta*w for all pairs of data
        and stores them in histograms to prepare for the correlation function. The
        histograms are stored by healpix pixel of the first element in the pair.
        Parameters:
        pixel   int
                The healpix pixel of the first element in the pair.
        """
        b = self.buffers
        shape_hist = self.shape_hist

        # Preparing data structure for the partial histograms
        w_hist_d = gpuarray.zeros(shape_hist, dtype = myfloat)
        dw_hist_d = gpuarray.zeros(shape_hist, dtype = myfloat)

        # The per-launch scalars go to the kernel by value (numpy scalars: pycuda
        # picks the C type from the numpy type), not through small device arrays.
        numpix_rp, numpix_rt = np.int32(shape_hist[0]), np.int32(shape_hist[1])
        rpmax, rtmax = myfloat(params.rpmax), myfloat(params.rtmax)
        # y and z genuinely parallelize the kernel's two strided loops (over
        # pixels within each neighbour, and over neighbours -- y is the
        # coalesced one, see the kernel's own comment), so they're the same
        # shared 2D block used for order_active. x must stay 1: the kernel
        # reads its pixel-in-forest1 index from blockIdx.x, not threadIdx.x,
        # so blockDim.x > 1 would run the same accumulation redundantly and
        # double-count into the histogram.
        threads_per_block = (1,) + params.threads_per_block_2d

        for forest1 in self.data[pixel]:

            # Looking for neighbors
            neighbors = forest1.neighborhood(self.data, self.angmax)
            if len(neighbors) == 0:
                # This forest have zero neighbors
                continue
            forest1_lenght = forest1.num_points
            neigh_index = np.array([forest2.index for forest2 in neighbors],dtype=np.int32)
            neigh_sizes = np.array([forest2.num_points for forest2 in neighbors], dtype = np.int32)
            neigh_index_d = gpuarray.to_gpu(neigh_index)
            neigh_sizes_d = gpuarray.to_gpu(neigh_sizes)

            # Be careful, this can not change unless the kernel procedure change.
            blocks_per_grid = (forest1_lenght, 1, 1)

            pair_correlation(np.int32(forest1.index), np.int32(forest1_lenght), np.int32(len(neighbors)),
                    neigh_index_d, neigh_sizes_d,
                    numpix_rp, numpix_rt, np.int32(b.max_lenght),
                    rpmax, rtmax, w_hist_d, dw_hist_d,
                b.gran_dc_d, b.gran_rx_d, b.gran_ry_d, b.gran_rz_d, b.gran_we_d, b.gran_dw_d,
                b.gran_x_d, b.gran_y_d, b.gran_z_d,
                block = threads_per_block, grid = blocks_per_grid, shared = self.shared_bytes)

            # This is necessary to avoid to overwrite x12, y12, z12, bin_r12 with the next forest
            pycuda.autoinit.context.synchronize()

        return (w_hist_d.get(), dw_hist_d.get())
