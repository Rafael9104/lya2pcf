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

# Must match NUM_HISTOGRAMS in cuda_kernels.cpp: w, delta*w, w*z, w*rp, w*rt.
NUM_HISTOGRAMS = 5


def plan_shared_histograms(nbins):
    """ Decides how many of pair_correlation's histograms go in shared memory.
    Returns (n_shared, shared_bytes): the number of histograms (0 to NUM_HISTOGRAMS)
    that the block accumulates in shared memory and the bytes to request for it.

    A block gets 48 KB of shared memory by default; some GPUs (Volta and newer)
    allow more if the kernel asks for it, up to their MAX_SHARED_MEMORY_PER_BLOCK_OPTIN.
    Whatever the GPU allows is used, unless parameters.yml's shared_histograms says
    otherwise. What is not in shared memory is added directly to global memory.
    """
    device = cuda.Context.get_device()
    attributes = cuda.device_attribute
    limit = device.get_attribute(attributes.MAX_SHARED_MEMORY_PER_BLOCK)
    try:
        limit = max(limit, device.get_attribute(attributes.MAX_SHARED_MEMORY_PER_BLOCK_OPTIN))
    except AttributeError:
        pass
    limit -= pair_correlation.shared_size_bytes
    histogram_bytes = nbins * np.dtype(myfloat).itemsize
    n_shared = min(NUM_HISTOGRAMS, limit // histogram_bytes)
    if params.shared_histograms is not None:
        if not 0 <= params.shared_histograms <= NUM_HISTOGRAMS:
            raise ValueError("shared_histograms must be between 0 and %d, got %r"
                             % (NUM_HISTOGRAMS, params.shared_histograms))
        if params.shared_histograms > n_shared:
            raise RuntimeError(
                "shared_histograms: %d needs %d bytes of shared memory per block, but this GPU allows %d. "
                "Use at most %d, fewer/coarser bins, or leave it unset."
                % (params.shared_histograms, params.shared_histograms * histogram_bytes,
                   limit, n_shared))
        n_shared = params.shared_histograms
    shared_bytes = int(n_shared * histogram_bytes)
    # Above the default 48 KB a kernel has to ask for it.
    if shared_bytes > 48 * 1024:
        pair_correlation.set_attribute(cuda.function_attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_bytes)
    return int(n_shared), shared_bytes


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
    gran_redshift_d: object
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
        7 * forest_bytes                                  # dc, rx, ry, rz, we, dw, redshift
        + 3 * count_forests * itemsize,                   # x, y, z
        "forest data (%d forests, longest %d pixels)" % (count_forests, max_lenght),
        ["split the deltas into more files with the extraction step's "
         "--split-number, and run with more MPI ranks -- each rank only "
         "loads the files its own pixels (plus their neighbour buffer) "
         "actually need, not the whole dataset",
         "coadd/rebin the deltas upstream, which shortens every forest",
         "run on more GPUs: each MPI rank takes a share of the pixels"])

    # 'redshift' rides along with the other per-pixel fields the streaming upload
    # already handles; the z*w histogram needs each pixel's redshift, set at
    # extraction time (delta_reader.py) since forest.z is the unit-vector
    # component, not a redshift.
    big, small, light_data = streaming_upload.stream_forests_to_gpu(
        plan, ('dc', 'rx', 'ry', 'rz', 'we', 'dw', 'redshift'), ('x', 'y', 'z'), myfloat)
    numpix_d = gpuarray.to_gpu(np.array([params.numpix_r, params.numpix_mu, params.numpix_theta], dtype = np.int32))

    return ForestBuffers(
        gran_dc_d=big['dc'], gran_rx_d=big['rx'], gran_ry_d=big['ry'], gran_rz_d=big['rz'],
        gran_we_d=big['we'], gran_dw_d=big['dw'], gran_x_d=small['x'], gran_y_d=small['y'], gran_z_d=small['z'],
        gran_redshift_d=big['redshift'], numpix_d=numpix_d, max_lenght=max_lenght, forest_count=count_forests,
        data=light_data)


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
    `n_shared`/`shared_bytes` (see plan_shared_histograms) say how many of the 5
    histograms this run keeps in shared memory -- worth logging after construction.
    """

    def __init__(self, plan, shape_hist, angmax):
        self.shape_hist = shape_hist
        self.angmax = angmax

        self.buffers = upload_forests(plan)
        self.data = self.buffers.data

        self.n_shared, self.shared_bytes = plan_shared_histograms(int(np.prod(shape_hist)))

    def two_point_per_pixel(self, pixel):
        """ This function computes the weighted sum of w and delta*w for all pairs of data
        and stores them in histograms to prepare for the correlation function. Three more
        histograms hold the sums of w*z, w*rp and w*rt (z is the mean redshift of the pair),
        which post-processing divides by the w histogram to get the weighted average of each
        in every bin. The histograms are stored by healpix pixel of the first element in the pair.
        Parameters:
        pixel   int
                The healpix pixel of the first element in the pair.

        Returns an array of shape (5,) + shape_hist with the histograms w, delta*w, w*z,
        w*rp and w*rt, in that order.
        """
        b = self.buffers
        shape_hist = self.shape_hist

        # Preparing data structure for the partial histograms: the five, one after the
        # other, in a single buffer (see pair_correlation).
        hist = np.zeros((NUM_HISTOGRAMS,) + tuple(shape_hist), dtype = myfloat)
        hist_d = cuda.mem_alloc(hist.nbytes)
        # It is necessary to initiallize the histograms at zero, otherwise resicual noise
        # from the memory can get in the computation
        cuda.memcpy_htod(hist_d, hist)

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
                    numpix_rp, numpix_rt, np.int32(b.max_lenght), np.int32(self.n_shared),
                    rpmax, rtmax, hist_d,
                b.gran_dc_d, b.gran_we_d, b.gran_dw_d,
                b.gran_x_d, b.gran_y_d, b.gran_z_d,
                b.gran_redshift_d,
                block = threads_per_block, grid = blocks_per_grid, shared = self.shared_bytes)

            # Wait for this forest's launch before starting the next one.
            pycuda.autoinit.context.synchronize()

        cuda.memcpy_dtoh(hist, hist_d)

        return hist
