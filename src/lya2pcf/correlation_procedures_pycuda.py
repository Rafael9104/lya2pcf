import numpy as np
import time
from dataclasses import dataclass

import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray

from . import parameters as params
from . import gpu_support


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
    """Device buffers holding every uploaded forest, as packed by
    upload_forests(). Public so other GPU code over the same forests (a
    three-point correlation, say) can consume it without going through
    two_point_per_pixel's module-level globals.

    forest.index into these flat, max_lenght-strided buffers is set as a
    side effect of the upload, which is how a caller locates a given
    forest's own slice of them.
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


def upload_forests(data, pixel_list = None):
    """ Packs forests into the flat gran_* host arrays and uploads them to
    the GPU. This is the host-to-device transfer any GPU code over the same
    forests needs, independent of which correlation is computed afterwards.
    Parammeters:
    data        dict
                Dictionary of healpix pixels to list of forests

    pixel_list  list
                Optional list with the pixels to be uploaded. If not present, the entire
                dataset is uploaded, careful should be taken when changing this option the gpu needs more data than
                the pixels that it is computing, it also needs the neigboring pixels.

    Returns a ForestBuffers with the device pointers and the max_lenght the
    buffers were sized from. Also sets forest.index on every uploaded forest.
    """
    if not pixel_list:
        pixel_list = list(data.keys())

    # max_lenght is a property of whichever data is actually loaded, not a
    # static parameter, so it's computed here rather than read from parameters.
    count_forests = 0
    max_lenght = 0
    for pixel_aux in pixel_list:
        for forest in data[pixel_aux]:
            count_forests += 1
            forest_lenght = len(forest.we)
            if forest_lenght > max_lenght:
                max_lenght = forest_lenght
    # The kernel takes this as an int argument, which pycuda can only marshal
    # from a fixed-width type, not a plain Python int.
    max_lenght = np.int32(max_lenght)

    # Checked before the host arrays are built, not just before the uploads:
    # these buffers are the same size on both sides, so on a large dataset
    # allocating and filling them first would exhaust host memory before the
    # GPU was ever asked for anything.
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

    gran_dc = np.zeros((count_forests * max_lenght), dtype = myfloat)
    gran_rx = np.zeros((count_forests * max_lenght), dtype = myfloat)
    gran_ry = np.zeros((count_forests * max_lenght), dtype = myfloat)
    gran_rz = np.zeros((count_forests * max_lenght), dtype = myfloat)
    gran_we = np.zeros((count_forests * max_lenght), dtype = myfloat)
    gran_dw = np.zeros((count_forests * max_lenght), dtype = myfloat)
    gran_redshift = np.zeros((count_forests * max_lenght), dtype = myfloat)
    gran_x = np.zeros(count_forests, dtype = myfloat)
    gran_y = np.zeros(count_forests, dtype = myfloat)
    gran_z = np.zeros(count_forests, dtype = myfloat)

    count = int(0)
    for pixel in pixel_list:
        for forest in data[pixel]:
            len_forest = len(forest.we)
            gran_dc[count * max_lenght : count * max_lenght + len_forest] = forest.dc
            gran_rx[count * max_lenght : count * max_lenght + len_forest] = forest.rx
            gran_ry[count * max_lenght : count * max_lenght + len_forest] = forest.ry
            gran_rz[count * max_lenght : count * max_lenght + len_forest] = forest.rz
            gran_we[count * max_lenght : count * max_lenght + len_forest] = forest.we
            gran_dw[count * max_lenght : count * max_lenght + len_forest] = forest.dw
            gran_redshift[count * max_lenght : count * max_lenght + len_forest] = forest.redshift
            gran_x[count] = forest.x
            gran_y[count] = forest.y
            gran_z[count] = forest.z
            forest.index = count
            count += 1

    lenght_data = gran_dw.nbytes
    lenght_data_small = gran_x.nbytes
    gran_dc_d = cuda.mem_alloc(lenght_data)
    gran_rx_d = cuda.mem_alloc(lenght_data)
    gran_ry_d = cuda.mem_alloc(lenght_data)
    gran_rz_d = cuda.mem_alloc(lenght_data)
    gran_we_d = cuda.mem_alloc(lenght_data)
    gran_dw_d = cuda.mem_alloc(lenght_data)
    gran_redshift_d = cuda.mem_alloc(lenght_data)
    gran_x_d = cuda.mem_alloc(lenght_data_small)
    gran_y_d = cuda.mem_alloc(lenght_data_small)
    gran_z_d = cuda.mem_alloc(lenght_data_small)

    cuda.memcpy_htod(gran_dc_d, gran_dc)
    cuda.memcpy_htod(gran_rx_d, gran_rx)
    cuda.memcpy_htod(gran_ry_d, gran_ry)
    cuda.memcpy_htod(gran_rz_d, gran_rz)
    cuda.memcpy_htod(gran_we_d, gran_we)
    cuda.memcpy_htod(gran_dw_d, gran_dw)
    cuda.memcpy_htod(gran_redshift_d, gran_redshift)
    cuda.memcpy_htod(gran_x_d, gran_x)
    cuda.memcpy_htod(gran_y_d, gran_y)
    cuda.memcpy_htod(gran_z_d, gran_z)

    numpix_d = gpuarray.to_gpu(np.array([params.numpix_r, params.numpix_mu, params.numpix_theta], dtype = np.int32))

    return ForestBuffers(
        gran_dc_d=gran_dc_d, gran_rx_d=gran_rx_d, gran_ry_d=gran_ry_d, gran_rz_d=gran_rz_d,
        gran_we_d=gran_we_d, gran_dw_d=gran_dw_d, gran_x_d=gran_x_d, gran_y_d=gran_y_d, gran_z_d=gran_z_d,
        gran_redshift_d=gran_redshift_d, numpix_d=numpix_d, max_lenght=max_lenght, forest_count=count_forests)


def init(data_aux, log_file_aux, shape_hist_aux, angmax_aux, pixel_list = None):
    """ This function copies all the data from the forests to the GPU to reduce the overhead
    of copying it at every call. Might need to be more selective with larger datasets.
    Parammeters:
    data        dict
                Dictionary of healpix pixels to list of forests

    pixel_list  list
                Optional list with the pixels to be uploaded. If not present, the entire
                dataset is uploaded, careful should be taken when changing this option the gpu needs more data than
                the pixels that it is computing, it also needs the neigboring pixels.

    Returns a dictionary from names of forests to positions in the forest array
    """
    global data
    global log_file
    global shape_hist
    global angmax

    global gran_dc_d
    global gran_rx_d
    global gran_ry_d
    global gran_rz_d
    global gran_we_d
    global gran_dw_d
    global gran_x_d
    global gran_y_d
    global gran_z_d
    global gran_redshift_d
    global numpix_d
    global max_lenght
    global n_shared
    global shared_bytes

    data = data_aux
    log_file = log_file_aux
    shape_hist = shape_hist_aux
    angmax = angmax_aux

    buffers = upload_forests(data, pixel_list)
    gran_dc_d = buffers.gran_dc_d
    gran_rx_d = buffers.gran_rx_d
    gran_ry_d = buffers.gran_ry_d
    gran_rz_d = buffers.gran_rz_d
    gran_we_d = buffers.gran_we_d
    gran_dw_d = buffers.gran_dw_d
    gran_x_d = buffers.gran_x_d
    gran_y_d = buffers.gran_y_d
    gran_z_d = buffers.gran_z_d
    gran_redshift_d = buffers.gran_redshift_d
    numpix_d = buffers.numpix_d
    max_lenght = buffers.max_lenght

    n_shared, shared_bytes = plan_shared_histograms(int(np.prod(shape_hist)))
    log_file.write('\n%d of the %d histograms are accumulated in shared memory (%d bytes per block).'
                   % (n_shared, NUM_HISTOGRAMS, shared_bytes))


def two_point_per_pixel(pixel, **kargs):
    """ This function computes the weighted sum of w and delta*w for all pairs of data
    and stores them in histograms to prepare for the correlation function. Three more
    histograms hold the sums of w*z, w*rp and w*rt (z is the mean redshift of the pair),
    which post-processing divides by the w histogram to get the weighted average of each
    in every bin. The histograms are stored by healpix pixel of the first element in the pair.
    Parammeters:
    pixel   int
            The healpix pixel of the first element in the pair.
    angmax real
            Maximum angle between to forests to fit in the histogram.
    shape_hist  array int (np, nt)
            Shape of the histogram in bits

    Returns an array of shape (5,) + shape_hist with the histograms w, delta*w, w*z,
    w*rp and w*rt, in that order.
    """
    # Preparing data structure for the partial histograms: the five, one after the
    # other, in a single buffer (see pair_correlation).
    hist = np.zeros((NUM_HISTOGRAMS,) + tuple(shape_hist), dtype = myfloat)
    numpix2d_d = gpuarray.to_gpu(np.array(shape_hist, dtype = np.int32))

    hist_d = cuda.mem_alloc(hist.nbytes)
    # It is necessary to initiallize the histograms at zero, otherwise resicual noise
    # from the memory can get in the computation
    cuda.memcpy_htod(hist_d, hist)

    # Passing data to the GPU
    rmax_d = gpuarray.to_gpu(np.array([params.rpmax,params.rtmax],dtype=myfloat))
    # y and z genuinely parallelize the kernel's two strided loops (over
    # neighbours and over pixels within each neighbour), so they're the
    # same shared 2D block used for order_active. x must stay 1: the
    # kernel reads its pixel-in-forest1 index from blockIdx.x, not
    # threadIdx.x, so blockDim.x > 1 would run the same accumulation
    # redundantly and double-count into the histogram.
    threads_per_block = (1,) + params.threads_per_block_2d

    for forest1 in data[pixel]:

        # Looking for neighbors
        neighbors = forest1.neighborhood(data, angmax)
        if len(neighbors) == 0:
            # This forest have zero neighbors
            continue
        forest1_lenght = len(forest1.dc)
        base = np.array([forest1.index, forest1_lenght, len(neighbors)],dtype=np.int32)
        neigh_index = np.array([forest2.index for forest2 in neighbors],dtype=np.int32)
        neigh_sizes = np.array([len(forest2.dc) for forest2 in neighbors], dtype = np.int32)
        base_d = gpuarray.to_gpu(base)
        neigh_index_d = gpuarray.to_gpu(neigh_index)
        neigh_sizes_d = gpuarray.to_gpu(neigh_sizes)

        # Be careful, this can not change unless the kernel procedure change.
        blocks_per_grid = (forest1_lenght, 1, 1)

        pair_correlation(base_d, neigh_index_d, neigh_sizes_d,
                numpix2d_d, max_lenght, np.int32(n_shared),
            rmax_d, hist_d,
            gran_dc_d, gran_rx_d, gran_ry_d, gran_rz_d, gran_we_d, gran_dw_d, gran_x_d, gran_y_d, gran_z_d,
            gran_redshift_d,
            block = threads_per_block, grid = blocks_per_grid, shared = shared_bytes)
        
        # This is necessary to avoid to overwrite x12, y12, z12, bin_r12 with the next forest
        pycuda.autoinit.context.synchronize()

    cuda.memcpy_dtoh(hist, hist_d)

    return hist


