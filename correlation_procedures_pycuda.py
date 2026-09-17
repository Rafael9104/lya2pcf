import numpy as np
import time
import parameters as params
import gpu_support

import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule


# The kernel's precision and the dtype of the buffers we upload to it have to
# agree, so both come from the same setting.
myfloat = params.gpu_dtype
gpu_support.check_precision_supported()
with open('cuda_kernels.cpp') as f:
  mod = SourceModule(f.read(), options=['-DMYFLOAT=' + params.gpu_ctype])

pair_correlation = mod.get_function("pair_correlation")

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
    global numpix_d
    global max_lenght

    data = data_aux
    log_file = log_file_aux
    shape_hist = shape_hist_aux
    angmax = angmax_aux

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
        6 * forest_bytes                                  # dc, rx, ry, rz, we, dw
        + 3 * count_forests * itemsize                    # x, y, z
        + 2 * int(np.prod(shape_hist)) * itemsize,        # w_hist, dw_hist
        "forest data (%d forests, longest %d pixels)" % (count_forests, max_lenght),
        ["split the deltas into more files with delta_reader.py "
         "--split-number and run 2pla_multiple_data.py, which uploads one "
         "file at a time",
         "coadd/rebin the deltas upstream, which shortens every forest",
         "run on more GPUs: each MPI rank takes a share of the pixels"])

    gran_dc = np.zeros((count_forests * max_lenght), dtype = myfloat)
    gran_rx = np.zeros((count_forests * max_lenght), dtype = myfloat)
    gran_ry = np.zeros((count_forests * max_lenght), dtype = myfloat)
    gran_rz = np.zeros((count_forests * max_lenght), dtype = myfloat)
    gran_we = np.zeros((count_forests * max_lenght), dtype = myfloat)
    gran_dw = np.zeros((count_forests * max_lenght), dtype = myfloat)
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
    gran_x_d = cuda.mem_alloc(lenght_data_small)
    gran_y_d = cuda.mem_alloc(lenght_data_small)
    gran_z_d = cuda.mem_alloc(lenght_data_small)

    cuda.memcpy_htod(gran_dc_d, gran_dc)
    cuda.memcpy_htod(gran_rx_d, gran_rx)
    cuda.memcpy_htod(gran_ry_d, gran_ry)
    cuda.memcpy_htod(gran_rz_d, gran_rz)
    cuda.memcpy_htod(gran_we_d, gran_we)
    cuda.memcpy_htod(gran_dw_d, gran_dw)
    cuda.memcpy_htod(gran_x_d, gran_x)
    cuda.memcpy_htod(gran_y_d, gran_y)
    cuda.memcpy_htod(gran_z_d, gran_z)

    numpix_d = gpuarray.to_gpu(np.array([params.numpix_r, params.numpix_mu, params.numpix_theta], dtype = np.int32))


def two_point_per_pixel(pixel, **kargs):
    """ This function computes the weighted sum of w and delta*w for all pairs of data
    and stores them in histograms to prepare for the correlation function. The
    histograms are stored by healpix pixel of the first element in the pair.
    Parammeters:
    pixel   int
            The healpix pixel of the first element in the pair.
    angmax real
            Maximum angle between to forests to fit in the histogram.
    shape_hist  array int (np, nt)
            Shape of the histogram in bits
    """
    # Preparing data structure for the partial histograms
    w_hist = np.zeros(shape_hist, dtype = myfloat)
    dw_hist = np.zeros(shape_hist, dtype = myfloat)
    numpix2d_d = gpuarray.to_gpu(np.array(shape_hist, dtype = np.int32))

    w_hist_d = cuda.mem_alloc(w_hist.nbytes)
    dw_hist_d = cuda.mem_alloc(w_hist.nbytes)
    cuda.memcpy_htod(w_hist_d, w_hist)
    cuda.memcpy_htod(dw_hist_d, dw_hist)

    # Passing data to the GPU
    rmax_d = gpuarray.to_gpu(np.array([params.rpmax,params.rtmax],dtype=myfloat))
        # Be careful, this can not change unless the kernel procedure change.
    threads_per_block = (1, 16, 16)

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
                numpix2d_d, max_lenght,
            rmax_d, w_hist_d, dw_hist_d, 
            gran_dc_d, gran_rx_d, gran_ry_d, gran_rz_d, gran_we_d, gran_dw_d, gran_x_d, gran_y_d, gran_z_d,
            block = threads_per_block, grid = blocks_per_grid)
        
        # This is necessary to avoid to overwrite x12, y12, z12, bin_r12 with the next forest
        pycuda.autoinit.context.synchronize()

    cuda.memcpy_dtoh(w_hist, w_hist_d)
    cuda.memcpy_dtoh(dw_hist, dw_hist_d)

    return (w_hist, dw_hist)


