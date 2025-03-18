import numpy as np
import random
import time
from parameters import *

import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

random.seed(1)
with open('cuda_kernels.cpp') as f:
  mod = SourceModule(f.read())
precompute_distances = mod.get_function("precompute_distances")
compute_etas = mod.get_function("compute_etas")
compute_d = mod.get_function("compute_d")
order_active = mod.get_function("order_active")

def init(data_aux, log_file_aux, shape_hist_aux, angmax_aux, reject_aux, pixel_list = None):
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
    global reject_fraction

    global gran_rx_d
    global gran_ry_d
    global gran_rz_d
    global gran_we_d
    global gran_dw_d
    global gran_x_d
    global gran_y_d
    global gran_z_d
    global numpix_d
    global gran_dc_d
    global gran_dl_d
    global gran_odl2_d
    global gran_omega_d
    global weight_B
    global weight_B_d
    global total_bins

    data = data_aux
    log_file = log_file_aux
    shape_hist = shape_hist_aux
    angmax = angmax_aux
    reject_fraction = reject_aux

    total_bins = np.prod(shape_hist)

    if not pixel_list:
        pixel_list = list(data.keys())

    count_forests = 0
    for pixel_aux in pixel_list:
        count_forests += len(data[pixel_aux])

    gran_dc = np.zeros((count_forests * max_lenght), dtype = np.float32)
    gran_rx = np.zeros((count_forests * max_lenght), dtype = np.float32)
    gran_ry = np.zeros((count_forests * max_lenght), dtype = np.float32)
    gran_rz = np.zeros((count_forests * max_lenght), dtype = np.float32)
    gran_we = np.zeros((count_forests * max_lenght), dtype = np.float32)
    gran_dw = np.zeros((count_forests * max_lenght), dtype = np.float32)
    gran_x = np.zeros(count_forests, dtype = np.float32)
    gran_y = np.zeros(count_forests, dtype = np.float32)
    gran_z = np.zeros(count_forests, dtype = np.float32)
    gran_odl2 = np.zeros(count_forests, dtype = np.float32)
    gran_dl = np.zeros((count_forests * max_lenght), dtype = np.float32)
    gran_omega = np.zeros(count_forests, dtype = np.float32)
    weight_B = np.zeros(total_bins, dtype = np.float32)
    count = int(0)
    for pixel in pixel_list:
        for forest in data[pixel]:
            len_forest = len(forest.we)
            # lambdaforest.
            gran_dc[count * max_lenght : count * max_lenght + len_forest] = forest.dc
            gran_rx[count * max_lenght : count * max_lenght + len_forest] = forest.rx
            gran_ry[count * max_lenght : count * max_lenght + len_forest] = forest.ry
            gran_rz[count * max_lenght : count * max_lenght + len_forest] = forest.rz
            gran_we[count * max_lenght : count * max_lenght + len_forest] = forest.we
            gran_dw[count * max_lenght : count * max_lenght + len_forest] = forest.dw
            gran_x[count] = forest.x
            gran_y[count] = forest.y
            gran_z[count] = forest.z
            gran_odl2[count] = forest.omega_delta_lambda2
            gran_omega[count] = forest.omega
            gran_dl[count * max_lenght : count * max_lenght + len_forest] = forest.delta_lambda
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
    gran_dl_d = cuda.mem_alloc(lenght_data)
    gran_x_d = cuda.mem_alloc(lenght_data_small)
    gran_y_d = cuda.mem_alloc(lenght_data_small)
    gran_z_d = cuda.mem_alloc(lenght_data_small)
    gran_odl2_d = cuda.mem_alloc(lenght_data_small)
    gran_omega_d = cuda.mem_alloc(lenght_data_small)

    cuda.memcpy_htod(gran_dc_d, gran_dc)
    cuda.memcpy_htod(gran_rx_d, gran_rx)
    cuda.memcpy_htod(gran_ry_d, gran_ry)
    cuda.memcpy_htod(gran_rz_d, gran_rz)
    cuda.memcpy_htod(gran_we_d, gran_we)
    cuda.memcpy_htod(gran_dw_d, gran_dw)
    cuda.memcpy_htod(gran_dl_d, gran_dl)
    cuda.memcpy_htod(gran_x_d, gran_x)
    cuda.memcpy_htod(gran_y_d, gran_y)
    cuda.memcpy_htod(gran_z_d, gran_z)
    cuda.memcpy_htod(gran_odl2_d, gran_odl2)
    cuda.memcpy_htod(gran_omega_d, gran_omega)

    numpix_d = gpuarray.to_gpu(np.array(shape_hist, dtype = np.int32))
    weight_B_d = gpuarray.to_gpu(weight_B)


    global etas12
    global etas21
    global etas22
    global etas13
    global etas31
    global etas23
    global etas32
    global etas33
    global activeBs
    global activeBs_index
    global index_j
    global x12
    global y12
    global z12
    global r12
    global bin_rt
    global bin_rp

    global le1
    global le2
    global le3
    global le4

    global dist_hist_d
    global binner_d


    ldist = np.empty((total_bins,total_bins), dtype = np.float32).nbytes
    dist_hist_d = cuda.mem_alloc(ldist)

    onefloat = np.empty(1, dtype = np.float32).nbytes
    binner_d = gpuarray.to_gpu(np.array([numpix_rp / rmax, numpix_rt / rmax], dtype = np.float32))


    le1 = np.empty(shape_hist + (max_lenght,number_of_neighs), dtype = np.float32)
    le2 = np.empty(shape_hist + (number_of_neighs,), dtype = np.float32)
    le3 = np.zeros(shape_hist + (number_of_neighs,), dtype = np.bool_).nbytes
    le4 = np.empty((number_of_neighs,), dtype = np.float32)

    etas12 = cuda.mem_alloc(le1.nbytes)
    etas21 = cuda.mem_alloc(le1.nbytes)
    etas22 = cuda.mem_alloc(le2.nbytes)
    etas13 = cuda.mem_alloc(le1.nbytes)
    etas31 = cuda.mem_alloc(le1.nbytes)
    etas23 = cuda.mem_alloc(le2.nbytes)
    etas32 = cuda.mem_alloc(le2.nbytes)
    etas33 = cuda.mem_alloc(le2.nbytes)
    activeBs = gpuarray.zeros(shape_hist + (number_of_neighs,), dtype = np.bool_)
    activeBs_index = gpuarray.zeros(shape_hist + (number_of_neighs,), dtype = np.int32)
    index_j = cuda.mem_alloc(le4.nbytes)

    size_auxiliars_real = np.empty((max_lenght, max_lenght, number_of_neighs), dtype = np.float32).nbytes
    size_auxiliars_int = np.empty((max_lenght, max_lenght, number_of_neighs), dtype = np.int32).nbytes
    x12 = cuda.mem_alloc(size_auxiliars_real)
    y12 = cuda.mem_alloc(size_auxiliars_real)
    z12 = cuda.mem_alloc(size_auxiliars_real)
    r12 = cuda.mem_alloc(size_auxiliars_real)
    bin_rt = cuda.mem_alloc(size_auxiliars_int)
    bin_rp = cuda.mem_alloc(size_auxiliars_int)
def distortion_per_pixel(forest_list, **kargs):
    """ This function loops over the forests in a pixel and finds its neighbors
    I will use the method by Helion and only setting r1 as the center node
    of the triangle.
    """

    # Preparing data structure for the partial histograms
    dist_hist = np.empty((total_bins, total_bins), dtype = np.float32)

    cuda.memset_d32_async(dist_hist_d, 0, int(np.prod(dist_hist.shape)))
    weight_B_d.fill(0)

    print('new pixel with', len(forest_list))

    for forest1 in forest_list[:]:
        # Preparing the data structure for the auxiliar histograms
        cuda.memset_d32_async(etas12,0,int(np.prod(le1.shape)))
        cuda.memset_d32_async(etas13,0,int(np.prod(le1.shape)))
        cuda.memset_d32_async(etas21,0,int(np.prod(le1.shape)))
        cuda.memset_d32_async(etas31,0,int(np.prod(le1.shape)))
        cuda.memset_d32_async(etas22,0,int(np.prod(le2.shape)))
        cuda.memset_d32_async(etas23,0,int(np.prod(le2.shape)))
        cuda.memset_d32_async(etas32,0,int(np.prod(le2.shape)))
        cuda.memset_d32_async(etas33,0,int(np.prod(le2.shape)))
        cuda.memset_d32_async(index_j,0,int(np.prod(le4.shape)))
        forest1_lenght = len(forest1.dc)
        # Looking for neighbors
        neighbors_full = forest1.neighborhood(data, angmax)
        number_of_neighs_full = len(neighbors_full)
        number_of_neighs = int(np.ceil(number_of_neighs_full*(1.-reject_fraction)))
        # Choosing only a percentage of the pairs
        random.shuffle(neighbors_full) 
        neighbors = neighbors_full[:number_of_neighs]
        neigh_index = np.array([forest2.index for forest2 in neighbors], dtype = np.int32)
        neigh_sizes = np.array([len(forest2.dc) for forest2 in neighbors], dtype = np.int32)
        base = np.array([forest1.index, forest1_lenght, number_of_neighs], dtype = np.int32)

        activeBs.fill(0)
        activeBs_index.fill(-1)

        if number_of_neighs == 0:
            # This forest have zero neighbors
            continue
        base_d = gpuarray.to_gpu(base)
        neigh_index_d = gpuarray.to_gpu(neigh_index)
        neigh_sizes_d = gpuarray.to_gpu(neigh_sizes)

        # Computing the total number of blocks per kernel. It is determined by the number of elements to be computed.
        total_blocks_x = int(np.ceil(forest1_lenght / threads_per_block[0]))
        total_blocks_y = int(np.ceil(max_lenght / threads_per_block[1]))
        total_blocks_z = int(np.ceil(number_of_neighs / threads_per_block[2]))
        total_blocks_dist = (total_blocks_x, total_blocks_y, total_blocks_z)
        total_blocks_x2 = int(np.ceil(shape_hist[0]*shape_hist[1] / threads_per_block_2[0]))
        total_blocks_y2 = int(np.ceil(number_of_neighs / threads_per_block_2[1]))
        total_blocks_ordering = (total_blocks_x2, total_blocks_y2, 1)

        # This kernel precomputes the distances from forest1 to every other forest in its neighborhood
        precompute_distances(max_lenght, base_d, neigh_index_d, neigh_sizes_d, binner_d,
            gran_rx_d, gran_ry_d, gran_rz_d, gran_x_d, gran_y_d, gran_z_d, gran_dc_d, 
            x12, y12, z12, r12, bin_rp, bin_rt, block = threads_per_block, grid = total_blocks_dist)


        compute_etas(max_lenght, numpix_d, base_d, neigh_index_d, neigh_sizes_d,
            gran_we_d, gran_dl_d, bin_rp, bin_rt,
            gran_odl2_d, gran_omega_d,
            activeBs,
            etas12, etas21, etas22, etas13, etas31, etas23, etas32, etas33,
            weight_B_d,
            block = threads_per_block, grid = total_blocks_dist)


        order_active(activeBs, activeBs_index, numpix_d, base_d, index_j, block=threads_per_block_2, grid=total_blocks_ordering)


        compute_d(max_lenght, numpix_d, base_d, neigh_index_d, neigh_sizes_d,
            gran_we_d, gran_dl_d,
            bin_rp, bin_rt,
            activeBs_index,
            etas12, etas21, etas22, etas13, etas31, etas23, etas32, etas33,
            dist_hist_d,
            block = threads_per_block, grid = total_blocks_dist
            )

    cuda.memcpy_dtoh(dist_hist, dist_hist_d)
    weight_B = weight_B_d.get()

    return (dist_hist, weight_B)
