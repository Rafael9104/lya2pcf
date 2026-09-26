import numpy as np
import random

import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401  (creates the CUDA context every pycuda call in this module uses)
import pycuda.gpuarray as gpuarray

from . import parameters as params
from . import gpu_support
from . import streaming_upload

mod = gpu_support.compile_kernels()
precompute_distances = mod.get_function("precompute_distances")
compute_etas = mod.get_function("compute_etas")
compute_d = mod.get_function("compute_d")
order_active = mod.get_function("order_active")


class DistortionGPU:
    """ The forests of one MPI rank on the GPU, the distortion's scratch buffers, and
    the kernel launches that compute the distortion matrix over them.

    Copies the forests of `plan` to the GPU once, to avoid the overhead of copying
    them at every call, and allocates the scratch buffers.
    Parameters:
    plan        pixel_partition.ForestPlan
                The rank's own pixels plus their neighbour buffer. The data*.npy files are
                read one at a time and copied pixel by pixel, never all held in host memory
                (streaming_upload.py); each forest's per-pixel arrays are dropped once
                uploaded.
    shape_hist  shape of the distortion's (rp, rt) histogram
    angmax      maximum angle between two forests that are neighbours
    reject_fraction  fraction of each forest's neighbours that is left out

    `data` is the {pixel: [forests]} dict distortion_per_pixel's forests come from,
    with the per-pixel arrays already dropped. `clamped_forests` counts the forests
    whose kept neighbours were capped at params.number_of_neighs, and
    `forests_seen` all forests, over the whole run (reported at the end by
    distortion.py).
    """

    def __init__(self, plan, shape_hist, angmax, reject_fraction):
        self.shape_hist = shape_hist
        self.angmax = angmax
        self.reject_fraction = reject_fraction
        self.clamped_forests = 0
        self.forests_seen = 0
        # The neighbours kept for each forest are a random subset.
        random.seed(1)

        self.total_bins = np.prod(shape_hist)
        total_bins = self.total_bins

        count_forests = plan.count_forests
        # The kernels take this as an int argument, which pycuda can only marshal
        # from a fixed-width type, not a plain Python int.
        self.max_lenght = np.int32(plan.max_lenght)
        max_lenght = self.max_lenght

        # Checked before anything is allocated, so a slice that does not fit is
        # reported as a whole rather than as whichever allocation happened to fail.
        # int() guards against the int32 max_lenght overflowing these products.
        itemsize = np.dtype(params.gpu_dtype).itemsize
        ml, neighs, bins = int(max_lenght), params.number_of_neighs, int(total_bins)
        forest_bytes = count_forests * ml * itemsize
        gpu_support.require_memory(
            7 * forest_bytes                             # dc, rx, ry, rz, we, dw, dl
            + 5 * count_forests * itemsize               # x, y, z, odl2, omega
            + bins * itemsize                            # weight_B
            + bins * bins * itemsize                     # dist_hist
            + 4 * bins * ml * neighs * itemsize          # etas12/21/13/31
            + 4 * bins * neighs * itemsize               # etas22/23/32/33
            + bins * neighs + bins * neighs * 4          # activeBs + index
            + neighs * 4                                 # index_j
            + 4 * ml * ml * neighs * itemsize            # x12/y12/z12/r12
            + 2 * ml * ml * neighs * 4,                  # bin_rp/bin_rt
            "distortion buffers (longest forest %d pixels, %d neighbours, %d bins)"
            % (ml, neighs, bins),
            ["coadd/rebin the deltas upstream: the x12/y12/z12/r12 buffers scale "
             "as the square of the forest length, so coadding by 3 saves about 4x",
             "lower 'number_of_neighs' in parameters.yml, which scales most buffers",
             "use coarser binning (larger bin_size_r, or smaller rmax)"])

        big, small, self.data = streaming_upload.stream_forests_to_gpu(
            plan, ('dc', 'rx', 'ry', 'rz', 'we', 'dw', 'delta_lambda'),
            ('x', 'y', 'z', 'omega_delta_lambda2', 'omega'), params.gpu_dtype)
        self.gran_dc_d, self.gran_rx_d, self.gran_ry_d, self.gran_rz_d = big['dc'], big['rx'], big['ry'], big['rz']
        self.gran_we_d, self.gran_dw_d, self.gran_dl_d = big['we'], big['dw'], big['delta_lambda']
        self.gran_x_d, self.gran_y_d, self.gran_z_d = small['x'], small['y'], small['z']
        self.gran_odl2_d, self.gran_omega_d = small['omega_delta_lambda2'], small['omega']

        self.numpix_d = gpuarray.to_gpu(np.array(shape_hist, dtype = np.int32))
        self.weight_B_d = gpuarray.to_gpu(np.zeros(total_bins, dtype = params.gpu_dtype))

        # compute_d's per-block cache of order_active's output (see the kernel's
        # own comment and IMPROVEMENTS.md #22): one tot_pix-wide int32 slice per
        # distinct neighbour (blockDim.z) the block covers, sized for the worst
        # case since the real per-neighbour count is only known at kernel
        # runtime. Checked once here, at the same size for every launch, for the
        # same reason as pair_correlation's shared-memory check in
        # correlation_procedures_pycuda.py.
        self.compute_d_shared_bytes = (
            params.distortion_threads_per_block[2] * int(total_bins) * np.dtype(np.int32).itemsize)
        max_shared = cuda.Context.get_device().get_attribute(
            cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK)
        if self.compute_d_shared_bytes > max_shared:
            raise RuntimeError(
                "compute_d's per-block active-bin cache needs %d bytes of shared "
                "memory (distortion_threads_per_block[2]=%d * %d bins * %d bytes), "
                "but this GPU only has %d bytes per block. Use fewer/coarser bins "
                "(numpix_rp x numpix_rt, from rmax and bin_size_r in "
                "parameters.yml) or a smaller distortion_threads_per_block[2] to fit."
                % (self.compute_d_shared_bytes, params.distortion_threads_per_block[2],
                   int(total_bins), np.dtype(np.int32).itemsize, max_shared))

        ldist = np.empty((total_bins,total_bins), dtype = params.gpu_dtype).nbytes
        self.dist_hist_d = cuda.mem_alloc(ldist)

        self.binner_d = gpuarray.to_gpu(np.array([params.numpix_rp / params.rmax, params.numpix_rt / params.rmax], dtype = params.gpu_dtype))

        # Sizes in bytes of the etas buffers (the ones with a pixel axis, the ones
        # without) and of index_j; memset to zero for every forest.
        self.etas_long_bytes = np.empty(shape_hist + (max_lenght,params.number_of_neighs), dtype = params.gpu_dtype).nbytes
        self.etas_short_bytes = np.empty(shape_hist + (params.number_of_neighs,), dtype = params.gpu_dtype).nbytes
        self.index_j_bytes = np.empty((params.number_of_neighs,), dtype = np.int32).nbytes

        self.etas12 = cuda.mem_alloc(self.etas_long_bytes)
        self.etas21 = cuda.mem_alloc(self.etas_long_bytes)
        self.etas22 = cuda.mem_alloc(self.etas_short_bytes)
        self.etas13 = cuda.mem_alloc(self.etas_long_bytes)
        self.etas31 = cuda.mem_alloc(self.etas_long_bytes)
        self.etas23 = cuda.mem_alloc(self.etas_short_bytes)
        self.etas32 = cuda.mem_alloc(self.etas_short_bytes)
        self.etas33 = cuda.mem_alloc(self.etas_short_bytes)
        self.activeBs = gpuarray.zeros(shape_hist + (params.number_of_neighs,), dtype = np.bool_)
        self.activeBs_index = gpuarray.zeros(shape_hist + (params.number_of_neighs,), dtype = np.int32)
        self.index_j = cuda.mem_alloc(self.index_j_bytes)

        size_auxiliars_real = np.empty((max_lenght, max_lenght, params.number_of_neighs), dtype = params.gpu_dtype).nbytes
        size_auxiliars_int = np.empty((max_lenght, max_lenght, params.number_of_neighs), dtype = np.int32).nbytes
        self.x12 = cuda.mem_alloc(size_auxiliars_real)
        self.y12 = cuda.mem_alloc(size_auxiliars_real)
        self.z12 = cuda.mem_alloc(size_auxiliars_real)
        self.r12 = cuda.mem_alloc(size_auxiliars_real)
        self.bin_rt = cuda.mem_alloc(size_auxiliars_int)
        self.bin_rp = cuda.mem_alloc(size_auxiliars_int)

    def distortion_per_pixel(self, forest_list):
        """ This function loops over the forests in a pixel and finds its neighbors
        I will use the method by Helion and only setting r1 as the center node
        of the triangle.
        """
        shape_hist = self.shape_hist
        max_lenght = self.max_lenght

        # Preparing data structure for the partial histograms
        dist_hist = np.empty((self.total_bins, self.total_bins), dtype = params.gpu_dtype)

        cuda.memset_d8_async(self.dist_hist_d, 0, dist_hist.nbytes)
        self.weight_B_d.fill(0)

        for forest1 in forest_list[:]:
            # Preparing the data structure for the auxiliar histograms
            cuda.memset_d8_async(self.etas12, 0, self.etas_long_bytes)
            cuda.memset_d8_async(self.etas13, 0, self.etas_long_bytes)
            cuda.memset_d8_async(self.etas21, 0, self.etas_long_bytes)
            cuda.memset_d8_async(self.etas31, 0, self.etas_long_bytes)
            cuda.memset_d8_async(self.etas22, 0, self.etas_short_bytes)
            cuda.memset_d8_async(self.etas23, 0, self.etas_short_bytes)
            cuda.memset_d8_async(self.etas32, 0, self.etas_short_bytes)
            cuda.memset_d8_async(self.etas33, 0, self.etas_short_bytes)
            cuda.memset_d8_async(self.index_j, 0, self.index_j_bytes)
            forest1_lenght = forest1.num_points
            # Looking for neighbors
            neighbors_full = forest1.neighborhood(self.data, self.angmax)
            number_of_neighs_full = len(neighbors_full)
            self.forests_seen += 1
            number_of_neighs = int(np.ceil(number_of_neighs_full*(1.-self.reject_fraction)))
            # The scratch buffers are sized for params.number_of_neighs neighbours
            # per forest; keeping more would make the kernels index past them.
            if number_of_neighs > params.number_of_neighs:
                number_of_neighs = params.number_of_neighs
                self.clamped_forests += 1
            # Choosing only a percentage of the pairs
            random.shuffle(neighbors_full)
            neighbors = neighbors_full[:number_of_neighs]
            neigh_index = np.array([forest2.index for forest2 in neighbors], dtype = np.int32)
            neigh_sizes = np.array([forest2.num_points for forest2 in neighbors], dtype = np.int32)
            base = np.array([forest1.index, forest1_lenght, number_of_neighs], dtype = np.int32)

            self.activeBs.fill(0)
            self.activeBs_index.fill(-1)

            if number_of_neighs == 0:
                # This forest have zero neighbors
                continue
            base_d = gpuarray.to_gpu(base)
            neigh_index_d = gpuarray.to_gpu(neigh_index)
            neigh_sizes_d = gpuarray.to_gpu(neigh_sizes)

            # Computing the total number of blocks per kernel. It is determined by the number of elements to be computed.
            total_blocks_x = int(np.ceil(forest1_lenght / params.distortion_threads_per_block[0]))
            # y only needs to cover this launch's actual kept neighbours
            # (neigh_sizes), not the dataset-wide max_lenght every forest's
            # buffer slot is padded to -- a forest1's neighbours are usually
            # shorter than the longest forest in the whole set, so this skips
            # blocks that would do no work (every thread in them fails the
            # kernels' own `j < size2` check). Safe by construction: no
            # neighbour's size2 exceeds neigh_sizes.max(), so no valid j is cut
            # off. See IMPROVEMENTS.md #22.
            total_blocks_y = int(np.ceil(int(neigh_sizes.max()) / params.distortion_threads_per_block[1]))
            total_blocks_z = int(np.ceil(number_of_neighs / params.distortion_threads_per_block[2]))
            total_blocks_dist = (total_blocks_x, total_blocks_y, total_blocks_z)
            total_blocks_x2 = int(np.ceil(shape_hist[0]*shape_hist[1] / params.threads_per_block_2d[0]))
            total_blocks_y2 = int(np.ceil(number_of_neighs / params.threads_per_block_2d[1]))
            total_blocks_ordering = (total_blocks_x2, total_blocks_y2, 1)
            # order_active is 2D (it never reads a z thread/block index), but
            # pycuda's block= always takes a 3-tuple, so the unused z=1 is
            # appended here rather than carried in parameters.yml.
            block_ordering = params.threads_per_block_2d + (1,)

            # This kernel precomputes the distances from forest1 to every other forest in its neighborhood
            precompute_distances(max_lenght, base_d, neigh_index_d, neigh_sizes_d, self.binner_d,
                self.gran_rx_d, self.gran_ry_d, self.gran_rz_d, self.gran_x_d, self.gran_y_d, self.gran_z_d, self.gran_dc_d,
                self.x12, self.y12, self.z12, self.r12, self.bin_rp, self.bin_rt,
                block = params.distortion_threads_per_block, grid = total_blocks_dist)


            compute_etas(max_lenght, self.numpix_d, base_d, neigh_index_d, neigh_sizes_d,
                self.gran_we_d, self.gran_dl_d, self.bin_rp, self.bin_rt,
                self.gran_odl2_d, self.gran_omega_d,
                self.activeBs,
                self.etas12, self.etas21, self.etas22, self.etas13, self.etas31, self.etas23, self.etas32, self.etas33,
                self.weight_B_d,
                block = params.distortion_threads_per_block, grid = total_blocks_dist)


            order_active(self.activeBs, self.activeBs_index, self.numpix_d, base_d, self.index_j,
                block=block_ordering, grid=total_blocks_ordering)


            compute_d(max_lenght, self.numpix_d, base_d, neigh_index_d, neigh_sizes_d,
                self.gran_we_d, self.gran_dl_d,
                self.bin_rp, self.bin_rt,
                self.activeBs_index, self.index_j,
                self.etas12, self.etas21, self.etas22, self.etas13, self.etas31, self.etas23, self.etas32, self.etas33,
                self.dist_hist_d,
                block = params.distortion_threads_per_block, grid = total_blocks_dist,
                shared = self.compute_d_shared_bytes
                )

        cuda.memcpy_dtoh(dist_hist, self.dist_hist_d)
        weight_B = self.weight_B_d.get()

        return (dist_hist, weight_B)
