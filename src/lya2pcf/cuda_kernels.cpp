#include <stdio.h>

/* Precision of the two-point correlation kernel and its device buffers.
   Set from Python via -DMYFLOAT=<float|double>, driven by the gpu_precision
   setting in parameters.yml. The default below applies only when this file is
   compiled by hand.

   Note atomicAdd on double needs compute capability >= 6.0 (Pascal); on older
   cards only float compiles. This applies to every kernel in this file, since
   they are compiled together. */
#ifndef MYFLOAT
#define MYFLOAT double
#endif
using myfloat = MYFLOAT;

/* Signature convention: a kernel's inputs are `const T *`, and the buffers it
   writes are plain `T *`. On top of that, `__restrict__` is on every pointer of
   precompute_distances, compute_etas and order_active, where it was measured to
   help: with the promise (no two pointers of one launch refer to the same
   memory; the drivers allocate a distinct buffer for every argument) sm_5x
   (Maxwell) loads go through the cached, non-coherent path (LDG.E.CI) and the
   compiler stops reloading values it already has (compute_etas: 41 loads ->
   17), 7-35% faster. It is deliberately NOT on pair_correlation (about 1.4%
   slower in four alternated runs) nor on compute_d (0-4% slower, never faster):
   for those two `const` alone changes nothing in the generated code. Measured
   on a GTX 970, float32 (IMPROVEMENTS.md #25); recheck on other hardware. */

/* Expands to __float2int_rd or __double2int_rd to match MYFLOAT, so the
   rounding intrinsic follows the same single setting. An overloaded function
   would be simpler, but pycuda compiles this file inside extern "C", which
   does not allow overloads. */
#define CONCAT_(a, b) a##b
#define CONCAT(a, b) CONCAT_(a, b)
#define myfloat2int_rd CONCAT(CONCAT(__, MYFLOAT), 2int_rd)


__global__ void precompute_distances(int max_lenght, const int * __restrict__ base,
    const int * __restrict__ neigh_index, const int * __restrict__ neigh_sizes, const myfloat * __restrict__ binner,
    const myfloat * __restrict__ x, const myfloat * __restrict__ y, const myfloat * __restrict__ z,
    const myfloat * __restrict__ dc,
    int * __restrict__ bin_rp, int * __restrict__ bin_rt) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    const int j = blockDim.y*blockIdx.y + threadIdx.y;
    const int f2 = blockDim.z*blockIdx.z + threadIdx.z;

    const int indice1 = base[0];
    const int size1 = base[1];
    const int number_of_neighs = base[2];
    const myfloat binner_rp = binner[0];
    const myfloat binner_rt = binner[1];

    if (i < size1 &&  f2 < number_of_neighs){
        const int indice2 = neigh_index[f2];
        const int size2 = neigh_sizes[f2];
        if (j < size2){
            int indice12 = (i * number_of_neighs + f2) * max_lenght + j;
            int indice2j = indice2 * max_lenght + j;
            int indice1i = indice1 * max_lenght + i;
            myfloat cos_sq =  x[indice1]*x[indice2] + y[indice1]*y[indice2] + z[indice1]*z[indice2];
            myfloat cos_half12 = sqrt(0.5 * (1. + cos_sq));
            myfloat sin_half12 = sqrt(0.5 * (1. - cos_sq));
            myfloat rp = fabs(dc[indice1i] - dc[indice2j]) * cos_half12;
            myfloat rt = (dc[indice1i] + dc[indice2j]) * sin_half12;

            bin_rp[indice12] = int(rp * binner_rp);
            bin_rt[indice12] = int(rt * binner_rt);
        }
    }
}


__global__ void __launch_bounds__(1024, 2) pair_correlation(
        const int indice1, const int size1, const int numero_neigs,
        const int * neigh_index, const int * neigh_sizes,
        const int numpix_rp, const int numpix_rt, const int max_lenght,
        const myfloat rpmax, const myfloat rtmax,
        myfloat * w_hist, myfloat * dw_hist,
        const myfloat * dc, const myfloat * we, const myfloat * dw,
        const myfloat * x, const myfloat * y, const myfloat * z){
    /* __launch_bounds__(1024, 2): the block is 1024 threads, and two blocks
       fit on an SM only if the kernel uses at most 32 registers per thread
       (65536 / (1024 * 2)); at 33 or more the count floors to one block per SM
       (50% occupancy). Without the hint nvcc allocates 37 in float, so the hint
       is what keeps two blocks resident. Passing the per-launch scalars by
       value (below) brings the natural demand down to 32 in float, so the hint
       no longer forces a spill there; in double it still spills a little
       (IMPROVEMENTS.md #23, #24).

       Per-block (per pixel-of-forest1) private histogram, in shared memory.
       Sized dynamically at launch (2 * numpix_rp * numpix_rt * sizeof(myfloat),
       see correlation_procedures_pycuda.py's `shared=`) since numpix comes from
       parameters.yml, not a compile-time constant. Every thread of the block
       accumulates into this instead of the global histogram, so the ~1024
       threads' worth of atomics per pixel pair only ever contend on-chip;
       only the final per-bin reduction below touches global memory, and only
       for bins this block actually hit (see IMPROVEMENTS.md's kernel
       profiling entry for the atomic-contention numbers this replaces).

       The per-launch scalars (indice1, size1, numero_neigs, numpix_rp/rt,
       max_lenght, rpmax/rtmax) are passed by value, not through 2-3
       element device arrays: kernel parameters sit in the constant
       parameter bank and are read as instruction operands, so they need
       neither a global load nor a register to hold them, and the driver
       no longer builds and copies a small array per forest. */
    extern __shared__ myfloat pc_shared[];

    const myfloat binner_rp = numpix_rp/rpmax;
    const myfloat binner_rt = numpix_rt/rtmax;
    const int tot_pix = numpix_rp * numpix_rt;

    myfloat *sh_w = pc_shared;
    myfloat *sh_dw = pc_shared + tot_pix;


    const int i = blockIdx.x;
    /* threadIdx.y is the fastest-varying dimension a warp packs (blockDim.x
       is forced to 1 by the caller), so it drives k, the index *within* a
       neighbour forest: dc/we/dw store a forest's pixels
       contiguously (offset = indice2*max_lenght + k), so a warp of
       consecutive k at fixed neighbour reads consecutive addresses --
       coalesced. threadIdx.z drives j, the neighbour index, which is a
       gather over widely separated forests regardless of dimension order,
       so it is the one kept outside the coalesced dimension. (The previous
       version had this the other way round: a warp spanned 32 different
       neighbours at one shared k, i.e. 32 unrelated addresses per load --
       measured at 14.6% global load efficiency, see IMPROVEMENTS.md.) */
    const int startk = threadIdx.y;
    const int startj = threadIdx.z;
    const int stridek = blockDim.y;
    const int stridej = blockDim.z;

    const int tid = threadIdx.y + threadIdx.z * blockDim.y;
    const int nthreads = blockDim.y * blockDim.z;

    for (int b = tid; b < tot_pix; b += nthreads) {
        sh_w[b] = 0;
        sh_dw[b] = 0;
    }
    __syncthreads();

    const myfloat x1 = x[indice1];
    const myfloat y1 = y[indice1];
    const myfloat z1 = z[indice1];

    int hist_index;

    if(i < size1){
        int indice1i = indice1 * max_lenght + i;
        myfloat rc_1 = dc[indice1i];
        myfloat w_1 = we[indice1i];
        myfloat dw_1 = dw[indice1i];

        for(int j = startj; j < numero_neigs; j+=stridej){
            int indice2 = neigh_index[j];
            int size2 = neigh_sizes[j];

            myfloat x2 = x[indice2];
            myfloat y2 = y[indice2];
            myfloat z2 = z[indice2];
            myfloat cos12 = x1*x2 +  y1*y2 + z1*z2;
            if(cos12 > 1.){ printf("Error in cos12");}
            myfloat cos_half12 = sqrt(0.5 * (1. + cos12));
            myfloat sin_half12 = sqrt(0.5 * (1. - cos12));

            for(int  k = startk; k < size2; k+=stridek){
                int indice2k = indice2 * max_lenght + k;
                myfloat rc_2 = dc[indice2k];
                myfloat w_2 = we[indice2k];
                myfloat dw_2 = dw[indice2k];

                myfloat rp = fabs(rc_1 - rc_2) * cos_half12;
                myfloat rt = (rc_1 + rc_2) * sin_half12;

                int binp = myfloat2int_rd(rp * binner_rp);
                int bint = myfloat2int_rd(rt * binner_rt);

                if(binp < numpix_rp && bint < numpix_rt){
                    hist_index = binp*numpix_rt + bint;
                    atomicAdd(&sh_w[hist_index], w_1*w_2);
                    atomicAdd(&sh_dw[hist_index], dw_1*dw_2);

                }

            }
        }
    }

    __syncthreads();
    for (int b = tid; b < tot_pix; b += nthreads) {
        if (sh_w[b] != 0) atomicAdd(&w_hist[b], sh_w[b]);
        if (sh_dw[b] != 0) atomicAdd(&dw_hist[b], sh_dw[b]);
    }
}


__global__ void compute_etas(int max_lenght, const int * __restrict__ numpix, const int * __restrict__ base,
    const int * __restrict__ neigh_index, const int * __restrict__ neigh_sizes,
    const myfloat * __restrict__ we, const myfloat * __restrict__ delta_lambda,
    const int * __restrict__ bin_rp, const int * __restrict__ bin_rt,
    const myfloat * __restrict__ omega_delta_lambda2, const myfloat * __restrict__ omega,
    bool * __restrict__ ActiveBs,
    myfloat * __restrict__ eta12, myfloat * __restrict__ eta21, myfloat * __restrict__ eta22,
    myfloat * __restrict__ eta13, myfloat * __restrict__ eta31, myfloat * __restrict__ eta23,
    myfloat * __restrict__ eta32, myfloat * __restrict__ eta33,
    myfloat * __restrict__ weight_B) {

    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    const int j = blockDim.y*blockIdx.y + threadIdx.y;
    const int f2 = blockDim.z*blockIdx.z + threadIdx.z;

    const int indice1 = base[0];
    const int size1 = base[1];
    const int number_of_neighs = base[2];

    if (i < size1 &&  f2 < number_of_neighs){
        const int indice2 = neigh_index[f2];
        const int size2 = neigh_sizes[f2];
        if (j < size2){
            const int indice12 = (i * number_of_neighs + f2) * max_lenght + j;
            const int numpix_rt = numpix[1];
            const int numpix_rp = numpix[0];
            const int tot_pix = numpix_rp*numpix_rt;
            const int B = bin_rp[indice12] * numpix_rt + bin_rt[indice12];
            const int indice1i = indice1 * max_lenght + i;
            const int indice2j = indice2 * max_lenght + j;
            const int small_index =  f2*tot_pix + B;

            if (bin_rt[indice12] < numpix_rt && bin_rp[indice12] < numpix_rp){
                ActiveBs[f2*tot_pix + B] = true;
                atomicAdd(&eta12[small_index*max_lenght + i], we[indice2j]/omega[indice2]);
                atomicAdd(&eta21[small_index*max_lenght + j], we[indice1i]/omega[indice1]);
                atomicAdd(&eta13[small_index*max_lenght + i], we[indice2j]*delta_lambda[indice2j]/omega_delta_lambda2[indice2]);
                atomicAdd(&eta31[small_index*max_lenght + j], we[indice1i]*delta_lambda[indice1i]/omega_delta_lambda2[indice1]);
                atomicAdd(&eta22[small_index], we[indice1i]*we[indice2j]/(omega[indice1]*omega[indice2]));
                atomicAdd(&eta23[small_index], we[indice1i]*we[indice2j]*delta_lambda[indice2j]/(omega_delta_lambda2[indice2]*omega[indice1]));
                atomicAdd(&eta32[small_index], we[indice1i]*delta_lambda[indice1i]*we[indice2j]/(omega_delta_lambda2[indice1]*omega[indice2]));
                atomicAdd(&eta33[small_index], we[indice1i]*delta_lambda[indice1i]*we[indice2j]*delta_lambda[indice2j]/(omega_delta_lambda2[indice1]*omega_delta_lambda2[indice2]));
                atomicAdd(&weight_B[B], we[indice1i]*we[indice2j]);
                // if (B==1299&&base[0]==61){printf("este fue %d, %d",base[0],f2);}
            }
        }
    }
}

__global__ void order_active(
    const bool * __restrict__ ActiveBs, int * __restrict__ ActiveBs_index,
    const int * __restrict__ numpix, const int * __restrict__ base, int * __restrict__ index_j) {
    const int f2 = blockDim.y*blockIdx.y + threadIdx.y;
    const int B = blockDim.x*blockIdx.x + threadIdx.x;
    const int tot_pix = numpix[0] * numpix[1];
    const int number_of_neighs = base[2];
    int j;
    if (f2 < number_of_neighs && B < tot_pix){
        if (ActiveBs[f2*tot_pix + B] == true) {
            j = atomicAdd(&index_j[f2], 1);
            ActiveBs_index[f2*tot_pix + j] = B;
        }
    }
}

__global__ void compute_d(int max_lenght, const int * numpix, const int * base,
    const int * neigh_index, const int * neigh_sizes,
    const myfloat * we, const myfloat * delta_lambda,
    const int * bin_rp, const int * bin_rt,
    const int * ActiveBs_index, const int * index_j,
    const myfloat * eta12, const myfloat * eta21, const myfloat * eta22,
    const myfloat * eta13, const myfloat * eta31, const myfloat * eta23,
    const myfloat * eta32, const myfloat * eta33,
    myfloat * d_hist) {
    /* Every (i, j) thread that shares a neighbour f2 needs the same short
       list of that neighbour's active bins -- order_active's output,
       ActiveBs_index[f2*tot_pix : f2*tot_pix + index_j[f2]]. The original
       version had every thread read that list from global memory on its
       own, an independent serially-dependent chain repeated up to
       blockDim.x * blockDim.y times per f2 (measured: 141M atomic
       transactions and 77% memory-dependency stalls on one launch, see
       IMPROVEMENTS.md #22). Caching it once per block in shared memory,
       read by every thread from there instead, removes that redundant
       traffic. Sized for the worst case (every bin active) since the real
       count (index_j[f2]) is only known once the kernel runs; see
       distortion_procedures_pycuda.py's shared-memory size check. */
    extern __shared__ int sh_active[];

    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    const int j = blockDim.y*blockIdx.y + threadIdx.y;
    const int f2 = blockDim.z*blockIdx.z + threadIdx.z;

    const int indice1 = base[0];
    const int size1 = base[1];
    const int number_of_neighs = base[2];
    const int numpix_rt = numpix[1];
    const int numpix_rp = numpix[0];
    const int tot_pix = numpix_rp*numpix_rt;

    int k;
    int small_index;
    int B;

    // Cooperative load: this block covers up to blockDim.z distinct f2's
    // (one per threadIdx.z), each copied into its own tot_pix-wide slice
    // of shared memory by the blockDim.x*blockDim.y threads that share
    // it -- regardless of whether those threads' own (i, j) turns out
    // in-bounds below, since the destination only depends on threadIdx.z.
    int *my_active = sh_active + threadIdx.z * tot_pix;
    if (f2 < number_of_neighs) {
        const int active_count = index_j[f2];
        const int tid_xy = threadIdx.x + threadIdx.y * blockDim.x;
        const int nthreads_xy = blockDim.x * blockDim.y;
        for (int k2 = tid_xy; k2 < active_count && k2 < tot_pix; k2 += nthreads_xy) {
            my_active[k2] = ActiveBs_index[f2*tot_pix + k2];
        }
    }
    __syncthreads();

    if (i < size1 &&  f2 < number_of_neighs){
        const int indice2 = neigh_index[f2];
        const int size2 = neigh_sizes[f2];
        if (j < size2){
            const int indice12 = (i * number_of_neighs + f2) * max_lenght + j;
            const int A = bin_rp[indice12] * numpix_rt + bin_rt[indice12];
            const int indice1i = indice1 * max_lenght + i;
            const int indice2j = indice2 * max_lenght + j;
            const myfloat w12 = we[indice1i]*we[indice2j];

            if (bin_rt[indice12] < numpix_rt && bin_rp[indice12] < numpix_rp){
                atomicAdd(&d_hist[A*(1+tot_pix)], w12);
                const int active_count = index_j[f2];
                for (k = 0; k < active_count && k < tot_pix; k++){
                    B = my_active[k];
                    small_index = f2*tot_pix + B;
                    atomicAdd(&d_hist[A*tot_pix + B], w12*(
                            delta_lambda[indice1i]*delta_lambda[indice2j]*eta33[small_index]
                            + delta_lambda[indice1i]*eta32[small_index]
                            + delta_lambda[indice2j]*eta23[small_index]
                            + eta22[small_index]
                            - delta_lambda[indice1i]*eta31[small_index*max_lenght + j]
                            - delta_lambda[indice2j]*eta13[small_index*max_lenght + i]
                            - eta21[small_index*max_lenght + j]
                            - eta12[small_index*max_lenght + i]
                            ));
                }
            }
        }
    }
}

