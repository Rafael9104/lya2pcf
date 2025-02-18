#include <stdio.h>
/* If we want to switch between single and float, change this lines as well as the two lines around 144 */
using myfloat = double;
/* using myfloat = float; */


__global__ void precompute_distance_and_angles(int max_lenght, int *base, int *neigh_index, int *neigh_sizes, float *binner,
    float *rx, float *ry, float *rz, float *x, float *y, float *z,
    float *x12, float *y12, float *z12,
    float *r12, int *bin_r12, int *bin_theta12) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    const int j = blockDim.y*blockIdx.y + threadIdx.y;
    const int k = blockDim.z*blockIdx.z + threadIdx.z;

    const int indice1 = base[0];
    const int size1 = base[1];
    const int number_of_neighs = base[2];
    const float binner_r = binner[0];
    const float binner_theta = binner[1];

    if (i < size1 &&  k < number_of_neighs){
        const int indice2 = neigh_index[k];
        const int size2 = neigh_sizes[k];
        if (j < size2){
            int indice12 = (i * number_of_neighs + k) * max_lenght + j;
            int indice2j = indice2 * max_lenght + j;
            int indice1i = indice1 * max_lenght + i;
            float rx_12 = rx[indice2j] - rx[indice1i];
            float ry_12 = ry[indice2j] - ry[indice1i];
            float rz_12 = rz[indice2j] - rz[indice1i];
            float n_12 = sqrtf(rx_12 * rx_12 + ry_12 * ry_12 + rz_12 * rz_12);
            float inv = 1. / n_12;
            float cos_theta;

            x12[indice12] = rx_12 * inv;
            y12[indice12] = ry_12 * inv;
            z12[indice12] = rz_12 * inv;
            r12[indice12] = n_12;
            cos_theta = x[indice1] * x12[indice12] + y[indice1] * y12[indice12] + z[indice1] * z12[indice12];
            bin_r12[indice12] = int(n_12*binner_r);
            bin_theta12[indice12] = int(acos(cos_theta) * binner_theta);
        }
    }
}

__global__ void precompute_distances(int max_lenght, int *base, int *neigh_index, int *neigh_sizes, float *binner,
    float *rx, float *ry, float *rz, float *x, float *y, float *z, float *dc,
    float *x12, float *y12, float *z12,
    float *r12, int *bin_rp, int *bin_rt) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    const int j = blockDim.y*blockIdx.y + threadIdx.y;
    const int f2 = blockDim.z*blockIdx.z + threadIdx.z;

    const int indice1 = base[0];
    const int size1 = base[1];
    const int number_of_neighs = base[2];
    const float binner_rp = binner[0];
    const float binner_rt = binner[1];

    if (i < size1 &&  f2 < number_of_neighs){
        const int indice2 = neigh_index[f2];
        const int size2 = neigh_sizes[f2];
        if (j < size2){
            int indice12 = (i * number_of_neighs + f2) * max_lenght + j;
            int indice2j = indice2 * max_lenght + j;
            int indice1i = indice1 * max_lenght + i;
            float rx_12 = rx[indice2j] - rx[indice1i];
            float ry_12 = ry[indice2j] - ry[indice1i];
            float rz_12 = rz[indice2j] - rz[indice1i];
            float n_12 = sqrtf(rx_12 * rx_12 + ry_12 * ry_12 + rz_12 * rz_12);
            float inv = 1. / n_12;
            float cos_sq =  x[indice1]*x[indice2] + y[indice1]*y[indice2] + z[indice1]*z[indice2];
            float cos_half12 = sqrtf(0.5 * (1. + cos_sq));
            float sin_half12 = sqrtf(0.5 * (1. - cos_sq));
            float rp = abs(dc[indice1i] - dc[indice2j]) * cos_half12;
            float rt = (dc[indice1i] + dc[indice2j]) * sin_half12;

            x12[indice12] = rx_12 * inv;
            y12[indice12] = ry_12 * inv;
            z12[indice12] = rz_12 * inv;
            r12[indice12] = n_12;

            bin_rp[indice12] = int(rp * binner_rp);
            bin_rt[indice12] = int(rt * binner_rt);
        }
    }
}


__global__ void pair_correlation(int *base, int *neigh_index, int *neigh_sizes,
        int *numpix, int max_lenght,
        myfloat *rmax, myfloat *w_hist, myfloat *dw_hist, 
        myfloat *dc, myfloat *rx, myfloat *ry, myfloat *rz,  myfloat *we, myfloat *dw, myfloat *x, myfloat *y, myfloat *z){
    const myfloat rpmax = rmax[0];
    const myfloat rtmax = rmax[1];
    const int numpix_rp  = numpix[0];
    const int numpix_rt = numpix[1];
    const myfloat binner_rp = numpix_rp/rpmax;
    const myfloat binner_rt = numpix_rt/rtmax;

    const int indice1 = base[0];
    const int size1 = base[1];

    const int numero_neigs = base[2];

    const int i = blockIdx.x;
    const int starty = threadIdx.y;
    const int startz = threadIdx.z;
    const int stridey = blockDim.y;
    const int stridez = blockDim.z;

    const myfloat x1 = x[indice1];
    const myfloat y1 = y[indice1];
    const myfloat z1 = z[indice1];

    int hist_index;

    if(i < size1){
        int indice1i = indice1 * max_lenght + i;
        myfloat rc_1 = dc[indice1i];
        myfloat w_1 = we[indice1i];
        myfloat dw_1 = dw[indice1i];

        for(int j = starty; j < numero_neigs; j+=stridey){
            int indice2 = neigh_index[j];
            int size2 = neigh_sizes[j];

            myfloat x2 = x[indice2];
            myfloat y2 = y[indice2];
            myfloat z2 = z[indice2];
            myfloat cos12 = x1*x2 +  y1*y2 + z1*z2;
            if(cos12 > 1.){ printf("Error in cos12");}
            myfloat cos_half12 = sqrt(0.5 * (1. + cos12));
            myfloat sin_half12 = sqrt(0.5 * (1. - cos12));

            for(int  k = startz; k < size2; k+=stridez){
                int indice2k = indice2 * max_lenght + k;
                myfloat rc_2 = dc[indice2k];
                myfloat w_2 = we[indice2k];
                myfloat dw_2 = dw[indice2k];

                myfloat rp = fabs(rc_1 - rc_2) * cos_half12;
                myfloat rt = (rc_1 + rc_2) * sin_half12;
                
                /* Here we need to use one of the convertions depending on the type of variable */
                
                /* int binp = __float2int_rd(rp * binner_rp); */
                /* int bint = __float2int_rd(rt * binner_rt); */
                int binp = __double2int_rd(rp * binner_rp);
                int bint = __double2int_rd(rt * binner_rt);

                if(binp < numpix_rp && bint < numpix_rt){
                    hist_index = binp*numpix_rt + bint;
                    atomicAdd(&w_hist[hist_index], w_1*w_2);
                    atomicAdd(&dw_hist[hist_index], dw_1*dw_2);

                }

            }
        }
    }
}


__global__ void compute_etas(int max_lenght, int *numpix, int *base, int *neigh_index, int *neigh_sizes,
    float *we, float *delta_lambda,  int *bin_rp, int *bin_rt,
    float *omega_delta_lambda2, float *omega,
    bool *ActiveBs,
    float *eta12, float *eta21, float *eta22, float *eta13, float *eta31, float *eta23, float *eta32, float *eta33,
    float *weight_B) {

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
    bool *ActiveBs, int *ActiveBs_index, int *numpix, int *base, int *index_j) {
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

__global__ void compute_d(int max_lenght, int *numpix, int *base, int *neigh_index, int *neigh_sizes,
    float *we, float *delta_lambda, 
    int *bin_rp, int *bin_rt,
    int *ActiveBs_index,
    float *eta12, float *eta21, float *eta22, float *eta13, float *eta31, float *eta23, float *eta32, float *eta33,
    float *d_hist) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;
    const int j = blockDim.y*blockIdx.y + threadIdx.y;
    const int f2 = blockDim.z*blockIdx.z + threadIdx.z;

    const int indice1 = base[0];
    const int size1 = base[1];
    const int number_of_neighs = base[2];

    int k;
    int small_index;
    int B;

    if (i < size1 &&  f2 < number_of_neighs){
        const int indice2 = neigh_index[f2];
        const int size2 = neigh_sizes[f2];
        if (j < size2){
            const int indice12 = (i * number_of_neighs + f2) * max_lenght + j;
            const int numpix_rt = numpix[1];
            const int numpix_rp = numpix[0];
            const int tot_pix = numpix_rp*numpix_rt;
            const int A = bin_rp[indice12] * numpix_rt + bin_rt[indice12];
            const int indice1i = indice1 * max_lenght + i;
            const int indice2j = indice2 * max_lenght + j;
            const float w12 = we[indice1i]*we[indice2j];

            if (bin_rt[indice12] < numpix_rt && bin_rp[indice12] < numpix_rp){
                atomicAdd(&d_hist[A*(1+tot_pix)], w12);
                k = 0;
                while(ActiveBs_index[f2*tot_pix + k] > -1 && k < tot_pix){
            // printf("voyy en i = %d,f2= %d, k = %d \n", i, f2,k);
                    B =  ActiveBs_index[f2*tot_pix + k];
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
                    k++;
                }
            }
        }
    }
}

