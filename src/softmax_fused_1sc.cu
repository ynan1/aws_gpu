#include "softmax_cuda_fused.cuh"

//Kernel to find the softmax
__device__ float max_val[N_BLOCKS]; // Pointer to max_val in device memory
__device__ int total_inc=0;
__device__ float norm_glb[N_BLOCKS];
__device__ volatile bool is_max_val_set_g=false;
__device__ volatile bool is_norm_val_set_g=false;

__global__ void softmax_fused_1sc( const float* d_in,float* d_out,const int* N_loops) {
    int idx = threadIdx.x;
    int ix= blockIdx.x * blockDim.x + idx;
    __shared__ float shared_max_norm[THREADS_PER_BLOCK];
    float* d_in_ptr = const_cast<float*>(d_in + blockIdx.x * *N_loops * blockDim.x); // Cast to non-const pointer
    float max_l = -FLT_MAX;
    int stride = blockDim.x; // Number of threads in a block

    // Find the maximum value per thread block
    for (int i = idx; i < *N_loops * blockDim.x; i+= stride) {
        max_l = fmaxf(max_l, d_in_ptr[i]);
    }
 
    shared_max_norm[idx] = max_l; // Store the maximum value in shared memory
    __syncthreads();


    // Reduce to find the maximum value in shared memory
    for (int s = stride / 2; s > 0; s >>= 1) {
        if (idx < s && idx + s < THREADS_PER_BLOCK) {
            shared_max_norm[idx] = fmaxf(shared_max_norm[idx], shared_max_norm[idx + s]);
        }
        __syncthreads();
    }

    // Write the maximum value to global memory
    if (idx == 0) { // Only the first thread writes to global memory
        max_val[blockIdx.x] = shared_max_norm[0]; // Store the maximum value for this block
        // __threadfence();
        // is_max_val_set[blockIdx.x] = true;
        int ticket=atomicAdd(&total_inc, 1);
        if (ticket == gridDim.x-1) {
            is_max_val_set_g = true; // Set the global flag to indicate max value is set for the block
        }

    }


    // Considering N_blocks < THREADS_PER_BLOCK, it should run within a single block
    while(!(is_max_val_set_g));
    for (int i = gridDim.x/2; i > 0; i >>= 1) {
        if (ix < i && ix + i < N_BLOCKS) {
            // while (!(is_max_val_set[ix] && is_max_val_set[ix + i]));
            max_val[ix] = fmaxf(max_val[ix], max_val[ix + i]);
        }
        __syncthreads();
    }

    if (ix == 0) {
        // __threadfence();
        is_max_val_set_g = false; // Set the global flag to indicate max value is set for the block
        total_inc = 0;
    }
 

    while(is_max_val_set_g);
    max_l = max_val[0]; // maximum value across all blocks


    // if (ix == 0||ix == 32767){
    //     printf("Max value for block %d: %.9f\n", blockIdx.x, max_l);
    // }

    //Exponent and normalization
    float *d_out_ptr = d_out + blockIdx.x * *N_loops * blockDim.x; // Adjust output pointer to the correct block
    float norm = 0.0f;
    //__shared__ float shared_norm[THREADS_PER_BLOCK];

    for (int i = idx; i < *N_loops * blockDim.x; i += stride) {
        d_out_ptr[i] = expf(d_in_ptr[i] - max_l);
        if (d_in_ptr[i] != 0.0f){
            norm += d_out_ptr[i];
        }
    }
    shared_max_norm[idx] = norm;
    __syncthreads();

    // Reduce to find the total norm in shared memory
    for (int s = stride / 2; s > 0; s >>= 1) {
        if (idx < s && idx + s < THREADS_PER_BLOCK)
        {
            shared_max_norm[idx] += shared_max_norm[idx + s];
        }
        __syncthreads();
    }

    if (idx == 0) {
        norm_glb[blockIdx.x] = shared_max_norm[0]; // Total norm for the block
        // __threadfence();
        // is_norm_val_set[blockIdx.x] = true;
        int ticket=atomicAdd(&total_inc, 1);
        if (ticket == gridDim.x-1) {
            is_norm_val_set_g = true; // Set the global flag to indicate max value is set for the block
        }
    }
    // __syncthreads();

    while(!is_norm_val_set_g);
    for (int i = gridDim.x/2; i > 0; i >>= 1) {
        if (ix < i && ix + i < N_BLOCKS) {
            // while (!(is_norm_val_set[ix] && is_norm_val_set[ix + i]))
            norm_glb[ix]+= norm_glb[ix+i];  
        }
        __syncthreads();
    }

    if (ix == 0) {
        // __threadfence();
        is_norm_val_set_g = false; // Set the global flag to indicate max value is set for the block
    }

    while(is_norm_val_set_g);
    norm = norm_glb[0]; // Total norm across all blocks

    // if (ix == 0 || ix == 1025) {
    //     printf("Total norm for block %d: %.9f %.9f\n", blockIdx.x, norm_glb[0], norm_glb[1]);
    // }

    // Calculate softmax
    for (int i = idx; i < *N_loops * blockDim.x; i += stride) {
        d_out_ptr[i] /= norm; // Normalize the exponentiated values
    }

}