#include "softmax_cuda_fused.cuh"

//Kernel to find the softmax
__device__ float max_val[N_BLOCKS]; // Pointer to max_val in device memory
__device__ bool is_max_val_set[N_BLOCKS];

__global__ void softmax_fused_1sc( const float* d_in,float* d_out,const int& N_loops) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float shared_max[THREADS_PER_BLOCK];
    float* d_in_ptr = const_cast<float*>(d_in + blockIdx.x * N_loops * blockDim.x); // Cast to non-const pointer
    float max_l = -FLT_MAX;
    int stride = blockDim.x; // Number of threads in a block

    // Find the maximum value per thread block
    for (int i = idx; i < N_loops * blockDim.x; i+= stride) {
        max_l = fmaxf(max_l, d_in_ptr[i]);
    }
    shared_max[idx] = max_l; // Store the maximum value in shared memory
    __syncthreads();

    // Reduce to find the maximum value in shared memory
    for (int s = stride / 2; s > 0; s >>= 1) {
        if (idx < s && idx + s < THREADS_PER_BLOCK) {
            shared_max[idx] = fmaxf(shared_max[idx], shared_max[idx + s]);
        }
        __syncthreads();
    }

    // Write the maximum value to global memory
    if (idx == 0) { // Only the first thread writes to global memory
        max_val[blockIdx.x] = shared_max[0]; // Store the maximum value for this block
        __threadfence();
        is_max_val_set[blockIdx.x] = true;
    }
    __syncthreads();

    while(!is_max_val_set[blockIdx.x]);
    if (threadIdx.x ==0){
        printf("Max value for block %d: %f\n", blockIdx.x, max_val[blockIdx.x]);
    }
}