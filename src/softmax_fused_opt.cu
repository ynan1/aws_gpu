#include "softmax_cuda_fused.cuh"

//Kernel to find the softmax
__device__ float max_val[THREADS_PER_BLOCK]; // Pointer to max_val in device memory
__device__ bool flag_fin[THREADS_PER_BLOCK];
__device__ bool flag_fin_exp[32768*THREADS_PER_BLOCK];

__global__ void softmax_fused_opt( const float* d_in,float* d_out,const int& N_blocks) {
    if (threadIdx.x >= THREADS_PER_BLOCK) return; // Ensure we don't access out of bounds
    int idx_g = blockIdx.x * blockDim.x + threadIdx.x;

    if (blockIdx.x < THREADS_PER_BLOCK){
    int idx = threadIdx.x; // Thread index within the block
    __shared__ float shared_max[THREADS_PER_BLOCK];
    float* d_in_ptr = const_cast<float*>(d_in + blockIdx.x * N_blocks); // Cast to non-const pointer
    float max_l = -FLT_MAX;
    int stride = blockDim.x; // Number of threads in a block

    // Find the maximum value per thread block
    for (int i = idx; i < N_blocks; i+= stride) {
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
        flag_fin[blockIdx.x] = true;
    }

    }
    
    while (!flag_fin[idx_g/gridDim.x]);
    d_out[idx_g] = expf(d_in[idx_g] - max_val[idx_g/gridDim.x]); // Normalize by the maximum value for the block
    __threadfence();
    flag_fin_exp[idx_g] = true;


    if (blockIdx.x >= THREADS_PER_BLOCK) return;

    // Initialize shared memory for sum
    int stride = blockDim.x; // Number of threads in a block
    int ix= threadIdx.x; // Thread index within the block
    __shared__ float norm_shd[THREADS_PER_BLOCK]; // Shared memory for sum
    float* dout_ptr = d_out + blockIdx.x * N_blocks; // Pointer to the output array for this block
    float sum=0.0f;

    // Calculate the sum of exponentials in this block
    for (int i = ix; i < N_blocks; i += stride) {
        while(!flag_fin_exp[i+blockIdx.x * N_blocks]);
        sum+= dout_ptr[i];
    }
    norm_shd[ix] = sum; // Store the sum in shared memory
    __syncthreads(); // Ensure all threads have completed before proceeding
    

    // Reduce to find the total sum in shared memory
    for (int s = stride / 2; s > 0; s >>= 1) {
        if (ix < s && ix + s < THREADS_PER_BLOCK) {
            norm_shd[ix] += norm_shd[ix + s];
        }
        __syncthreads();
    }

    // Divide by the sum to normalize
    for(int i = ix; i < N_blocks; i += stride) {
        dout_ptr[i] = dout_ptr[i] / norm_shd[0]; // Normalize by the total sum
    }
}