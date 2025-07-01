#include "softmax_cuda_fused.cuh"

//Kernel to find the softmax
__device__ float max_val[N_BLOCKS]; // Pointer to max_val in device memory
__device__ float norm_blk[N_BLOCKS];
__device__ volatile bool is_max_val_set[N_BLOCKS];
__device__ float norm_glb;

__global__ void softmax_fused_1sc_v2( const float* d_in,float* d_out,const int* N_loops) {
    int idx = threadIdx.x;
    int ix= blockIdx.x * blockDim.x + idx;
    __shared__ float shared_max_norm[THREADS_PER_BLOCK];
    // __shared__ float shared_norm[THREADS_PER_BLOCK];
    float* d_in_ptr = const_cast<float*>(d_in + blockIdx.x * *N_loops * blockDim.x); // Cast to non-const pointer
    float max_l = -FLT_MAX;
    float norm_l = 0.0f;

    // Find the maximum value per thread block
    for (int i = *N_loops*idx; i < *N_loops *(idx+1); i++) {
        if (d_in_ptr[i] > max_l) {
            norm_l*= expf(max_l - d_in_ptr[i]);
            max_l = d_in_ptr[i];
        }
        norm_l += expf(d_in_ptr[i] - max_l);
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

    if (idx == 0){
        max_val[blockIdx.x] = shared_max_norm[0]; // Store the maximum value for this block globally
    }

    float max_blk= shared_max_norm[0]; // The maximum value for this block
    __syncthreads();

    // Calculate the norm for this block
    shared_max_norm[idx]=norm_l*expf(max_l-max_blk);
    __syncthreads();

    for (int s = stride / 2; s > 0; s >>= 1) {
        if (idx < s && idx + s < THREADS_PER_BLOCK) {
            shared_max_norm[idx] += shared_max_norm[idx + s];
        }
        __syncthreads();
    }

    if (idx == 0) {
        norm_blk[blockIdx.x] = shared_max_norm[0]; // Store the norm for this block globally
        is_max_val_set[blockIdx.x] = true; // Mark that the norm is set for this block
    }

    if (ix < gridDim.x){
    //Reduce block wise max to global max
    for (int s=gridDim.x/2; s>0; s>>1){
        if (ix < s && ix + s < gridDim.x) {
            while(!(is_max_val_set[ix] && is_max_val_set[ix+s]));
            max_val[ix]=fmaxf(max_val[ix],max_val[ix+s]);
        }
        __syncthreads();
    }

    max_l=max_val[0];//Global max is set for each thread
    __syncthreads();

    //Adjust global max for the per block norms
    norm_blk[ix]*=expf(max_blk - max_l);
    __syncthreads();

    //Reduce norm to global norm
    for (int s=gridDim.x/2; i>0; i>>1){
        if (ix<s && ix+s <gridDim.x){
            norm_blk[ix]+=norm_blk[ix+s];
        }
    }

    if (ix==0){
        is_max_val_set[0] = false;
    }

    }

    float* d_out_ptr = (d_out + blockIdx.x * *N_loops * blockDim.x); // Cast to non-const pointer

    while(is_max_val_set[0]);
    for (int i = *N_loops*idx; i < *N_loops *(idx+1); i++) {
        d_out_ptr[i]=expf(d_in_ptr[i] - max_val[0])/norm_blk[0];
    }

}