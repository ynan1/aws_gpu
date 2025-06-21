#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void softmax_fused(float* resd, const float* xd, const int M,const int N) {
     // max and norm reduction will happen in shared memory (static)
    __shared__ float smem[1024];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    // edge condition (we don't process further)
    if (row >= M) return;

    float* input_row = xd + row * N;
    float* output_row = resd + row * N;
    float local_max = -INFINITY;
    float local_norm = 0.0f;

    // compute local max and norm for each thread
    // and then finally have a sync barrier before moving on
    for (int i = tid; i < N; i += blockDim.x) {
        float x = input_row[i];
        if (x > local_max) {
            local_norm *= expf(local_max - x);
            local_max = x;
        }
        local_norm += expf(x - local_max);
    }
    __syncthreads();

    // each thread will have its own local max
    // we store it in the tid of the shared memory
    smem[tid] = local_max;
    __syncthreads();

    // block-level reduction in O(log(N)) time over all threads
    // is faster than linear reduction over all threads
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            smem[tid] = max(smem[tid], smem[tid + stride]);
        }
        // sync barrier before next iteration to ensure correctness
        __syncthreads();
    }

    // the first element after max reduction from all threads
    // will contain the global max for the row
    float row_max = smem[0];
    __syncthreads();

    // each thread will have its own local norm
    // we will store the corrected local norm in the shared memory
    // again, exploits property of exponentials
    smem[tid] = local_norm * expf(local_max - row_max);
    __syncthreads();

    // sum reduction similar to above for global norm factor
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    float row_norm = smem[0];
    __syncthreads();

    // finally, compute softmax
    for (int i = tid; i < N; i += blockDim.x) {
        output_row[i] = expf(input_row[i] - row_max) / row_norm;
    }
}

