#include <iostream>
#include <random>
#include <ctime>
#include <thread>
#include <float.h>
#include <algorithm>
#include <numeric>
#include <functional>
#include <atomic>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


#define N 1000000000  // Correctly define as integer
#define THREADS_PER_BLOCK 1024


using namespace std;

__device__ float sum_arr= 0.0f; // Initialize sum_arr in device memory

// CUDA kernel
__global__ void max_n(float* max_val, const float* d_in) {
    int idx = threadIdx.x;
    int stride = blockDim.x;
    __shared__ float shared_max[THREADS_PER_BLOCK];
    float max_l = -FLT_MAX;
    //shared_max[idx] = (idx < N) ? d_in[idx] : -FLT_MAX; // Initialize shared memory with input values or -FLT_MAX
    __syncthreads();
    // Find the maximum value per thread block
    for (int i=idx; i < N; i += stride) {
        max_l = fmaxf(max_l, d_in[i]);
    }
    shared_max[idx] = max_l;
    __syncthreads();
    // Reduce to find the maximum value in shared memory
   for (int s = stride / 2; s > 0; s >>= 1) {
       if (idx < s && idx + s < N) {
           shared_max[idx] = fmaxf(shared_max[idx], shared_max[idx + s]);
       }
       __syncthreads();
   }
    // Write the maximum value to global memory
    if (idx == 0) {
        *max_val = shared_max[0];
    }
}

__global__ void exponent(float* d_out, const float* d_in,float* max_val) {
    //2D Block indexing
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_out[idx] = expf(d_in[idx] - *max_val);
    }
    __syncthreads();

    //Add all the values in d_out
    atomicAdd(&sum_arr, d_out[idx]);
    __syncthreads();

    // Normalize the output
    if (idx < N) {
        d_out[idx] /= sum_arr;
    }
}

// Thread-safe random number generator
thread_local std::mt19937 generator(std::random_device{}());

// Generate random numbers for a part of the array
void drand(float* arr, int size) {
    uniform_real_distribution<float> distribution(0.0f, 3.0f);
    for (int i = 0; i < size; i++) {
        arr[i] = distribution(generator);
    }
}

int main(int argc,char* argv[]) {
    #ifdef USE_MANAGED
    float* din= nullptr;
    float* dout= nullptr;
    float* max_val= nullptr; // Initialize max_val to a very small value
    cudaMallocManaged(&max_val, sizeof(float));
    cudaMallocManaged(&din, sizeof(float) * N);
    cudaMallocManaged(&dout, sizeof(float) * N);
    #else
    float* din = new float[N];
    float* dout = new float[N];

    if (!din || !dout) {
        cerr << "Host memory allocation failed!" << endl;
        return -1;
    }

    #endif

    // Fill input array using 4 CPU threads
    thread threads[4];
    int chunk = N / 4;
    for (int i = 0; i < 4; ++i) {
        threads[i] = thread(drand, din + i * chunk, chunk);
    }
    for (int i = 0; i < 4; ++i) {
        threads[i].join();
    }

    float max_val_cpu=*max_element(din,din+N);

    // CUDA device memory
#ifdef USE_MANAGED
    cudaMemPrefetchAsync(din, sizeof(float) * N, 0);
#else
    float *gpu_din, *gpu_dout, *max_val_ptr;
    cudaError_t err;
    err = cudaMalloc(&gpu_din, sizeof(float) * N);
    err = cudaMalloc(&gpu_dout, sizeof(float) * N);
    err = cudaMalloc(&max_val_ptr, sizeof(float));

    if (err != cudaSuccess) {
        cerr << "GPU memory allocation failed!" << endl;
        return -1;
    }

    cudaMemcpy(gpu_din, din, sizeof(float) * N, cudaMemcpyHostToDevice);
    //cudaMemcpy(max_val_ptr, &max_val, sizeof(float), cudaMemcpyHostToDevice);
#endif

    // Use streams if enabled
#ifdef USE_STREAM
    const int nStreams = 3;
    cudaStream_t stream[nStreams];
    int streamSize = N / nStreams;

    for (int i = 0; i < nStreams; ++i)
        cudaStreamCreate(&stream[i]);

    for (int i = 0; i < nStreams; ++i) {
        int offset = i * streamSize;
        cudaMemcpyAsync(gpu_din + offset, din + offset, streamSize * sizeof(float), cudaMemcpyHostToDevice, stream[i]);
        int blocks = (streamSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        exponent<<<blocks, THREADS_PER_BLOCK, 0, stream[i]>>>(gpu_dout + offset, gpu_din + offset);
        cudaMemcpyAsync(dout + offset, gpu_dout + offset, streamSize * sizeof(float), cudaMemcpyDeviceToHost, stream[i]);
    }

    for (int i = 0; i < nStreams; ++i)
        cudaStreamSynchronize(stream[i]);

#else
int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
#ifdef USE_MANAGED
    // Launch the kernel (single stream)
    // Find the max_val of the array
    max_n<<<1, THREADS_PER_BLOCK>>>(max_val, din);
    exponent<<<blocks, THREADS_PER_BLOCK>>>(dout, din, max_val);
    cudaDeviceSynchronize();
    cout<<*max_val<<" "<<max_val_cpu<<endl;
#else
    // Launch the kernel (single stream)
    max_n<<<1,THREADS_PER_BLOCK>>>(max_val_ptr,gpu_din);
    exponent<<<blocks, THREADS_PER_BLOCK>>>(gpu_dout, gpu_din,max_val_ptr);
    cudaMemcpy(dout, gpu_dout, sizeof(float) * N, cudaMemcpyDeviceToHost);
    //cudaMemcpy(&max_val, max_val_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    //cout<<max_val<<" "<<max_val_cpu<<endl;
#endif
#endif

    float cum_abs_err=0.0f;
	float max_abs=FLT_MIN;
    float sum_arr_host=0.0f;
    float* exp_arr=new float[N];

    for (int i = 0; i < N; ++i) {
        exp_arr[i] = expf(din[i] - max_val_cpu);
        sum_arr_host += exp_arr[i];
    }

	for (int i=0;i<N;i++){
        float exp= exp_arr[i]/sum_arr_host;
		cum_abs_err+=fabs(dout[i]-exp);
		max_abs=fmax(max_abs,fabs(dout[i]-exp));
	}

	cout<<"cumm_abs_error: "<<cum_abs_err<<endl;
	cout<<"max_abs_err: "<<max_abs<<endl;    


// Cleanup
    delete[] exp_arr;

#ifdef USE_MANAGED
    cudaFree(din);
    cudaFree(dout);
    cudaFree(max_val);
#else
    delete[] din;
    delete[] dout;
    cudaFree(gpu_din);
    cudaFree(gpu_dout);
    cudaFree(max_val_ptr);
#endif


#ifdef USE_STREAM
    for (int i = 0; i < nStreams; ++i)
        cudaStreamDestroy(stream[i]);
#endif

    return 0;
}

