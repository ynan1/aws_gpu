#include <iostream>
#include <random>
#include <ctime>
#include <thread>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define N 1000000  // Correctly define as integer
#define THREADS_PER_BLOCK 256

using namespace std;

// CUDA kernel
__global__ void exponent(float* d_out, const float* d_in) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_out[idx] = expf(d_in[idx]);
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

int main() {
    float* din = new float[N];
    float* dout = new float[N];

    if (!din || !dout) {
        cerr << "Host memory allocation failed!" << endl;
        return -1;
    }

    // Fill input array using 4 CPU threads
    thread threads[4];
    int chunk = N / 4;
    for (int i = 0; i < 4; ++i) {
        threads[i] = thread(drand, din + i * chunk, chunk);
    }
    for (int i = 0; i < 4; ++i) {
        threads[i].join();
    }

    // CUDA device memory
#ifdef USE_MANAGED
    float *gpu_din, *gpu_dout;
    cudaMallocManaged(&gpu_din, sizeof(float) * N);
    cudaMallocManaged(&gpu_dout, sizeof(float) * N);
    memcpy(gpu_din, din, sizeof(float) * N);  // Fill managed memory
#else
    float *gpu_din, *gpu_dout;
    cudaError_t err;
    err = cudaMalloc(&gpu_din, sizeof(float) * N);
    err = cudaMalloc(&gpu_dout, sizeof(float) * N);

    if (err != cudaSuccess) {
        cerr << "GPU memory allocation failed!" << endl;
        return -1;
    }

    cudaMemcpy(gpu_din, din, sizeof(float) * N, cudaMemcpyHostToDevice);
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
    // Launch the kernel (single stream)
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    exponent<<<blocks, THREADS_PER_BLOCK>>>(gpu_dout, gpu_din);
    cudaDeviceSynchronize();

#ifndef USE_MANAGED
    cudaMemcpy(dout, gpu_dout, sizeof(float) * N, cudaMemcpyDeviceToHost);
#endif
#endif

float cum_abs_err;
	float max_abs=0;
	for (int i=0;i<N;i++){
		cum_abs_err+=fabs(dout[i]-expf(din[i]));
		max_abs=fmax(max_abs,fabs(dout[i]-expf(din[i])));			
	}

	cout<<"cumm_abs_error: "<<cum_abs_err<<endl;
	cout<<"max_abs_err: "<<max_abs<<endl;    


// Cleanup
    delete[] din;
    delete[] dout;

#ifndef USE_MANAGED
    cudaFree(gpu_din);
    cudaFree(gpu_dout);
#ifdef USE_STREAM
    for (int i = 0; i < nStreams; ++i)
        cudaStreamDestroy(stream[i]);
#endif
#else
    cudaFree(gpu_din);
    cudaFree(gpu_dout);
#endif

    return 0;
}

