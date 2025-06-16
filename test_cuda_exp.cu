#include<iostream>
#include<random>
#include<time.h>
#include<cuda_runtime.h>

#define N 1e6
using namespace std;
// Thread-safe random number generator
thread_local std::mt19937 generator(std::random_device{}());
// Function to generate a random float in the range [0, 1)
void drand48( float* arr, int size) {
    std::uniform_real_distribution<float> distribution(0.0f, 3.0f);
    for (int i = 0; i < size; i++) {
        arr[i] = distribution(generator);
    }
}
__global__ void exponent(float* d_out, const float* d_in){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return; // Ensure we don't access out of bounds
    d_out[idx]= expf(d_in[idx]);
}

int main() {
    srand(time(0));
    float* din = new float[N];
    float* dout = new float[N];

    if (din == nullptr || dout == nullptr) {
        cerr << "Memory allocation failed!" << endl;
        delete[] din;
        delete[] dout;
        return -1;
    }
    // Initialize input data with random values

    thread local_random_engine[4];
    // Use a random number generator to fill the input array
    // Note: In a real application, you might want to use a more robust random number generator
    // and ensure that the random numbers are uniformly distributed.
    // Here we use a simple approach for demonstration purposes.
    // Fill the input array with random values
    for (int i=0;i<4;i++){
        float* din_ptr = din + i * (N / 4);
        local_random_engine[i]=thread(drand48, din_ptr, N / 4);
    }
    for (int i=0;i<4;i++){
        local_random_engine[i].join();
    }
    
    // cudamalloc
    
    #ifdef USE_MANAGED //unified memory
    cudaMallocManaged(din, sizeof(float) * N);  
    cudaMallocManaged(dout, sizeof(float) * N);
    #else
    float *gpu_din,*gpu_dout;
    cudaMalloc((void**)&gpu_din,sizeof(float)*N);
    cudaMalloc((void**)&gpu_dout,sizeof(float)*N);

    if (gpu_din == nullptr || gpu_dout == nullptr) {
        cerr << "GPU memory allocation failed!" << endl;
        delete[] din;
        delete[] dout;
        cudaFree(gpu_din);
        cudaFree(gpu_dout);
        return -1;
    }

    #endif

    //copy cpu memory to gpu memory
    #ifdef USE_STREAM
    // If using streams, create streams for asynchronous operations
    cudaStream_t stream[3];
    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);
    cudaStreamCreate(&stream[2]);
    #elifndef USE_MANAGED
    cudaMemcpy(gpu_din,din,sizeof(float)*N,cudaMemcpyHostToDevice);
    #endif

    //cudaMemcpy(gpu_dout,dout,sizeof(float)*N);


    //dim3 threadsPerBlock(10);
    
    #ifdef USE_MANAGED
    // Launch the kernel with a number of blocks and threads
    exponent<<<1, 1000>>>(dout, din);
    cudaDeviceSynchronize(); // Wait for the kernel to finish
    #else
    // Wait for the GPU to finish before accessing on host
    #ifndef USE_STREAM
    exponent<<<10,100>>>(gpu_dout,gpu_din);
    //copy gpu memory to cpu memory
    cudaMemcpy(dout,gpu_dout,sizeof(float)*N,cudaMemcpyDeviceToHost);
    #elif defined(USE_STREAM)
    // Launch the kernel with streams
    int size=N*sizeof(float)/3;
    for (int i = 0; i < 3; i++) {
        int offset = i*N/nStreams;
        cudaMemcpyAsync(gpu_din + offset, din + offset, size, cudaMemcpyHostToDevice, stream[i]);
        exponent<<<(size + 99) / 100, 100, 0, stream[i]>>>(gpu_dout + offset, gpu_din + offset);
    }
    // Copy results back to host
    for (int i = 0; i < 3; i++) {
        // Ensure the kernel has completed before copying results back
        int offset = i*N/nStreams;
        cudaStreamSynchronize(stream[i]);
        cudaMemcpyAsync(dout + offset, gpu_dout + offset, size, cudaMemcpyDeviceToHost, stream[i]);
    }
    #endif
    


    for (int i=0;i<N;i++){
        cout<<dout[i]<<endl;
    }

    /*
    delete[] din;
    delete[] dout;
    cudaFree(gpu_din);
    cudaFree(gpu_dout);
    */
    return;
}
