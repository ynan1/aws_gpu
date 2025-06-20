#include "softmax_cuda.cuh"

#define N 33554432  // Correctly define as integer
#define THREADS_PER_BLOCK 1024

using namespace std;

// Thread-safe random number generator
thread_local std::mt19937 generator(std::random_device{}());

// Generate random numbers for a part of the array
void drand(float* arr, int size) {
    uniform_real_distribution<float> distribution(5.0f, 10.0f);
    for (int i = 0; i < size; i++) {
        arr[i] = distribution(generator);
    }
}

//Kernel to find the softmax
//__device__ float sum_arr = 0.0f; // Initialize sum_arr in device memory
__device__ float max_val[THREADS_PER_BLOCK]; // Pointer to max_val in device memory

__global__ void row_max( const float* d_in,const int N_blocks) {
    int idx = threadIdx.x;
    if (idx >= THREADS_PER_BLOCK) return; // Ensure we don't access out of bounds
    //__shared__ float shared_max[THREADS_PER_BLOCK];
    float max_l = -FLT_MAX;
    // Find the maximum value per thread block
    for (int i = 0; i < N_blocks; i++) {
        int col = idx * blockDim.x + i;
        max_l = fmaxf(max_l, d_in[col]);
    }
    //printf("max_l: %f\n", max_l); // Debugging output
    //shared_max[idx] = max_l;
    //__syncthreads();
    max_val[idx] = max_l;
}

__global__ void exponent(float* d_out, const float* d_in) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float sum_arr_shd; // Shared memory for sum
    if (idx >= N) return; // Ensure we don't access out of bounds
    d_out[idx] = expf(d_in[idx] - max_val[blockIdx.x]);
    if (threadIdx.x == 0) sum_arr_shd = 0.0f; // Initialize shared sum to zero
    __syncthreads();
    // Atomic addition to shared sum
    atomicAdd(&sum_arr_shd, d_out[idx]);
    __syncthreads();
    //printf(" sum_arr_shd: %f\n", sum_arr_shd); // Debugging output
    // Divide by the sum to normalize
    if (sum_arr_shd > 0.0f) {
        d_out[idx] /= sum_arr_shd;
    }else {
        d_out[idx] = 0.0f; // Avoid division by zero
    }
}

int main(int argc,char* argv[]) {

    cudaError_t err;
    // Allocate managed memory for input and output arrays
    float* din= nullptr;
    float* dout= nullptr;
    //float* max_val = nullptr;

    err=cudaMallocManaged(&din, sizeof(float) * N);
    err=cudaMallocManaged(&dout, sizeof(float) * N);
    //err=cudaMallocManaged(&max_val, sizeof(float) * THREADS_PER_BLOCK);

    if (err != cudaSuccess) {
        cerr << "Managed memory allocation failed!" << endl;
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

    // Copy input data to GPU
    err=cudaMemPrefetchAsync(din, sizeof(float) * N, 0);

    int M=THREADS_PER_BLOCK; //rows
    int N_blocks=(N+M-1)/M; //cols


    float* max_elem= new float[M];//per row max
    float* norm= new float[M];//per row norm
    float* exp_cpu = new float[N]; // CPU array for exponentiation
    if (!max_elem || !norm || !exp_cpu) {
        cerr << "Host memory allocation failed!" << endl;
        return -1;
    }
    
    // Initialize max_elem and norm
    fill(max_elem, max_elem + M, FLT_MIN);
    fill(norm, norm + M, 0.0f);
    
    // Calculate max and norm on CPU
    for (int i=0;i<M;i++){
        int start = i * N_blocks;
        int end = min(start + N_blocks, N);
        for (int j = start; j < end; j++){
            max_elem[i] = fmax(max_elem[i], din[j]);
        }   
    }
    
    for (int i = 0; i < N; i++) {
        //cout<< max_elem[i / M] << " "<<endl;
        exp_cpu[i] = expf(din[i] - max_elem[i / N_blocks]);
        norm[i / N_blocks] += exp_cpu[i];
    }
    //cout<<N/M<<endl;
    
    for (int i=0;i<N;i++){
        exp_cpu[i] = exp_cpu[i] / norm[i / N_blocks];
    }

    //cout<<norm[0]<<" "<<norm[M-1]<<endl;
    
    
    
    
    if (err != cudaSuccess) {
        cerr << "Memory prefetch failed!" << endl;
        return -1;
    }

    //Kernel launch parameters
    dim3 block(THREADS_PER_BLOCK);
    //cudaStream_t stream[3];
    /*
    for (int i = 0; i < 3; ++i) {
        err=cudaStreamCreate(&stream[i]);
        if (err != cudaSuccess) {
            cerr << "Stream creation failed!" << endl;
            return -1;
        }
    }*/



    row_max<<<1, block>>>(din, N_blocks);

    exponent<<<N_blocks, block>>>(dout, din);

   err=cudaDeviceSynchronize();

    if (err != cudaSuccess) {
         cerr << "Device synchronization failed!" << endl;
         return -1;
    }

    //CPU simulation of softmax
    float cum_abs_err=0.0f;
	float max_abs=FLT_MIN;

    //cout<<"sum_arr_host: "<<sum_arr_host<<endl;
    //cout<<"sum_arr_gpu: "<<*sum_arr<<" sum_arr_cpu: "<<sum_arr_host<<endl;
    for (int i = 0; i < N; i++) {
        cum_abs_err += fabs(dout[i] - exp_cpu[i]);
        max_abs = fmax(max_abs, fabs(dout[i] - exp_cpu[i]));
    }

    cout << "cumm_abs_error: " << cum_abs_err << endl;
    cout << "max_abs_err: " << max_abs << endl;

// Cleanup
    delete[] max_elem;
    delete[] norm;
    delete[] exp_cpu;

    cudaFree(din);
    cudaFree(dout);
    cudaFree(max_val);
    /*
    for (int i = 0; i < 3; ++i)
        cudaStreamDestroy(stream[i]);
    */
    return 0;
}
