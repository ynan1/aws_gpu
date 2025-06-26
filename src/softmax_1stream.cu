#include "softmax_seq.cuh"

#define N 33554432  // Correctly define as integer
#define THREADS_PER_BLOCK 1024

using namespace std;

// Thread-safe random number generator
thread_local std::mt19937 generator(std::random_device{}());

// Generate random numbers for a part of the array
void drand(float* arr, int size) {
    normal_distribution<float> distribution(0.0f, 1.0f);
    for (int i = 0; i < size; i++) {
        arr[i] = distribution(generator);
    }
}

//Kernel to find the softmax
// __device__ float sum_arr=0.0f; // Initialize sum_arr in device memory
__device__ float max_val[THREADS_PER_BLOCK]; // Pointer to max_val in device memory

__global__ void row_red_max( const float* d_in,const int& N_blocks) {
    if (blockIdx.x >= THREADS_PER_BLOCK) return; // Ensure we don't access out of bounds
    if (threadIdx.x >= THREADS_PER_BLOCK) return; // Ensure we don't access out of bounds
    
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
    }

}

__global__ void exponent(float* d_out, const float* d_in) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return; // Ensure we don't access out of bounds
    d_out[idx] = expf(d_in[idx] - max_val[idx/gridDim.x]); // Normalize by the maximum value for the block
    
}

__global__ void normalize(float* d_out, const int& N_blocks) {
    if (blockIdx.x >= THREADS_PER_BLOCK) return; // Ensure we don't access out of bounds
    if (threadIdx.x >= THREADS_PER_BLOCK) return; // Ensure we don't access out of bounds
    // Initialize shared memory for sum
    int stride = blockDim.x; // Number of threads in a block
    int ix= threadIdx.x; // Thread index within the block
    __shared__ float norm_shd[THREADS_PER_BLOCK]; // Shared memory for sum
    float* dout_ptr = d_out + blockIdx.x * N_blocks; // Pointer to the output array for this block
    float sum=0.0f;

    // Calculate the sum of exponentials in this block
    for (int i = ix; i < N_blocks; i += stride) {
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
        //cout<<max_elem[i] << " "<<endl;   
    }

    
    
    for (int i = 0; i < N; i++) {
        //cout<< max_elem[i / M] << " "<<endl;
        exp_cpu[i] = expf(din[i] - max_elem[i / N_blocks]);
        norm[i / N_blocks] += exp_cpu[i];
    }
    //cout<<N/M<<endl;
    // cout<<"exp: "<<setprecision(7)<<"exp:"<<exp_cpu[78762]<<endl;

    for (int i=0;i<N;i++){
        exp_cpu[i] = exp_cpu[i] / norm[i / N_blocks];
    }

    // cout<<"after normalize exp:"<<fixed<<setprecision(7)<<exp_cpu[76254]<<" norm:"<<norm[76254/N_blocks]<<endl;

    if (err != cudaSuccess) {
        cerr << "Memory prefetch failed!" << endl;
        return -1;
    }

    //Kernel launch parameters
    dim3 block(THREADS_PER_BLOCK);

    int grid=M;

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    row_red_max<<<grid, block>>>(din, N_blocks);

    exponent<<<N_blocks, block>>>(dout, din);

    normalize<<<grid, block>>>(dout, N_blocks);

    err=cudaDeviceSynchronize();

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec)*1e-9;
    cout<< "Time taken for GPU computation: "
         << elapsed_time << " seconds" << endl;

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
   
    return 0;
}
