// #include "softmax_cuda.cuh"
#include "softmax_cuda_fused.cuh"

using namespace std;

// Thread-safe random number generator
thread_local std::mt19937 generator(std::random_device{}());

// Generate random numbers for a part of the array
void drand(float *arr, int size)
{
    uniform_real_distribution<float> distribution(5.0f, 10.0f);
    for (int i = 0; i < size; i++)
    {
        arr[i] = distribution(generator);
    }
}

int main(int argc, char *argv[])
{
    int* N_loops = nullptr;

    
    cudaError_t err;
    // Allocate managed memory for input and output arrays
    float *din = nullptr;
    float *dout = nullptr;
    int zero_pad_n = 0; // Variable to hold the number of elements for zero padding
 
    err = cudaMallocManaged(&N_loops, sizeof(int));
    
    if (err != cudaSuccess)
    {
        cerr << "Managed memory allocation failed!" << endl;
        return -1;
    }

    // *N_loops=32;

    // if (argc == 2)
    // {
    //     N = atoi(argv[1]);
    //     if (N <= 0 )
    //     {
    //         fprintf(stderr, "Invalid input. N should be a positive integer.\n");
    //         return -1;
    //     }
    // }


    int grid = N_BLOCKS;

    *N_loops =  (N + grid* THREADS_PER_BLOCK - 1)/(THREADS_PER_BLOCK*grid);

    if (N % (THREADS_PER_BLOCK * *N_loops) != 0){
        zero_pad_n = (THREADS_PER_BLOCK * *N_loops)*grid-N;// Considering N>THREADS_PER_BLOCK * N_loops
    }

    err = cudaMallocManaged(&din, sizeof(float) * (N+zero_pad_n));
    err = cudaMallocManaged(&dout, sizeof(float) * (N+zero_pad_n));
    if (err != cudaSuccess)
    {
        cerr << "Managed memory allocation failed!" << endl;
        return -1;
    }

    if (zero_pad_n!=0){
        memset(din+N,0,sizeof(float)*zero_pad_n);
        memset(dout+N,0,sizeof(float)*zero_pad_n);
    }


    // Fill input array using 4 CPU threads
    thread threads[4];
    int chunk[4];
    fill(chunk, chunk + 3, N / 4);
    chunk[3] = N / 4 + N % 4;

    for (int i = 0; i < 4; ++i)
    {
        threads[i] = thread(drand, din + (i * N) / 4, chunk[i]);
    }

    for (int i = 0; i < 4; ++i)
    {
        threads[i].join();
    }

    // Copy input data to GPU
    err = cudaMemPrefetchAsync(din, sizeof(float) * (N+zero_pad_n), 0);

    if (err != cudaSuccess)
    {
        cerr << "Memory prefetch failed!" << endl;
        return -1;
    }

    float max_elem = FLT_MIN;
    float norm = 0.0f;             // per row norm
    float *exp_cpu = new float[N]; // CPU array for exponentiation
    if (!exp_cpu)
    {
        cerr << "Host memory allocation failed!" << endl;
        return -1;
    }

    // Calculate max and norm on CPU
    for (int i = 0; i < N; i++)
    {
        if (din[i] > max_elem)
        {
            norm *= expf(max_elem - din[i]);
            max_elem = din[i];
        }
        norm += expf(din[i] - max_elem);
    }

    
    for (int i = 0; i < N; i++)
    {
        exp_cpu[i] = expf(din[i] - max_elem) / norm;
    }

    cout<< "Max element: " <<setprecision(9)<< max_elem << endl;
    cout<< "Norm: " <<setprecision(9)<< norm << endl;

    // Kernel launch parameters
 

    dim3 block(THREADS_PER_BLOCK);


    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    softmax_fused_1sc_v2<<<grid, block>>>(din, dout,N_loops);

    err = cudaDeviceSynchronize();

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9;
    cout << "Time taken for GPU computation: "
         << elapsed_time << " seconds" << endl;

    if (err != cudaSuccess)
    {
        cerr << "Device synchronization failed!" << endl;
        return -1;
    }

    // CPU simulation of softmax
    float cum_abs_err = 0.0f;
    float max_abs = FLT_MIN;

    // cout<<"sum_arr_host: "<<sum_arr_host<<endl;
    // cout<<"sum_arr_gpu: "<<*sum_arr<<" sum_arr_cpu: "<<sum_arr_host<<endl;
    for (int i = 0; i < N; i++)
    {
        cum_abs_err += fabs(dout[i] - exp_cpu[i]);
        max_abs = fmax(max_abs, fabs(dout[i] - exp_cpu[i]));
    }

    cout << "cumm_abs_error: " << cum_abs_err << endl;
    cout << "max_abs_err: " << max_abs << endl;

    // Cleanup
    delete[] exp_cpu;

    cudaFree(din);
    cudaFree(dout);

    return 0;
}