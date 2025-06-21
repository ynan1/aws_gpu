#include "softmax_cuda.cuh"



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




int main() {

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

    softmax_fused<<<grid, block>>>(dout, din, M, N);

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
    cudaFree(max_val);
    /*
    for (int i = 0; i < 3; ++i)
        cudaStreamDestroy(stream[i]);
    */
    return 0;
}