#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    }

// Softmax kernel (row-wise) for 1D array interpreted as matrix
__global__ void softmax_kernel(float* data, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float max_val = data[row * cols];
        for (int j = 1; j < cols; ++j)
            max_val = fmaxf(max_val, data[row * cols + j]);
        float sum = 0.0f;
        for (int j = 0; j < cols; ++j) {
            data[row * cols + j] = expf(data[row * cols + j] - max_val);
            sum += data[row * cols + j];
        }
        for (int j = 0; j < cols; ++j)
            data[row * cols + j] /= sum;
    }
}

int main() {
    using Element = float;
    int M = 4, K = 8, N = 8; // C = A (MxK) * B (KxN)

    // Allocate unified memory as 1D arrays
    std::vector<Element> h_A(M * K), h_B(K * N);
    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<Element>(rand() % 10);
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<Element>(rand() % 10);

    Element *A, *B, *C;
    CUDA_CHECK(cudaMallocManaged(&A, M * K * sizeof(Element)));
    CUDA_CHECK(cudaMallocManaged(&B, K * N * sizeof(Element)));
    CUDA_CHECK(cudaMallocManaged(&C, M * N * sizeof(Element)));

    // Copy data from host 1D arrays to device
    CUDA_CHECK(cudaMemcpy(A, h_A.data(), M * K * sizeof(Element), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B, h_B.data(), K * N * sizeof(Element), cudaMemcpyHostToDevice));

    // CUTLASS GEMM setup (operates on 1D arrays interpreted as matrices)
    using Gemm = cutlass::gemm::device::Gemm<
        Element, cutlass::layout::RowMajor,
        Element, cutlass::layout::RowMajor,
        Element, cutlass::layout::RowMajor
    >;

    Gemm gemm_op;
    Gemm::Arguments args(
        {M, N, K},
        {A, K},
        {B, N},
        {C, N},
        {C, N},
        {1.0f, 0.0f}
    );

    cutlass::Status status = gemm_op(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS GEMM failed\n";
        return -1;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Apply softmax row-wise to C (1D array interpreted as MxN matrix)
    int threads = 128;
    int blocks = (M + threads - 1) / threads;
    softmax_kernel<<<blocks, threads>>>(C, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Print result
    std::cout << "Softmax output:" << std::endl;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j)
            std::cout << C[i * N + j] << " ";
        std::cout << std::endl;
    }

    cudaFree(A); cudaFree(B); cudaFree(C);
    return 0;
}