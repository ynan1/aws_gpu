#pragma once
#include <iostream>
#include <random>
#include <ctime>
#include <thread>
#include <float.h>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <functional>
#include <cmath>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define THREADS_PER_BLOCK 1024

constexpr int N = 33554432;

constexpr int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

constexpr int ceil_log2(unsigned int x) {
    int log = 0;
    unsigned int pow2 = 1;
    while (pow2 < x) {
        pow2 <<= 1;
        log++;
    }
    return log;
}

constexpr int N_BLOCKS = 32;//1 << (ceil_log2(ceil_div(N, THREADS_PER_BLOCK))/2);


__global__ void softmax_fused_opt( const float* d_in,float* d_out,const int& N_blocks);

__global__ void softmax_fused_1sc( const float* d_in,float* d_out,const int* N_loops);