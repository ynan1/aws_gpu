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
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define N_BLOCKS CEIL_DIV(1000000,(THREADS_PER_BLOCK*32))


__global__ void softmax_fused_opt( const float* d_in,float* d_out,const int& N_blocks);

__global__ void softmax_fused_1sc( const float* d_in,float* d_out,const int& N_loops);