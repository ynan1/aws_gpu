#ifndef SOFTMAX_CUDA_H
#define SOFTMAX_CUDA_H

# pragma once
#include <iostream>
#include <random>
#include <ctime>
#include <thread>
#include <float.h>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <functional>
#include <atomic>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


__global__ void row_max( const float* d_in,const int N_blocks) ;

__global__ void exponent(float* d_out, const float* d_in) ;

#endif // SOFTMAX_CUDA_H