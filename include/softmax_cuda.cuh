#ifndef SOFTMAX_CUDA_CUH
#define SOFTMAX_CUDA_CUH
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


__global__ void softmax_fused(float* resd, const float* xd, const int M,const int N)

#endif // SOFTMAX_CUDA_CUH